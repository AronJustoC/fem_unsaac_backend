"""
envelope.py - Análisis de Envolvente y Transformada de Hilbert
==============================================================
Detección de impactos, análisis de envolvente, periodicidad.
Especialmente útil para detectargolpes, juntas y paso de ejes en puentes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple, List
import numpy as np
from scipy import signal


@dataclass
class EnvelopeResult:
    """
    Resultado del análisis de envolvente.
    
    Attributes:
        time: Array de tiempo
        envelope: Envolvente de la señal
        analytic_signal: Señal analítica de Hilbert
        peak_times: Tiempos de picos en la envolvente
        peak_amplitudes: Amplitudes de los picos
    """
    time: np.ndarray
    envelope: np.ndarray
    analytic_signal: Optional[np.ndarray] = None
    peak_times: Optional[np.ndarray] = None
    peak_amplitudes: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        """Convierte a diccionario serializable."""
        return {
            'time': self.time.tolist(),
            'n_samples': len(self.envelope),
            'peak_times': self.peak_times.tolist() if self.peak_times is not None else [],
            'n_peaks': len(self.peak_times) if self.peak_times is not None else 0,
        }


@dataclass
class ImpactResult:
    """
    Resultado del análisis de impactos.
    
    Attributes:
        impact_times: Tiempos de los impactos
        impact_amplitudes: Amplitudes de los impactos
        avg_interval: Intervalo promedio entre impactos
        periodicity_score: Score de periodicidad (0-1)
        probable_source: Fuente probable del impacto
    """
    impact_times: np.ndarray
    impact_amplitudes: np.ndarray
    avg_interval: float = 0.0
    periodicity_score: float = 0.0
    probable_source: str = ""
    
    def to_dict(self) -> dict:
        """Convierte a diccionario serializable."""
        return {
            'impact_times': self.impact_times.tolist(),
            'impact_amplitudes': self.impact_amplitudes.tolist(),
            'n_impacts': len(self.impact_times),
            'avg_interval_s': self.avg_interval,
            'periodicity_score': self.periodicity_score,
            'probable_source': self.probable_source,
        }


class EnvelopeAnalyzer:
    """
    Analizador de envolvente para detección de impactos.
    
    Usa la Transformada de Hilbert para obtener la envolvente
    y detectar impactos repetitive en puentes y estructuras.
    """
    
    def __init__(self, fs: float):
        """
        Inicializa el analizador.
        
        Args:
            fs: Frecuencia de muestreo en Hz
        """
        self.fs = fs
    
    def compute_envelope(
        self,
        amplitude: np.ndarray,
        time: Optional[np.ndarray] = None,
        smooth_window: int = 50,
        apply_bandpass: bool = False,
        bandpass_range: Optional[Tuple[float, float]] = None,
    ) -> EnvelopeResult:
        """
        Calcula la envolvente de la señal.
        
        Args:
            amplitude: Array de amplitudes
            time: Array de tiempo (opcional)
            smooth_window: Ventana de suavizado
            apply_bandpass: Si aplicar bandpass antes de Hilbert
            bandpass_range: Rango de frecuencias para bandpass
            
        Returns:
            EnvelopeResult con la envolvente
        """
        # Aplicar bandpass si se solicita
        if apply_bandpass and bandpass_range is not None:
            amplitude = self._bandpass_filter(amplitude, bandpass_range)
        
        # Transformada de Hilbert para obtener la envolvente
        analytic_signal = signal.hilbert(amplitude)
        envelope = np.abs(analytic_signal)
        
        # Suavizar la envolvente
        if smooth_window > 1:
            envelope = self._smooth_envelope(envelope, smooth_window)
        
        return EnvelopeResult(
            time=time if time is not None else np.arange(len(amplitude)) / self.fs,
            envelope=envelope,
            analytic_signal=analytic_signal,
        )
    
    def compute_envelope_from_filtered(
        self,
        amplitude: np.ndarray,
        time: np.ndarray,
        low_freq: float = 10.0,
        high_freq: float = 100.0,
        filter_order: int = 4,
    ) -> EnvelopeResult:
        """
        Calcula la envolvente filtrando primero.
        
        Ruta típica para análisis de impactos en puentes:
        1. Filtro bandpass (ej: 10-40 Hz o 20-80 Hz)
        2. Hilbert transform
        3. Detección de picos
        
        Args:
            amplitude: Array de amplitudes
            time: Array de tiempo
            low_freq: Frecuencia baja del bandpass
            high_freq: Frecuencia alta del bandpass
            filter_order: Orden del filtro
            
        Returns:
            EnvelopeResult
        """
        from .filters import SignalFilter
        signal_filter = SignalFilter(self.fs)
        
        filtered = signal_filter.apply(
            amplitude,
            filter_type='bandpass',
            order=filter_order,
            cutoff_freq=(low_freq, high_freq),
        )
        
        return self.compute_envelope(
            filtered,
            time=time,
            smooth_window=int(self.fs * 0.05),  # 50 ms de suavizado
        )
    
    def detect_impacts(
        self,
        envelope: EnvelopeResult,
        threshold: Optional[float] = None,
        min_interval: float = 0.05,
        prominence_ratio: float = 0.3,
    ) -> ImpactResult:
        """
        Detecta impactos a partir de la envolvente.
        
        Args:
            envelope: EnvelopeResult de compute_envelope
            threshold: Umbral para detectar picos
            min_interval: Intervalo mínimo entre impactos (s)
            prominence_ratio: Ratio mínimo de prominencia
            
        Returns:
            ImpactResult con los impactos detectados
        """
        e = envelope.envelope
        
        # Definir umbral
        if threshold is None:
            threshold = prominence_ratio * np.max(e)
        
        # Encontrar picos
        peaks, properties = signal.find_peaks(
            e,
            height=threshold,
            prominence=threshold * 0.5,
        )
        
        peak_times = envelope.time[peaks] if len(peaks) > 0 else np.array([])
        peak_amplitudes = e[peaks]
        
        # Calcular intervalos entre impactos
        if len(peak_times) >= 2:
            intervals = np.diff(peak_times)
            avg_interval = float(np.mean(intervals))
            std_interval = float(np.std(intervals))
            
            # Score de periodicidad (1 = perfectamente periódico)
            if avg_interval > 0:
                coefficient_of_variation = std_interval / avg_interval
                periodicity_score = max(0, 1 - coefficient_of_variation)
            else:
                periodicity_score = 0
        else:
            intervals = np.array([])
            avg_interval = 0.0
            periodicity_score = 0
        
        # Estimar fuente probable
        probable_source = self._identify_impact_source(avg_interval, periodicity_score)
        
        return ImpactResult(
            impact_times=peak_times,
            impact_amplitudes=peak_amplitudes,
            avg_interval=avg_interval,
            periodicity_score=periodicity_score,
            probable_source=probable_source,
        )
    
    def _identify_impact_source(
        self,
        interval: float,
        periodicity_score: float,
    ) -> str:
        """
        Identifica la fuente probable del impacto basándose en el intervalo.
        
        Args:
            interval: Intervalo promedio en segundos
            periodicity_score: Score de periodicidad
            
        Returns:
            Descripción de la fuente probable
        """
        if periodicity_score < 0.3:
            return "Impactos no periódicos o noise"
        
        if interval < 0.1:
            return "Posible ruido de alta frecuencia o vibración estructural"
        elif interval < 0.3:
            return "Posible paso de ejesvehiculares pequeños"
        elif interval < 0.5:
            return "Posible paso de ejes de camión o vehículo pesado"
        elif interval < 1.0:
            return "Posible paso de trenes o equipo pesado"
        elif interval < 2.0:
            return "Posible carga móvil lenta o impacto de viento"
        else:
            return "Intervalo muy largo, fuente no identificada"
    
    def analyze_periodicity(
        self,
        impact_result: ImpactResult,
    ) -> dict:
        """
        Analiza la periodicidad de los impactos detectados.
        
        Usa autocorrelation y análisis de intervalos.
        
        Args:
            impact_result: ImpactResult
            
        Returns:
            Diccionario con análisis de periodicidad
        """
        times = impact_result.impact_times
        amplitudes = impact_result.impact_amplitudes
        
        if len(times) < 3:
            return {
                'is_periodic': False,
                'reason': 'Insuficientes impactos detectados',
            }
        
        # Autocorrelation de los tiempos
        intervals = np.diff(times)
        
        # FFT de intervalos
        if len(intervals) > 2:
            fft_intervals = np.fft.fft(intervals - np.mean(intervals))
            autocorr_intervals = np.real(np.fft.ifft(fft_intervals * np.conj(fft_intervals)))
            autocorr_intervals = autocorr_intervals / (autocorr_intervals[0] + 1e-10)
        else:
            autocorr_intervals = np.array([1.0])
        
        # Calcular Fourier de la envolvente
        envelope_time = times
        envelope_amp = amplitudes
        
        if len(envelope_amp) > 2:
            fft_env = np.fft.fft(envelope_amp - np.mean(envelope_amp))
            autocorr = np.real(np.fft.ifft(fft_env * np.conj(fft_env)))
            autocorr = autocorr / (autocorr[0] + 1e-10)
        else:
            autocorr = np.array([1.0])
        
        # FFT de la envolvente original
        return {
            'is_periodic': impact_result.periodicity_score > 0.5,
            'periodicity_score': impact_result.periodicity_score,
            'avg_period_s': float(impact_result.avg_interval),
            'source': impact_result.probable_source,
        }
    
    def compute_hilbert_phase(
        self,
        amplitude: np.ndarray,
        filter_band: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Calcula la fase instantánea usando Hilbert.
        
        Args:
            amplitude: Array de amplitudes
            filter_band: Banda de frecuencias a filtrar (opcional)
            
        Returns:
            Array de fase instantánea
        """
        if filter_band is not None:
            from .filters import SignalFilter
            signal_filter = SignalFilter(self.fs)
            amplitude = signal_filter.apply(
                amplitude,
                filter_type='bandpass',
                order=4,
                cutoff_freq=filter_band,
            )
        
        analytic_signal = signal.hilbert(amplitude)
        phase = np.angle(analytic_signal)
        
        # Desenvolver la fase
        phase_unwrapped = np.unwrap(phase)
        
        return phase_unwrapped
    
    def compute_instantaneous_frequency(
        self,
        amplitude: np.ndarray,
        filter_band: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula la frecuencia instantánea.
        
        Args:
            amplitude: Array de amplitudes
            filter_band: Banda de frecuencias a filtrar
            
        Returns:
            Tuple de (tiempo, frecuencia_instantánea)
        """
        phase = self.compute_hilbert_phase(amplitude, filter_band)
        instantaneous_freq = np.diff(phase) * self.fs / (2 * np.pi)
        
        time = np.arange(len(instantaneous_freq)) / self.fs
        
        return time, instantaneous_freq
    
    def _bandpass_filter(
        self,
        amplitude: np.ndarray,
        band: Tuple[float, float],
    ) -> np.ndarray:
        """
        Aplica filtros bandpass para limpiar la señal para Hilbert.
        
        Args:
            amplitude: Array de entrada
            band: Tuple (f_low, f_high)
            
        Returns:
            Señal filtrada
        """
        from .filters import SignalFilter
        signal_filter = SignalFilter(self.fs)
        
        return signal_filter.apply(
            amplitude,
            filter_type='bandpass',
            order=4,
            cutoff_freq=band,
        )
    
    def _smooth_envelope(
        self,
        envelope: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """
        Suaviza la envolvente usando media móvil.
        
        Args:
            envelope: Array de envolvente
            window: Tamaño de vetana
            
        Returns:
            Envolvente suavizada
        """
        kernel = np.ones(window) / window
        smoothed = np.convolve(envelope, kernel, mode='same')
        
        return smoothed
    
    def extract_impact_sequence(
        self,
        time: np.ndarray,
        amplitude: np.ndarray,
        event_time: float,
        window_before: float = 0.1,
        window_after: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrae una secuencia de impactos alrededor de un evento.
        
        Args:
            time: Array de tiempo
            amplitude: Array de amplitudes
            event_time: Tiempo del evento
            window_before: Ventana antes del evento
            window_after: Ventana después del evento
            
        Returns:
            Tuple de (tiempos, amplitudes) del segmento
        """
        start_time = event_time - window_before
        end_time = event_time + window_after
        
        mask = (time >= start_time) & (time <= end_time)
        
        return time[mask], amplitude[mask]
    
    def get_envelope_fft(
        self,
        envelope: EnvelopeResult,
        max_freq: float = 20.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula FFT de la envolvente para ver periodicidad.
        
        Args:
            envelope: EnvelopeResult
            max_freq: Frecuencia máxima a analizar
            
        Returns:
            Tuple de (frecuencias, espectro)
        """
        from scipy import fft
        
        env = envelope.envelope - np.mean(envelope.envelope)
        n = len(env)
        
        # FFT
        spectrum = fft.fft(env)
        half_n = n // 2
        frequencies = fft.fftfreq(n, 1.0 / self.fs)[:half_n]
        amplitude = np.abs(spectrum[:half_n]) * 2 / n
        
        # Filtrar por frecuencia máxima
        mask = frequencies <= max_freq
        frequencies = frequencies[mask]
        amplitude = amplitude[mask]
        
        return frequencies, amplitude
    
    def detect_sidebands(
        self,
        envelope_fft_result: Tuple[np.ndarray, np.ndarray],
        carrier_freq: float,
        tolerance: float = 1.0,
    ) -> List[dict]:
        """
        Detecta bandas laterales en el espectro de la envolvente.
        
        Indica modulación y posibles mecanismos de vibración.
        
        Args:
            envelope_fft_result: Tuple de (frecuencias, espectro)
            carrier_freq: Frecuencia portadora
            tolerance: Tolerancia para agrupar
            
        Returns:
            Lista de bandas laterales
        """
        frequencies, amplitude = envelope_fft_result
        
        # Encontrar picos
        peaks, _ = signal.find_peaks(amplitude, height=0.1 * np.max(amplitude))
        
        sidebands = []
        for peak_idx in peaks:
            freq = frequencies[peak_idx]
            amp = amplitude[peak_idx]
            
            # Calcular distancia a la portadora
            distance = abs(freq - carrier_freq)
            
            if distance > tolerance:
                sidebands.append({
                    'frequency': float(freq),
                    'amplitude': float(amp),
                    'distance_from_carrier': float(distance),
                    'sideband_type': 'upper' if freq > carrier_freq else 'lower',
                })
        
        return sidebands