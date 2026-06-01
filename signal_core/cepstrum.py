"""
cepstrum.py - Análisis de Cepstrum
==================================
Detección de periodicidad en el espectro, análisis de quefrency.
Especialmente útil para identificar repetición de cargas y sidebands.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from scipy import fft, signal


@dataclass
class CepstrumResult:
    """
    Resultado del análisis de Cepstrum.
    
    Attributes:
        quefrency: Array de quefrency (similar al tiempo, pero en el dominio del espectro)
        cepstrum: Amplitud del cepstrum
        dominant_quefrencies: Quefrencys dominantes (separaciones espectrales)
        echo_interpretation: Interpretación de ecos/repeticiones
    """
    quefrency: np.ndarray
    cepstrum: np.ndarray
    dominant_quefrencies: List[float]
    dominant_amplitudes: List[float]
    echo_interpretation: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convierte a diccionario serializable."""
        return {
            'quefrency': self.quefrency.tolist(),
            'cepstrum': self.cepstrum.tolist(),
            'dominant_quefrencies': self.dominant_quefrencies,
            'dominant_amplitudes': self.dominant_amplitudes,
            'echo_interpretation': self.echo_interpretation,
        }


@dataclass
class QuefrencyPeak:
    """
    Pico en el dominio de quefrency.
    
    Attributes:
        quefrency_s: Posición del pico en segundos (o milisegundos)
        amplitude: Amplitud del pico
        period_hz: Periodo equivalente en Hz (1/quefrency)
        classification: Clasificación del pico
    """
    quefrency_s: float
    amplitude: float
    period_hz: float
    classification: str
    
    def __repr__(self):
        return f"QuefrencyPeak(q={self.quefrency_s:.4f}s, f={self.period_hz:.2f}Hz, class={self.classification})"


class CepstrumAnalyzer:
    """
    Analizador de Cepstrum para detección de periodicidad.
    
    El cepstrum es la FFT de la FFT (o equivalentemente,
    la transformada inversa del logaritmo del espectro).
    Es útil para detectar repeticiones regulares en el espectro
    que no son visibles directamente.
    """
    
    def __init__(self, fs: float):
        """
        Inicializa el analizador.
        
        Args:
            fs: Frecuencia de muestreo en Hz
        """
        self.fs = fs
    
    def compute_power_cepstrum(
        self,
        amplitude: np.ndarray,
        window_type: str = 'hann',
        scale: str = 'log',
    ) -> CepstrumResult:
        """
        Computa el power cepstrum.
        
        Pasos:
        1. FFT de la señal
        2. Logaritmo del cuadrado de la magnitud
        3. FFT inversa
        
        Args:
            amplitude: Array de amplitudes
            window_type: Tipo de ventana
            scale: Escala de salida ('log' o 'linear')
            
        Returns:
            CepstrumResult
        """
        n = len(amplitude)
        
        # Preparar ventana
        if window_type == 'hann':
            window = np.hanning(n)
        elif window_type == 'hamming':
            window = np.hamming(n)
        else:
            window = np.ones(n)
        
        windowed = amplitude * window
        
        # FFT
        spectrum = fft.fft(windowed)
        
        # Solo mitad positiva
        half_n = n // 2
        magnitude_squared = np.abs(spectrum[:half_n]) ** 2
        
        # Logaritmo
        log_magnitude = np.log(magnitude_squared + 1e-10)
        
        # FFT inversa
        log_spectrum = np.zeros(n, dtype=complex)
        log_spectrum[:half_n] = log_magnitude
        log_spectrum[-half_n + 1:] = log_magnitude[1:half_n][::-1].conj()
        
        cepstrum = np.real(fft.ifft(log_spectrum))
        
        # Escala
        if scale == 'log':
            cepstrum = 10 * np.log10(np.abs(cepstrum) + 1e-10)
        
        # Calcular quefrency
        dt = 1.0 / self.fs
        quefrency = np.arange(n) * dt
        
        return CepstrumResult(
            quefrency=quefrency,
            cepstrum=cepstrum,
            dominant_quefrencies=[],
            dominant_amplitudes=[],
        )
    
    def compute_real_cepstrum(
        self,
        amplitude: np.ndarray,
        window_type: str = 'hann',
    ) -> CepstrumResult:
        """
        Computa el real cepstrum.
        
        Pasos:
        1. FFT de la señal
        2. Logaritmo de la magnitud (sin平方)
        3. FFT inversa
        4. Parte real
        
        Args:
            amplitude: Array de amplitudes
            window_type: Tipo de ventana
            
        Returns:
            CepstrumResult
        """
        n = len(amplitude)
        
        # Preparar ventana
        if window_type == 'hann':
            window = np.hanning(n)
        elif window_type == 'hamming':
            window = np.hamming(n)
        else:
            window = np.ones(n)
        
        windowed = amplitude * window
        
        # FFT
        spectrum = fft.fft(windowed)
        
        # Solo mitad positiva
        half_n = n // 2
        magnitude = np.abs(spectrum[:half_n])
        
        # Logaritmo de la magnitud
        log_magnitude = np.log(magnitude + 1e-10)
        
        # FFT inversa
        log_spectrum = np.zeros(n, dtype=complex)
        log_spectrum[:half_n] = log_magnitude
        log_spectrum[-half_n + 1:] = log_magnitude[1:half_n][::-1].conj()
        
        cepstrum = np.real(fft.ifft(log_spectrum))
        
        # Calcular quefrency
        dt = 1.0 / self.fs
        quefrency = np.arange(n) * dt
        
        return CepstrumResult(
            quefrency=quefrency,
            cepstrum=cepstrum,
            dominant_quefrencies=[],
            dominant_amplitudes=[],
        )
    
    def compute_cepstrum_from_fft(
        self,
        fft_frequencies: np.ndarray,
        fft_amplitude: np.ndarray,
    ) -> CepstrumResult:
        """
        Computa el cepstrum directamente desde el espectro FFT.
        
        Útil cuando ya se tiene el resultado de FFT.
        
        Args:
            fft_frequencies: Array de frecuencias del FFT
            fft_amplitude: Array de amplitudes del FFT
            
        Returns:
            CepstrumResult
        """
        n = len(fft_amplitude)
        
        # Logaritmo de la magnitud
        log_amplitude = np.log(fft_amplitude + 1e-10)
        
        # FFT inversa
        cepstrum = np.fft.ifft(log_amplitude).real
        
        # Calcular quefrency (inverso de la resolución en frecuencia)
        df = fft_frequencies[1] - fft_frequencies[0] if len(fft_frequencies) > 1 else 1.0
        quefrency = np.arange(n) * (1.0 / (n * df))
        
        return CepstrumResult(
            quefrency=quefrency,
            cepstrum=cepstrum,
            dominant_quefrencies=[],
            dominant_amplitudes=[],
        )
    
    def find_dominant_quefrencies(
        self,
        cepstrum: CepstrumResult,
        min_quefrency: float = 0.01,
        max_quefrency: float = 10.0,
        n_peaks: int = 5,
        prominence_ratio: float = 0.1,
    ) -> CepstrumResult:
        """
        Encuentra los quefrencys dominantes.
        
        Args:
            cepstrum: CepstrumResult base
            min_quefrency: Quefrency mínimo a considerar
            max_quefrency: Quefrency máximo a considerar
            n_peaks: Número de picos a encontrar
            prominence_ratio: Prominencia mínima relativa
            
        Returns:
            CepstrumResult con picos añadidos
        """
        # Filtrar por rango de quefrency
        mask = (cepstrum.quefrency >= min_quefrency) & (cepstrum.quefrency <= max_quefrency)
        
        if not np.any(mask):
            return cepstrum
        
        quefrency_filtered = cepstrum.quefrency[mask]
        cepstrum_filtered = cepstrum.cepstrum[mask]
        
        # Encontrar picos
        threshold = prominence_ratio * np.max(cepstrum_filtered)
        peaks, _ = signal.find_peaks(cepstrum_filtered, prominence=threshold)
        
        if len(peaks) == 0:
            return cepstrum
        
        # Ordenar por amplitud
        amplitudes = cepstrum_filtered[peaks]
        order = np.argsort(amplitudes)[::-1][:n_peaks]
        peaks = peaks[order]
        
        dominant_q = quefrency_filtered[peaks]
        dominant_amps = cepstrum_filtered[peaks]
        
        cepstrum.dominant_quefrencies = dominant_q.tolist()
        cepstrum.dominant_amplitudes = dominant_amps.tolist()
        
        return cepstrum
    
    def interpret_peaks(
        self,
        cepstrum: CepstrumResult,
    ) -> List[QuefrencyPeak]:
        """
        Interpreta los picos del cepstrum como periodicidad en el espectro.
        
        Un pico en quefrency Q corresponde aproximadamente a una
        separación de 1/Q Hz en el espectro.
        
        Args:
            cepstrum: CepstrumResult con picos detectados
            
        Returns:
            Lista de QuefrencyPeak interpretados
        """
        results = []
        
        for q, amp in zip(cepstrum.dominant_quefrencies, cepstrum.dominant_amplitudes):
            if q > 0:
                period_hz = 1.0 / q
            else:
                period_hz = 0
            
            # Clasificar el pico
            classification = self._classify_quefrency(q)
            
            results.append(QuefrencyPeak(
                quefrency_s=q,
                amplitude=amp,
                period_hz=period_hz,
                classification=classification,
            ))
        
        return results
    
    def _classify_quefrency(self, quefrency: float) -> str:
        """
        Clasifica un quefrency según su valor.
        
        Args:
            quefrency: Valor en segundos
            
        Returns:
            Clasificación
        """
        if quefrency < 0.001:
            return "Muy corta - posiblemente ruido"
        elif quefrency < 0.01:
            return "Corta - alta frecuencia"
        elif quefrency < 0.05:
            return "Media - frecuencia de paso de ejes pequeños"
        elif quefrency < 0.1:
            return "Moderada - frecuencia de paso vehicular"
        elif quefrency < 0.5:
            return "Larga -周期的 loading o reflexiones"
        elif quefrency < 1.0:
            return "Muy larga - efectos de modo estructural"
        else:
            return "Extra larga - posiblemente artefacto"
    
    def detect_spectral_repetition(
        self,
        fft_result: Tuple[np.ndarray, np.ndarray],
        expected_spacing: Optional[float] = None,
        tolerance: float = 0.5,
    ) -> dict:
        """
        Detecta repeticiones regulares en el espectro FFT.
        
        Args:
            fft_result: Tuple de (frecuencias, amplitudes)
            expected_spacing: Espaciado esperado entre picos (Hz)
            tolerance: Tolerancia para detectar espaciado
            
        Returns:
            Diccionario con análisis
        """
        frequencies, amplitude = fft_result
        
        # Encontrar picos en el espectro
        peaks, _ = signal.find_peaks(amplitude, height=0.1 * np.max(amplitude))
        
        if len(peaks) < 2:
            return {
                'has_repetition': False,
                'reason': 'Pocos picos detectados',
            }
        
        peak_freqs = frequencies[peaks]
        
        # Calcular diferencias entre picos consecutivos
        diffs = np.diff(peak_freqs)
        
        # Agrupar diferencias similares
        mean_spacing = np.mean(diffs)
        std_spacing = np.std(diffs)
        
        cv = std_spacing / (mean_spacing + 1e-10)
        has_repetition = cv < 0.2  # Coeficiente de variación bajo
        
        return {
            'has_repetition': has_repetition,
            'mean_spacing_hz': float(mean_spacing),
            'std_spacing_hz': float(std_spacing),
            'coefficient_of_variation': float(cv),
            'n_peaks_analyzed': len(peaks),
            'expected_spacing_hz': expected_spacing,
        }
    
    def compute_minimum_phase_cepstrum(
        self,
        amplitude: np.ndarray,
    ) -> CepstrumResult:
        """
        Computa el cepstrum de fase mínima.
        
        Útil para análisis de sistemas de fase mínima.
        
        Args:
            amplitude: Array de amplitudes
            
        Returns:
            CepstrumResult de fase mínima
        """
        n = len(amplitude)
        
        # FFT
        spectrum = fft.fft(amplitude)
        
        # Calcular fase mínima
        # log(A * exp(jφ)) = log(A) + jφ
        log_spectrum = np.log(np.abs(spectrum) + 1e-10) + 1j * np.angle(spectrum)
        
        #傅立叶逆变换
        cepstrum = np.fft.ifft(log_spectrum).real
        
        # Quefrency
        dt = 1.0 / self.fs
        quefrency = np.arange(n) * dt
        
        return CepstrumResult(
            quefrency=quefrency,
            cepstrum=cepstrum,
            dominant_quefrencies=[],
            dominant_amplitudes=[],
        )
    
    def apply_lifter(
        self,
        cepstrum: CepstrumResult,
        lifter_type: str = 'low quefrency',
        cutoff_quefrency: float = 0.01,
    ) -> np.ndarray:
        """
        Aplica lifters (filtros en quefrency) para limpiar el cepstrum.
        
        Args:
            cepstrum: CepstrumResult
            lifter_type: 'low quefrency' (suavizar) o 'high quefrency' (detallar)
            cutoff_quefrency: Cutoff del lifter
            
        Returns:
            Cepstrum modificado
        """
        q = cepstrum.quefrency
        c = cepstrum.cepstrum
        
        # Encontrar índice del cutoff
        idx = np.argmin(np.abs(q - cutoff_quefrency))
        
        if lifter_type == 'low quefrency':
            # Mantener solo bajas quefrencys (cola suave)
            result = np.zeros_like(c)
            result[:idx] = c[:idx]
        else:
            # Mantener solo altas quefrencys (picos)
            result = np.zeros_like(c)
            result[idx:] = c[idx:]
        
        return result
    
    def detect_echo_effect(
        self,
        amplitude: np.ndarray,
        delay_range: Tuple[float, float] = (0.01, 1.0),
        min_correlation: float = 0.5,
    ) -> dict:
        """
        Detecta efecto de eco o reflexión.
        
        Un eco aparece como un pico en el cepstrum.
        
        Args:
            amplitude: Array de amplitudes
            delay_range: Rango de delays a considerar (s)
            min_correlation: Correlación mínima para detectar eco
            
        Returns:
            Diccionario con análisis de eco
        """
        n = len(amplitude)
        dt = 1.0 / self.fs
        max_delay_samples = int(delay_range[1] / dt)
        
        # Autocorrelation
        autocorr = np.correlate(amplitude, amplitude, mode='full')
        autocorr = autocorr[n - 1:]  # Solo la mitad positiva
        autocorr = autocorr / (autocorr[0] + 1e-10)
        
        # Buscar picos en la autocorrelation fuera de 0
        if max_delay_samples < len(autocorr):
            search_region = autocorr[1:max_delay_samples + 1]
        else:
            search_region = autocorr[1:]
        
        if len(search_region) > 2:
            peaks, _ = signal.find_peaks(search_region, height=min_correlation)
            
            if len(peaks) > 0:
                delays = (peaks + 1) * dt
                correlations = search_region[peaks]
                
                return {
                    'has_echo': True,
                    'echo_delays_s': delays.tolist(),
                    'echo_correlations': correlations.tolist(),
                    'n_echoes': len(peaks),
                }
        
        return {
            'has_echo': False,
            'n_echoes': 0,
        }
    
    def analyze_sidebands(
        self,
        fft_frequencies: np.ndarray,
        fft_amplitude: np.ndarray,
        carrier_freq: float,
        sideband_spacing: Optional[float] = None,
        n_sidebands: int = 5,
    ) -> dict:
        """
        Analiza sidebands (bandas laterales) alrededor de una frecuencia portadora.
        
        Args:
            fft_frequencies: Array de frecuencias
            fft_amplitude: Array de amplitudes
            carrier_freq: Frecuencia portadora
            sideband_spacing: Espaciado esperado de sidebands
            n_sidebands: Número de sidebands a buscar
            
        Returns:
            Diccionario con análisis
        """
        mask = (fft_frequencies >= carrier_freq - 20) & (fft_frequencies <= carrier_freq + 20)
        freq_region = fft_frequencies[mask]
        amp_region = fft_amplitude[mask]
        
        if len(freq_region) == 0:
            return {'has_sidebands': False, 'reason': 'Frecuencia portadora fuera del rango'}
        
        # Encontrar la portadora
        carrier_idx = np.argmin(np.abs(fft_frequencies - carrier_freq))
        carrier_amp = fft_amplitude[carrier_idx]
        
        # Buscar picos cerca de la portadora
        peaks, _ = signal.find_peaks(amp_region, height=0.1 * carrier_amp)
        
        if len(peaks) == 0:
            return {'has_sidebands': False, 'reason': 'No se detectaron picos'}
        
        sidebands = []
        for peak_idx in peaks:
            freq = freq_region[peak_idx]
            amp = amp_region[peak_idx]
            spacing = freq - carrier_freq
            
            sidebands.append({
                'frequency': float(freq),
                'spacing_from_carrier': float(spacing),
                'amplitude_ratio': float(amp / carrier_amp) if carrier_amp > 0 else 0,
            })
        
        # Verificar si las separaciones son regulares
        if len(sidebands) >= 2:
            spacings = [s['spacing_from_carrier'] for s in sidebands]
            std_spacing = np.std(spacings)
            mean_spacing = np.abs(np.mean(spacings))
            
            cv = std_spacing / (mean_spacing + 1e-10)
            has_regular_spacing = cv < 0.3
        else:
            has_regular_spacing = False
        
        return {
            'has_sidebands': len(sidebands) > 0,
            'carrier_frequency': float(carrier_freq),
            'sidebands': sidebands,
            'has_regular_spacing': has_regular_spacing,
        }