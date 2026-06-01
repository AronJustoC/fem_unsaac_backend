"""
frequency_domain.py - Análisis en el Dominio de la Frecuencia
=============================================================
FFT, análisis espectral, lectura de frecuencias naturales.
Basado en VibrationData Toolbox para análisis de puentes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple, List
import numpy as np
from scipy import signal, fft


@dataclass
class FFTResult:
    """
    Resultados del análisis FFT.
    
    Attributes:
        frequencies: Array de frecuencias
        amplitude_spectrum: Espectro de amplitud
        phase_spectrum: Espectro de fase
        window_type: Tipo de ventana utilizada
        peak_frequencies: Frecuencias de los picos dominantes
        peak_amplitudes: Amplitudes de los picos
    """
    frequencies: np.ndarray
    amplitude_spectrum: np.ndarray
    phase_spectrum: np.ndarray = None
    window_type: str = "hanning"
    peak_frequencies: np.ndarray = None
    peak_amplitudes: np.ndarray = None
    peak_indices: np.ndarray = None
    
    def to_dict(self) -> dict:
        """Convierte a diccionario serializable."""
        return {
            'frequencies': self.frequencies.tolist(),
            'amplitude_spectrum': self.amplitude_spectrum.tolist(),
            'phase_spectrum': self.phase_spectrum.tolist() if self.phase_spectrum is not None else None,
            'window_type': self.window_type,
            'peak_frequencies': self.peak_frequencies.tolist() if self.peak_frequencies is not None else [],
            'peak_amplitudes': self.peak_amplitudes.tolist() if self.peak_amplitudes is not None else [],
        }


class FrequencyDomainAnalyzer:
    """
    Analizador en el dominio de la frecuencia.
    
    Proporciona herramientas para análisis FFT de señales de vibración
    en puentes, incluyendo detección de frecuencias naturales, picos,
    y análisis espectral.
    """
    
    def __init__(self, fs: float):
        """
        Inicializa el analizador.
        
        Args:
            fs: Frecuencia de muestreo en Hz
        """
        self.fs = fs
        self.nyquist = fs / 2.0
    
    def compute_fft(
        self,
        amplitude: np.ndarray,
        window_type: Literal['hanning', 'hann', 'hamming', 'blackman', 'flattop', 'rectangular'] = 'hanning',
        detrend: bool = True,
        scale: Literal['amplitude', 'psd', 'density'] = 'amplitude',
        include_phase: bool = False,
    ) -> FFTResult:
        """
        Computa la FFT de la señal.
        
        Args:
            amplitude: Array de amplitudes
            window_type: Tipo de ventana ('hanning', 'hann', 'hamming', 'blackman', 'rectangular')
            detrend: Si aplicar detrend (remover media/tendencia)
            scale: Escala de salida ('amplitude', 'psd', 'density')
            include_phase: Si calcular fase. Desactivado por defecto para evitar trabajo extra
                en endpoints que solo usan amplitud y picos.
            
        Returns:
            FFTResult con el espectro calculado
        """
        n = len(amplitude)
        
        # Remover tendencia/media si se solicita
        if detrend:
            amplitude = amplitude - np.mean(amplitude)
        
        # Aplicar ventana
        if window_type in ['hanning', 'hann']:
            window = np.hanning(n)
        elif window_type == 'hamming':
            window = np.hamming(n)
        elif window_type == 'blackman':
            window = np.blackman(n)
        elif window_type == 'flattop':
            window = signal.windows.flattop(n, sym=True)
        else:
            window = np.ones(n)
        
        windowed = amplitude * window
        
        # rFFT evita calcular la mitad negativa del espectro para señales reales.
        spectrum_half = fft.rfft(windowed)
        frequencies = fft.rfftfreq(n, 1.0 / self.fs)
        
        # Escala según tipo
        if scale == 'amplitude':
            # Escala de amplitud one-sided: no duplicar DC ni Nyquist.
            amplitude_spectrum = np.abs(spectrum_half) / np.sum(window)
            if len(amplitude_spectrum) > 1:
                if n % 2 == 0:
                    amplitude_spectrum[1:-1] *= 2.0
                else:
                    amplitude_spectrum[1:] *= 2.0
        elif scale == 'psd':
            # Densidad espectral de potencia
            psd = (np.abs(spectrum_half) ** 2) / (self.fs * np.sum(window ** 2))
            if len(psd) > 1:
                if n % 2 == 0:
                    psd[1:-1] *= 2.0
                else:
                    psd[1:] *= 2.0
            amplitude_spectrum = psd
        else:
            # Densidad
            amplitude_spectrum = np.abs(spectrum_half) / np.sqrt(self.fs * np.sum(window ** 2))
            if len(amplitude_spectrum) > 1:
                if n % 2 == 0:
                    amplitude_spectrum[1:-1] *= 2.0
                else:
                    amplitude_spectrum[1:] *= 2.0
        
        # Fase
        phase_spectrum = np.angle(spectrum_half) if include_phase else None
        
        return FFTResult(
            frequencies=frequencies,
            amplitude_spectrum=amplitude_spectrum,
            phase_spectrum=phase_spectrum,
            window_type=window_type,
        )
    
    def find_peaks(
        self,
        result: FFTResult,
        min_height: Optional[float] = None,
        min_prominence: Optional[float] = None,
        max_peaks: int = 10,
        freq_range: Optional[Tuple[float, float]] = None,
    ) -> FFTResult:
        """
        Encuentra picos dominantes en el espectro.
        
        Args:
            result: FFTResult previo
            min_height: Altura mínima de los picos (default: 10% del max)
            min_prominence: Prominencia mínima de los picos
            max_peaks: Número máximo de picos a detectar
            freq_range: Rango de frecuencias a considerar (tuple)
            
        Returns:
            FFTResult con picos añadidos
        """
        frequencies = result.frequencies
        amplitude = result.amplitude_spectrum
        
        # Filtrar por rango de frecuencias
        if freq_range is not None:
            mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
            frequencies = frequencies[mask]
            amplitude = amplitude[mask]
        
        if len(amplitude) == 0:
            result.peak_frequencies = np.array([])
            result.peak_amplitudes = np.array([])
            result.peak_indices = np.array([], dtype=int)
            return result
        
        # Definir umbral de altura
        if min_height is None:
            min_height = 0.1 * np.max(amplitude)
        
        # Encontrar picos
        peak_indices, properties = signal.find_peaks(
            amplitude,
            height=min_height,
            prominence=min_prominence,
        )
        
        # Ordenar por amplitud y limitar
        if len(peak_indices) > max_peaks:
            sorted_indices = np.argsort(amplitude[peak_indices])[::-1]
            peak_indices = peak_indices[sorted_indices[:max_peaks]]
        
        # Ordenar por frecuencia
        sorted_order = np.argsort(frequencies[peak_indices])
        peak_indices = peak_indices[sorted_order]
        
        result.peak_frequencies = frequencies[peak_indices]
        result.peak_amplitudes = amplitude[peak_indices]
        result.peak_indices = peak_indices
        
        return result
    
    def compute_fft_segment(
        self,
        time: np.ndarray,
        amplitude: np.ndarray,
        segment_start: float,
        segment_end: float,
        window_type: Literal['hanning', 'hann', 'hamming', 'blackman'] = 'hanning',
    ) -> FFTResult:
        """
        Computa FFT para un segmento específico de la señal.
        
        Útil para analizar eventos individuales como pasos de vehículos.
        
        Args:
            time: Array de tiempo
            amplitude: Array de amplitudes
            segment_start: Tiempo inicial del segmento
            segment_end: Tiempo final del segmento
            window_type: Tipo de ventana
            
        Returns:
            FFTResult del segmento
        """
        mask = (time >= segment_start) & (time <= segment_end)
        segment = amplitude[mask]
        
        return self.compute_fft(segment, window_type=window_type)
    
    def compute_short_time_fft(
        self,
        amplitude: np.ndarray,
        nperseg: int = 1024,
        noverlap: Optional[int] = None,
        window_type: Literal['hanning', 'hann', 'hamming'] = 'hanning',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computa FFT de tiempo corto (STFT) para análisis tiempo-frecuencia.
        
        Args:
            amplitude: Array de amplitudes
            nperseg: Número de muestras por segmento
            noverlap: Número de muestras de solapamiento (default: nperseg/2)
            window_type: Tipo de ventana
            
        Returns:
            Tuple de (frecuencias, tiempos, STFT)
        """
        if noverlap is None:
            noverlap = nperseg // 2
        
        # STFT
        if window_type in ['hanning', 'hann']:
            window = 'hann'
        elif window_type == 'hamming':
            window = 'hamming'
        else:
            window = 'hann'
        
        f, t, stft = signal.stft(
            amplitude,
            fs=self.fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
        )
        
        # Magnitud
        stft_magnitude = np.abs(stft)
        
        return f, t, stft_magnitude
    
    def compute_spectrogram(
        self,
        amplitude: np.ndarray,
        nperseg: int = 1024,
        noverlap: Optional[int] = None,
        freq_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computa el espectrograma de la señal.
        
        Args:
            amplitude: Array de amplitudes
            nperseg: Longitud del segmento
            noverlap: Solapamiento
            freq_range: Rango de frecuencias a mostrar
            
        Returns:
            Tuple de (frecuencias, tiempos, espectrograma)
        """
        if noverlap is None:
            noverlap = nperseg // 2
        
        f, t, sxx = signal.spectrogram(
            amplitude,
            fs=self.fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hann',
        )
        
        # Filtrar por rango si se especifica
        if freq_range is not None:
            mask = (f >= freq_range[0]) & (f <= freq_range[1])
            f = f[mask]
            sxx = sxx[mask, :]
        
        return f, t, sxx
    
    def get_frequencies_in_range(
        self,
        result: FFTResult,
        freq_range: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtiene frecuencias y amplitudes en un rango específico.
        
        Args:
            result: FFTResult
            freq_range: Tuple (f_min, f_max)
            
        Returns:
            Tuple de (frecuencias_filtradas, amplitudes_filtradas)
        """
        mask = (result.frequencies >= freq_range[0]) & (result.frequencies <= freq_range[1])
        return result.frequencies[mask], result.amplitude_spectrum[mask]
    
    def identify_structural_modes(
        self,
        result: FFTResult,
        min_prominence_ratio: float = 0.1,
        mode_type: Literal['vertical', 'lateral', 'longitudinal'] = 'vertical',
    ) -> dict:
        """
        Identifica modos estructurales (frecuencias naturales) del puente.
        
        Args:
            result: FFTResult con picos detectados
            min_prominence_ratio: Ratio mínimo de prominencia respecto al máximo
            mode_type: Tipo de modo ('vertical', 'lateral', 'longitudinal')
            
        Returns:
            Diccionario con modos identificados
        """
        if result.peak_frequencies is None or len(result.peak_frequencies) == 0:
            return {
                'modes': [],
                'mode_type': mode_type,
                'confidence': 'low',
                'message': 'No se detectaron picos claros'
            }
        
        peaks = result.peak_frequencies
        amplitudes = result.peak_amplitudes
        
        # Filtrar por prominencia
        max_amp = np.max(amplitudes)
        min_prom = min_prominence_ratio * max_amp
        
        significant_indices = amplitudes >= min_prom
        significant_freqs = peaks[significant_indices]
        significant_amps = amplitudes[significant_indices]
        
        # Ordenar por amplitud
        order = np.argsort(significant_amps)[::-1]
        significant_freqs = significant_freqs[order]
        significant_amps = significant_amps[order]
        
        modes = []
        for i, (f, a) in enumerate(zip(significant_freqs, significant_amps)):
            period = 1.0 / f if f > 0 else 0
            modes.append({
                'mode_number': i + 1,
                'frequency_hz': float(f),
                'period_s': float(period),
                'amplitude': float(a),
                'classification': self._classify_mode(f, mode_type),
            })
        
        return {
            'modes': modes,
            'mode_type': mode_type,
            'n_modes': len(modes),
            'fundamental_frequency': float(significant_freqs[0]) if len(significant_freqs) > 0 else None,
            'confidence': 'high' if len(modes) >= 3 else 'medium' if len(modes) >= 1 else 'low',
        }
    
    def _classify_mode(
        self,
        frequency: float,
        mode_type: str
    ) -> str:
        """
        Clasifica un modo según su frecuencia.
        
        Args:
            frequency: Frecuencia en Hz
            mode_type: Tipo de análisis
            
        Returns:
            Clasificación del modo
        """
        if frequency < 1.0:
            classification = "Muy baja frecuencia"
        elif frequency < 3.0:
            classification = "Baja frecuencia (primer modo)"
        elif frequency < 8.0:
            classification = "Frecuencia media (segundo/tercer modo)"
        elif frequency < 15.0:
            classification = "Alta frecuencia (modos superiores)"
        else:
            classification = "Muy alta frecuencia (ruido/artefactos)"
        
        return classification
    
    def detect_noise_frequencies(
        self,
        result: FFTResult,
        power_line_freq: float = 50.0,
        harmonics: int = 3,
    ) -> dict:
        """
        Detecta frecuencias de ruido eléctrico (50/60 Hz y armónicos).
        
        Args:
            result: FFTResult
            power_line_freq: Frecuencia de línea (50 o 60 Hz)
            harmonics: Número de armónicos a buscar
            
        Returns:
            Diccionario con frecuencias de ruido detectadas
        """
        noise_freqs = []
        threshold = 5.0  # Hz de tolerancia
        
        for h in range(1, harmonics + 1):
            noise_freq = power_line_freq * h
            noise_freqs.append(noise_freq)
        
        # Buscar picos cerca de estas frecuencias
        detected_noise = []
        for noise_freq in noise_freqs:
            if noise_freq > self.nyquist:
                continue
            
            mask = np.abs(result.frequencies - noise_freq) < threshold
            if np.any(mask):
                peak_amp = np.max(result.amplitude_spectrum[mask])
                peak_freq = result.frequencies[mask][np.argmax(result.amplitude_spectrum[mask])]
                detected_noise.append({
                    'harmonic': int(round(noise_freq / power_line_freq)),
                    'expected_freq': noise_freq,
                    'detected_freq': float(peak_freq),
                    'amplitude': float(peak_amp),
                    'deviation_hz': float(peak_freq - noise_freq),
                })
        
        return {
            'noise_type': 'power_line',
            'power_line_freq': power_line_freq,
            'detected_noise': detected_noise,
            'is_noisy': len(detected_noise) > 0,
        }
    
    def compute_coherence(
        self,
        amplitude1: np.ndarray,
        amplitude2: np.ndarray,
        nperseg: int = 1024,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computa la función de coherencia entre dos señales.
        
        Útil para identificar correlaciones entre canales X, Y, Z.
        
        Args:
            amplitude1: Primera señal
            amplitude2: Segunda señal
            nperseg: Longitud del segmento
            
        Returns:
            Tuple de (frecuencias, coherencia)
        """
        f, _, cxy = signal.coherence(
            amplitude1,
            amplitude2,
            fs=self.fs,
            nperseg=nperseg,
        )
        
        return f, cxy
    
    def compute_phase_difference(
        self,
        amplitude1: np.ndarray,
        amplitude2: np.ndarray,
        freq: float,
        bandwidth: float = 0.5,
    ) -> float:
        """
        Computa la diferencia de fase entre dos señales en una frecuencia específica.
        
        Args:
            amplitude1: Primera señal
            amplitude2: Segunda señal
            freq: Frecuencia de interés
            bandwidth: Ancho de banda para el filtro
            
        Returns:
            Diferencia de fase en radianes
        """
        # Filtrar ambas señales alrededor de la frecuencia
        nyquist = self.fs / 2
        low = max(0.01, (freq - bandwidth) / nyquist)
        high = min(1.0, (freq + bandwidth) / nyquist)
        
        b, a = signal.butter(4, [low, high], btype='band')
        
        filtered1 = signal.filtfilt(b, a, amplitude1)
        filtered2 = signal.filtfilt(b, a, amplitude2)
        
        # Calcular fase usando Hilbert
        analytic1 = signal.hilbert(filtered1)
        analytic2 = signal.hilbert(filtered2)
        
        phase1 = np.angle(analytic1)
        phase2 = np.angle(analytic2)
        
        # Diferencia de fase media
        phase_diff = np.mean(np.unwrap(phase1 - phase2))
        
        return float(phase_diff)
