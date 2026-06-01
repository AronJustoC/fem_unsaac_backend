"""
spectral_analysis.py - Análisis Espectral Avanzado
==================================================
PSD (Power Spectral Density), Welch, Waterfall, Spectrogram.
Basado en VibrationData Toolbox para análisis de puentes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple, List
import numpy as np
from scipy import signal


@dataclass
class PSDResult:
    """
    Resultados del análisis de Densidad Espectral de Potencia.
    
    Attributes:
        frequencies: Array de frecuencias
        psd: Densidad espectral de potencia
        rms_bands: RMS calculado por bandas
        band_edges: Bordes de las bandas usadas
        method: Método usado ('welch', 'periodogram', etc.)
    """
    frequencies: np.ndarray
    psd: np.ndarray
    rms_bands: Optional[List[float]] = None
    band_edges: Optional[List[Tuple[float, float]]] = None
    method: str = "welch"
    
    def to_dict(self) -> dict:
        """Convierte a diccionario serializable."""
        result = {
            'frequencies': self.frequencies.tolist(),
            'psd': self.psd.tolist(),
            'method': self.method,
        }
        if self.rms_bands is not None:
            result['rms_bands'] = self.rms_bands
        if self.band_edges is not None:
            result['band_edges'] = [(e[0], e[1]) for e in self.band_edges]
        return result
    
    def get_band_rms(self, f_low: float, f_high: float) -> float:
        """
        Calcula el RMS en una banda de frecuencias específica.
        
        Args:
            f_low: Frecuencia baja
            f_high: Frecuencia alta
            
        Returns:
            RMS en la banda
        """
        mask = (self.frequencies >= f_low) & (self.frequencies <= f_high)
        if not np.any(mask):
            return 0.0
        
        psd_band = self.psd[mask]
        freqs_band = self.frequencies[mask]
        
        if len(freqs_band) < 2:
            return 0.0
        
        # Integral numérica usando trapezoid
        df = np.diff(freqs_band)
        rms_squared = np.sum(0.5 * (psd_band[:-1] + psd_band[1:]) * df)
        
        return float(np.sqrt(rms_squared))


@dataclass
class WaterfallResult:
    """
    Resultados del análisis Waterfall (FFT 3D).
    
    Attributes:
        frequencies: Array de frecuencias
        times: Array de tiempos
        amplitude_matrix: Matriz 3D de amplitudes
        peak_history: Historia de picos en el tiempo
    """
    frequencies: np.ndarray
    times: np.ndarray
    amplitude_matrix: np.ndarray
    peak_history: Optional[List[dict]] = None
    
    def to_dict(self) -> dict:
        """Convierte a diccionario serializable."""
        return {
            'frequencies': self.frequencies.tolist(),
            'times': self.times.tolist(),
            'amplitude_matrix': self.amplitude_matrix.tolist(),
            'n_slices': len(self.times),
        }


class SpectralAnalyzer:
    """
    Analizador espectral para señales de vibración.
    
    Proporciona PSD, Welch, y análisis de bandas para puentes.
    """
    
    def __init__(self, fs: float):
        """
        Inicializa el analizador.
        
        Args:
            fs: Frecuencia de muestreo en Hz
        """
        self.fs = fs
        self.nyquist = fs / 2.0
    
    def compute_welch_psd(
        self,
        amplitude: np.ndarray,
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
        window: str = 'hann',
        scaling: Literal['density', 'spectrum'] = 'density',
    ) -> PSDResult:
        """
        Computa la PSD usando el método de Welch.
        
        Método robusto y estable para señales aleatorias o largas.
        
        Args:
            amplitude: Array de amplitudes
            nperseg: Longitud de cada segmento (default: 1024 o menor si señal muy corta)
            noverlap: Solapamiento entre segmentos (default: nperseg/2)
            window: Tipo de ventana ('hann', 'hamming', 'blackman', etc.)
            scaling: Tipo de escala ('density' o 'spectrum')
            
        Returns:
            PSDResult con la PSD calculada
        """
        if nperseg is None:
            nperseg = min(1024, len(amplitude) // 4)
        
        if noverlap is None:
            noverlap = nperseg // 2
        
        # Asegurar que nperseg no sea mayor que la señal
        if nperseg > len(amplitude):
            nperseg = len(amplitude) // 2
            noverlap = nperseg // 4
        
        f, psd = signal.welch(
            amplitude,
            fs=self.fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
            scaling=scaling,
        )
        
        return PSDResult(
            frequencies=f,
            psd=psd,
            method='welch',
        )
    
    def compute_periodogram_psd(
        self,
        amplitude: np.ndarray,
        window_type: str = 'hann',
        scaling: Literal['density', 'spectrum'] = 'density',
    ) -> PSDResult:
        """
        Computa la PSD usando periodograma (FFT directa).
        
        Útil para señales transitorias o cortas.
        
        Args:
            amplitude: Array de amplitudes
            window_type: Tipo de ventana
            scaling: Tipo de escala
            
        Returns:
            PSDResult con la PSD calculada
        """
        n = len(amplitude)
        
        # Aplicar ventana
        if window_type == 'hann':
            window = np.hanning(n)
        elif window_type == 'hamming':
            window = np.hamming(n)
        elif window_type == 'blackman':
            window = np.blackman(n)
        else:
            window = np.ones(n)
        
        # FFT
        from scipy import fft
        spectrum = fft.fft(amplitude * window)
        
        # Solo mitad positiva
        n_half = n // 2
        f = fft.fftfreq(n, 1.0 / self.fs)[:n_half]
        psd_full = 2.0 * np.abs(spectrum[:n_half]) ** 2 / (self.fs * np.sum(window ** 2))
        
        return PSDResult(
            frequencies=f,
            psd=psd_full,
            method='periodogram',
        )
    
    def compute_band_rms(
        self,
        psd_result: PSDResult,
        band_edges: List[Tuple[float, float]],
    ) -> List[float]:
        """
        Calcula el RMS en múltiples bandas de frecuencia.
        
        Args:
            psd_result: PSDResult previo
            band_edges: Lista de tuples (f_low, f_high)
            
        Returns:
            Lista de RMS por banda
        """
        rms_values = []
        for f_low, f_high in band_edges:
            rms = psd_result.get_band_rms(f_low, f_high)
            rms_values.append(rms)
        
        return rms_values
    
    def get_cumulative_rms(
        self,
        psd_result: PSDResult,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula el RMS acumulativo vs frecuencia.
        
        Args:
            psd_result: PSDResult
            
        Returns:
            Tuple de (frecuencias, rms_acumulativo)
        """
        f = psd_result.frequencies
        psd = psd_result.psd
        
        # Integral acumulativa
        cumulative = np.zeros_like(psd)
        for i in range(len(psd)):
            if i == 0:
                cumulative[i] = psd[i]
            else:
                df = f[i] - f[i-1]
                cumulative[i] = cumulative[i-1] + 0.5 * (psd[i] + psd[i-1]) * df
        
        cumulative_rms = np.sqrt(cumulative)
        
        return f, cumulative_rms
    
    def get_spectral_peaks(
        self,
        psd_result: PSDResult,
        n_peaks: int = 5,
        min_prominence_ratio: float = 0.1,
    ) -> List[dict]:
        """
        Encuentra los picos más significativos en la PSD.
        
        Args:
            psd_result: PSDResult
            n_peaks: Número de picos a encontrar
            min_prominence_ratio: Ratio mínimo de prominencia
            
        Returns:
            Lista de diccionarios con información de cada pico
        """
        peaks, properties = signal.find_peaks(
            psd_result.psd,
            prominence=min_prominence_ratio * np.max(psd_result.psd),
        )
        
        if len(peaks) == 0:
            return []
        
        # Ordenar por amplitud
        amplitudes = psd_result.psd[peaks]
        order = np.argsort(amplitudes)[::-1][:n_peaks]
        peaks = peaks[order]
        amplitudes = amplitudes[order]
        
        results = []
        for i, (peak_idx, amp) in enumerate(zip(peaks, amplitudes)):
            freq = psd_result.frequencies[peak_idx]
            
            # Calcular ancho de banda (bandwidth)
            half_max = amp / 2
            below_half = psd_result.psd < half_max
            if np.any(below_half[:peak_idx]) and np.any(below_half[peak_idx:]):
                low_idx = np.where(below_half[:peak_idx])[0]
                high_idx = np.where(below_half[peak_idx:])[0] + peak_idx
                if len(low_idx) > 0 and len(high_idx) > 0:
                    bw = psd_result.frequencies[high_idx[-1]] - psd_result.frequencies[low_idx[0]]
                else:
                    bw = 0
            else:
                bw = 0
            
            results.append({
                'rank': i + 1,
                'frequency_hz': float(freq),
                'period_s': float(1.0 / freq) if freq > 0 else 0,
                'psd_value': float(amp),
                'bandwidth_hz': float(bw),
                'quality_factor': float(freq / bw) if bw > 0 else 0,
            })
        
        return results
    
    def compute_spectrogram_matrix(
        self,
        amplitude: np.ndarray,
        nperseg: int = 1024,
        noverlap: Optional[int] = None,
        window: str = 'hann',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computa la matriz de espectrograma (STFT).
        
        Args:
            amplitude: Array de amplitudes
            nperseg: Muestras por segmento
            noverlap: Solapamiento
            window: Tipo de ventana
            
        Returns:
            Tuple de (frecuencias, tiempos, matriz_espectrograma)
        """
        if noverlap is None:
            noverlap = nperseg // 2
        
        f, t, sxx = signal.spectrogram(
            amplitude,
            fs=self.fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
            mode='magnitude',
        )
        
        return f, t, sxx
    
    def compute_spectrogram_db(
        self,
        amplitude: np.ndarray,
        nperseg: int = 1024,
        noverlap: Optional[int] = None,
        ref: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computa el espectrograma en decibelios.
        
        Args:
            amplitude: Array de amplitudes
            nperseg: Muestras por segmento
            noverlap: Solapamiento
            ref: Referencia para dB
            
        Returns:
            Tuple de (frecuencias, tiempos, espectrograma_dB)
        """
        f, t, sxx = self.compute_spectrogram_matrix(amplitude, nperseg, noverlap)
        
        # Convertir a dB
        sxx_db = 10 * np.log10(sxx / ref + 1e-10)
        
        return f, t, sxx_db
    
    def analyze_vehicle_pass(
        self,
        amplitude: np.ndarray,
        time: np.ndarray,
        trigger_threshold: float,
        segment_duration: float = 10.0,
    ) -> dict:
        """
        Analiza específicamente el paso de un vehículo.
        
        Identifica: excitación transitoria, vibración libre, frecuencias naturales.
        
        Args:
            amplitude: Array de amplitudes
            time: Array de tiempo
            trigger_threshold: Umbral para detectar el paso
            segment_duration: Duración del análisis después del trigger
            
        Returns:
            Diccionario con análisis del paso vehicular
        """
        # Encontrar inicio del evento
        abs_amp = np.abs(amplitude)
        trigger_idx = np.argmax(abs_amp > trigger_threshold)
        
        # Segmentar desde trigger
        dt = time[1] - time[0] if len(time) > 1 else 0
        segment_samples = int(segment_duration / dt)
        
        end_idx = min(trigger_idx + segment_samples, len(amplitude))
        segment = amplitude[trigger_idx:end_idx]
        segment_time = time[trigger_idx:end_idx] - time[trigger_idx]
        
        # Análisis de la respuesta libre (después del pico)
        # Encontrar el pico máximo
        max_idx = np.argmax(np.abs(segment))
        free_start = max_idx + 1
        
        if free_start < len(segment):
            free_segment = segment[free_start:]
            free_time = segment_time[free_start:]
            
            # FFT de la respuesta libre
            from .frequency_domain import FrequencyDomainAnalyzer
            freq_analyzer = FrequencyDomainAnalyzer(self.fs)
            fft_result = freq_analyzer.compute_fft(free_segment, window_type='hann')
            fft_result = freq_analyzer.find_peaks(fft_result, min_height=0.1 * np.max(fft_result.amplitude_spectrum))
            
            # Estimar amortiguamiento
            from .time_domain import TimeDomainAnalyzer
            time_analyzer = TimeDomainAnalyzer(self.fs)
            
            # Detectar picos para decremento logarítmico
            peaks, _ = time_analyzer.detect_peaks(free_segment, prominence=0.1 * np.max(free_segment))
            
            # Calcular decremento logarítmico
            if len(peaks) >= 2:
                deltas = []
                for i in range(min(len(peaks) - 1, 5)):
                    if peaks[i] > 0 and peaks[i + 1] > 0:
                        delta = np.log(abs(peaks[i]) / abs(peaks[i + 1]))
                        if delta > 0:
                            deltas.append(delta)
                
                avg_delta = np.mean(deltas) if deltas else 0
                if avg_delta > 0:
                    zeta = avg_delta / np.sqrt(4 * np.pi**2 + avg_delta**2)
                else:
                    zeta = None
            else:
                zeta = None
            
            natural_modes = []
            if fft_result.peak_frequencies is not None:
                for f, amp in zip(fft_result.peak_frequencies, fft_result.peak_amplitudes):
                    natural_modes.append({
                        'frequency_hz': float(f),
                        'amplitude': float(amp),
                        'damping_ratio': float(zeta) if zeta is not None else None,
                    })
            
            return {
                'trigger_time_s': float(time[trigger_idx]),
                'peak_time_s': float(segment_time[max_idx]),
                'peak_amplitude': float(segment[max_idx]),
                'free_vibration_duration_s': float(free_time[-1]) if len(free_time) > 0 else 0,
                'natural_modes': natural_modes,
                'estimated_damping_ratio': float(zeta) if zeta is not None else None,
            }
        else:
            return {
                'trigger_time_s': float(time[trigger_idx]),
                'message': 'No se detectó suficiente vibración libre',
            }


class WaterfallAnalyzer:
    """
    Analizador Waterfall para visualización 3D tiempo-frecuencia.
    
    Genera stacked FFTs para ver la evolución de frecuencias durante
    el paso de vehículos o eventos sísmicos.
    """
    
    def __init__(self, fs: float):
        """
        Inicializa el analizador.
        
        Args:
            fs: Frecuencia de muestreo en Hz
        """
        self.fs = fs
    
    def compute_waterfall(
        self,
        amplitude: np.ndarray,
        segment_length: int = 1024,
        overlap_ratio: float = 0.75,
        window_type: str = 'hann',
        max_freq: Optional[float] = None,
    ) -> WaterfallResult:
        """
        Computa el waterfall (stack de FFTs).
        
        Args:
            amplitude: Array de amplitudes
            segment_length: Longitud de cada segmento
            overlap_ratio: Ratio de solapamiento (0.0 a 0.99)
            window_type: Tipo de ventana
            max_freq: Frecuencia máxima a incluir
            
        Returns:
            WaterfallResult con la matriz 3D
        """
        # Calcular pasos
        step = int(segment_length * (1 - overlap_ratio))
        if step == 0:
            step = 1
        
        n_segments = (len(amplitude) - segment_length) // step + 1
        
        if n_segments <= 0:
            # Señal muy corta
            from .frequency_domain import FrequencyDomainAnalyzer
            freq_analyzer = FrequencyDomainAnalyzer(self.fs)
            fft_result = freq_analyzer.compute_fft(amplitude, window_type=window_type)
            
            return WaterfallResult(
                frequencies=fft_result.frequencies,
                times=np.array([0.0]),
                amplitude_matrix=fft_result.amplitude_spectrum.reshape(1, -1),
            )
        
        # Preparar matriz
        from scipy import fft
        n_freqs = segment_length // 2
        
        if window_type == 'hann':
            window = np.hanning(segment_length)
        elif window_type == 'hamming':
            window = np.hamming(segment_length)
        elif window_type == 'blackman':
            window = np.blackman(segment_length)
        else:
            window = np.ones(segment_length)
        
        amplitude_matrix = np.zeros((n_segments, n_freqs))
        times = np.zeros(n_segments)
        
        for i in range(n_segments):
            start = i * step
            end = start + segment_length
            
            if end > len(amplitude):
                break
            
            segment = amplitude[start:end] - np.mean(amplitude[start:end])
            windowed = segment * window
            
            spectrum = fft.fft(windowed)[:n_freqs]
            amplitude_matrix[i, :] = np.abs(spectrum) * 2 / np.sum(window)
            
            # Tiempo central del segmento
            times[i] = (start + end) / 2 / self.fs
        
        # Filtrar por frecuencia máxima si se especifica
        if max_freq is not None:
            freq_resolution = self.fs / segment_length
            max_freq_idx = int(max_freq / freq_resolution)
            amplitude_matrix = amplitude_matrix[:, :max_freq_idx]
            frequencies = np.fft.fftfreq(segment_length, 1.0 / self.fs)[:max_freq_idx]
        else:
            frequencies = np.fft.fftfreq(segment_length, 1.0 / self.fs)[:n_freqs]
        
        return WaterfallResult(
            frequencies=frequencies,
            times=times,
            amplitude_matrix=amplitude_matrix,
        )
    
    def get_peak_evolution(
        self,
        waterfall_result: WaterfallResult,
        frequency_range: Tuple[float, float],
        n_peaks: int = 3,
    ) -> List[dict]:
        """
        Obtiene la evolución de los picos en el tiempo.
        
        Args:
            waterfall_result: WaterfallResult
            frequency_range: Rango de frecuencias a analizar
            n_peaks: Número de picos a trackear
            
        Returns:
            Lista de diccionarios con la evolución de cada pico
        """
        f = waterfall_result.frequencies
        matrix = waterfall_result.amplitude_matrix
        
        # Filtrar por rango de frecuencias
        mask = (f >= frequency_range[0]) & (f <= frequency_range[1])
        if not np.any(mask):
            return []
        
        f_filtered = f[mask]
        matrix_filtered = matrix[:, mask]
        
        evolution = []
        
        for time_idx in range(len(waterfall_result.times)):
            segment = matrix_filtered[time_idx, :]
            peak_indices = np.argsort(segment)[-n_peaks:][::-1]
            
            for rank, peak_idx in enumerate(peak_indices):
                if rank >= len(evolution):
                    evolution.append({
                        'rank': rank + 1,
                        'frequencies': [],
                        'amplitudes': [],
                        'times': [],
                    })
                
                evolution[rank]['times'].append(float(waterfall_result.times[time_idx]))
                evolution[rank]['frequencies'].append(float(f_filtered[peak_idx]))
                evolution[rank]['amplitudes'].append(float(segment[peak_idx]))
        
        return evolution
    
    def identify_constant_modes(
        self,
        waterfall_result: WaterfallResult,
        freq_tolerance: float = 0.5,
        amplitude_threshold_ratio: float = 0.3,
    ) -> List[dict]:
        """
        Identifica modos que permanecen constantes durante el tiempo.
        
        Estos modos representan frecuencias naturales del puente.
        
        Args:
            waterfall_result: WaterfallResult
            freq_tolerance: Tolerancia de frecuencia para agrupar (Hz)
            amplitude_threshold_ratio: Ratio mínimo de presencia temporal
            
        Returns:
            Lista de modos constantes identificados
        """
        f = waterfall_result.frequencies
        matrix = waterfall_result.amplitude_matrix
        
        # Encontrar picos en cada slice
        peak_tracks = []  # Lista de tracks por pico
        
        for time_idx in range(len(waterfall_result.times)):
            segment = matrix[time_idx, :]
            peaks, _ = signal.find_peaks(
                segment,
                height=amplitude_threshold_ratio * np.max(segment),
            )
            
            for peak_idx in peaks:
                freq = f[peak_idx]
                amp = segment[peak_idx]
                
                # Buscar si ya existe un track cercano
                found = False
                for track in peak_tracks:
                    # Verificar si la frecuencia está cerca del último punto del track
                    if len(track['indices']) > 0:
                        last_idx = track['indices'][-1]
                        if abs(f[last_idx] - freq) < freq_tolerance:
                            track['indices'].append(peak_idx)
                            track['amplitudes'].append(amp)
                            track['times'].append(waterfall_result.times[time_idx])
                            found = True
                            break
                
                if not found:
                    peak_tracks.append({
                        'indices': [peak_idx],
                        'amplitudes': [amp],
                        'times': [waterfall_result.times[time_idx]],
                        'frequencies': [freq],
                    })
        
        # Filtrar tracks que aparecen en suficiente tiempo
        min_appearances = int(0.3 * len(waterfall_result.times))
        
        constant_modes = []
        for track in peak_tracks:
            if len(track['indices']) >= min_appearances:
                avg_freq = np.mean(track['frequencies'])
                avg_amp = np.mean(track['amplitudes'])
                std_freq = np.std(track['frequencies'])
                
                constant_modes.append({
                    'average_frequency_hz': float(avg_freq),
                    'frequency_stability_hz': float(std_freq),
                    'avg_amplitude': float(avg_amp),
                    'presence_ratio': len(track['indices']) / len(waterfall_result.times),
                    'n_appearances': len(track['indices']),
                    'is_stable': std_freq < freq_tolerance,
                })
        
        # Ordenar por amplitud
        constant_modes.sort(key=lambda x: x['avg_amplitude'], reverse=True)
        
        return constant_modes