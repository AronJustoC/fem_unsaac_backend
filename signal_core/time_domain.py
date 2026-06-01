"""
time_domain.py - Análisis en el Dominio del Tiempo
===================================================
Análisis temporal de señales de vibración para puentes y estructuras.
Incluye: historial temporal, estadísticas, segmentación, detección de eventos.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple, List
import numpy as np
from scipy import signal


@dataclass
class TimeHistoryResult:
    """
    Resultados del análisis temporal.
    
    Attributes:
        time: Array de tiempo
        amplitude: Array de amplitudes
        statistics: Diccionario de estadísticas
        peaks: Lista de picos detectados
        crossings: Número de cruces por cero
        crest_factor: Factor de cresta
    """
    time: np.ndarray
    amplitude: np.ndarray
    statistics: dict
    peaks: np.ndarray
    peak_times: np.ndarray
    crossings: int
    crest_factor: float
    
    def to_dict(self) -> dict:
        """Convierte a diccionario serializable."""
        return {
            'time': self.time.tolist(),
            'amplitude': self.amplitude.tolist(),
            'statistics': self.statistics,
            'n_peaks': len(self.peaks),
            'n_crossings': self.crossings,
            'crest_factor': self.crest_factor,
        }


class TimeDomainAnalyzer:
    """
    Analizador en el dominio del tiempo.
    
    Proporciona herramientas para analizar señales temporales de vibración
    en puentes, incluyendo estadísticas, detección de eventos, y segmentación.
    """
    
    def __init__(self, fs: float):
        """
        Inicializa el analizador.
        
        Args:
            fs: Frecuencia de muestreo en Hz
        """
        self.fs = fs
        self.dt = 1.0 / fs if fs > 0 else 0.0
    
    def analyze(
        self,
        time: np.ndarray,
        amplitude: np.ndarray,
        segment_duration: Optional[float] = None
    ) -> TimeHistoryResult:
        """
        Análisis completo del historial temporal.
        
        Args:
            time: Array de tiempo
            amplitude: Array de amplitudes
            segment_duration: Duración de segmentos para análisis parcial (opcional)
            
        Returns:
            TimeHistoryResult con el análisis completo
        """
        # Detectar picos
        peaks, peak_times = self.detect_peaks(amplitude, time)
        
        # Cruces por cero
        crossings = self.count_zero_crossings(amplitude)
        
        # Factor de cresta
        rms = np.sqrt(np.mean(amplitude**2))
        peak = np.max(np.abs(amplitude))
        crest_factor = peak / rms if rms > 0 else 0
        
        # Estadísticas
        stats = self.calculate_statistics(amplitude)
        
        return TimeHistoryResult(
            time=time,
            amplitude=amplitude,
            statistics=stats,
            peaks=peaks,
            peak_times=peak_times,
            crossings=crossings,
            crest_factor=crest_factor
        )
    
    def calculate_statistics(self, amplitude: np.ndarray) -> dict:
        """
        Calcula estadísticas básicas de la señal.
        
        Args:
            amplitude: Array de amplitudes
            
        Returns:
            Diccionario con estadísticas
        """
        return {
            'mean': float(np.mean(amplitude)),
            'std': float(np.std(amplitude)),
            'min': float(np.min(amplitude)),
            'max': float(np.max(amplitude)),
            'peak': float(np.max(np.abs(amplitude))),
            'peak_to_peak': float(np.max(amplitude) - np.min(amplitude)),
            'rms': float(np.sqrt(np.mean(amplitude**2))),
            'variance': float(np.var(amplitude)),
            'skewness': float(self._skewness(amplitude)),
            'kurtosis': float(self._kurtosis(amplitude)),
        }
    
    def _skewness(self, data: np.ndarray) -> float:
        """Calcula el skewness (asimetría)."""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Calcula el kurtosis."""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def detect_peaks(
        self,
        amplitude: np.ndarray,
        time: Optional[np.ndarray] = None,
        height_threshold: Optional[float] = None,
        prominence: Optional[float] = None,
        distance: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detecta picos en la señal.
        
        Args:
            amplitude: Array de amplitudes
            time: Array de tiempo (opcional)
            height_threshold: Umbral mínimo de altura (default: rms)
            prominence: Prominencia mínima de los picos
            distance: Distancia mínima entre picos (en muestras)
            
        Returns:
            Tuple de (picos, tiempos_picos)
        """
        if height_threshold is None:
            height_threshold = np.sqrt(np.mean(amplitude**2))
        
        # Usar scipy.signal.find_peaks
        peaks, properties = signal.find_peaks(
            amplitude,
            height=height_threshold,
            prominence=prominence,
            distance=distance,
        )
        
        peak_values = amplitude[peaks]
        
        if time is not None:
            peak_times = time[peaks]
        else:
            peak_times = np.array([])
        
        return peak_values, peak_times
    
    def detect_valleys(
        self,
        amplitude: np.ndarray,
        time: Optional[np.ndarray] = None,
        depth_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detecta valles (picos negativos).
        
        Args:
            amplitude: Array de amplitudes
            time: Array de tiempo (opcional)
            depth_threshold: Umbral mínimo de profundidad
            
        Returns:
            Tuple de (valles, tiempos_valles)
        """
        if depth_threshold is None:
            depth_threshold = np.sqrt(np.mean(amplitude**2))
        
        # Invertir y detectar picos
        valleys, properties = signal.find_peaks(
            -amplitude,
            height=depth_threshold,
        )
        
        valley_values = amplitude[valles]
        
        if time is not None:
            valley_times = time[valles]
        else:
            valley_times = np.array([])
        
        return valley_values, valley_times
    
    def count_zero_crossings(self, amplitude: np.ndarray) -> int:
        """
        Cuenta el número de cruces por cero.
        
        Args:
            amplitude: Array de amplitudes
            
        Returns:
            Número de cruces por cero
        """
        return int(np.sum(np.diff(np.sign(amplitude)) != 0))
    
    def segment_by_events(
        self,
        time: np.ndarray,
        amplitude: np.ndarray,
        threshold: float,
        min_duration: float = 0.0,
        pre_event: float = 0.1,
        post_event: float = 0.5,
    ) -> List[dict]:
        """
        Segmenta la señal detectando eventos que superan un umbral.
        
        Args:
            time: Array de tiempo
            amplitude: Array de amplitudes
            threshold: Umbral para detectar eventos
            min_duration: Duración mínima del evento (s)
            pre_event: Tiempo antes del evento a incluir (s)
            post_event: Tiempo después del evento a incluir (s)
            
        Returns:
            Lista de eventos detectados con sus segmentos
        """
        above_threshold = np.abs(amplitude) > threshold
        dt = time[1] - time[0] if len(time) > 1 else 0
        min_samples = int(min_duration / dt) if dt > 0 else 0
        pre_samples = int(pre_event / dt) if dt > 0 else 0
        post_samples = int(post_event / dt) if dt > 0 else 0
        
        events = []
        in_event = False
        event_start = 0
        
        # Encontrar transiciones
        diff = np.diff(above_threshold.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        for start_idx in starts:
            # Buscar el final correspondiente
            end_idxs = ends[ends > start_idx]
            if len(end_idxs) == 0:
                end_idx = len(time) - 1
            else:
                end_idx = end_idxs[0]
            
            # Verificar duración mínima
            n_samples = end_idx - start_idx
            if n_samples >= min_samples:
                # Incluir pre y post event
                seg_start = max(0, start_idx - pre_samples)
                seg_end = min(len(time) - 1, end_idx + post_samples)
                
                events.append({
                    'start_time': float(time[start_idx]),
                    'end_time': float(time[end_idx]),
                    'start_idx': int(start_idx),
                    'end_idx': int(end_idx),
                    'segment_start_idx': int(seg_start),
                    'segment_end_idx': int(seg_end),
                    'peak_value': float(np.max(np.abs(amplitude[start_idx:end_idx + 1]))),
                    'peak_time': float(time[start_idx + np.argmax(np.abs(amplitude[start_idx:end_idx + 1]))]),
                })
        
        return events
    
    def extract_free_vibration(
        self,
        time: np.ndarray,
        amplitude: np.ndarray,
        trigger_threshold: float,
        trigger_idx: Optional[int] = None,
        segment_duration: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrae la respuesta de vibración libre después de un trigger.
        
        Especialmente útil para identificar modos naturales después del paso de vehículos.
        
        Args:
            time: Array de tiempo
            amplitude: Array de amplitudes
            trigger_threshold: Umbral de trigger
            trigger_idx: Índice de trigger manual (opcional)
            segment_duration: Duración del segmento a extraer (s)
            
        Returns:
            Tuple de (tiempo_segmento, amplitud_segmento)
        """
        if trigger_idx is None:
            # Encontrar el trigger automáticamente
            abs_amp = np.abs(amplitude)
            trigger_idx = np.argmax(abs_amp > trigger_threshold)
        
        dt = time[1] - time[0] if len(time) > 1 else 0
        if segment_duration is None:
            # Extraer hasta que la amplitud caiga al 10% del pico
            peak_amp = np.max(np.abs(amplitude[trigger_idx:]))
            threshold_end = 0.1 * peak_amp
            remaining = np.abs(amplitude[trigger_idx:])
            end_idxs = np.where(remaining < threshold_end)[0]
            if len(end_idxs) > 0:
                segment_samples = end_idxs[0] + 1
            else:
                segment_samples = len(time) - trigger_idx
        else:
            segment_samples = int(segment_duration / dt)
        
        end_idx = min(trigger_idx + segment_samples, len(time))
        
        segment_time = time[trigger_idx:end_idx] - time[trigger_idx]
        segment_amp = amplitude[trigger_idx:end_idx]
        
        return segment_time, segment_amp
    
    def estimate_damping_from_free_vibration(
        self,
        time: np.ndarray,
        amplitude: np.ndarray,
        method: Literal['logarithmic', 'peak'] = 'logarithmic',
    ) -> Optional[float]:
        """
        Estima el amortiguamiento a partir de la vibración libre.
        
        Args:
            time: Array de tiempo
            amplitude: Array de amplitudes (vibración libre)
            method: Método de estimación ('logarithmic' o 'peak')
            
        Returns:
            Ratio de amortiguamiento (ζ) o None si no se puede estimar
        """
        if method == 'logarithmic':
            return self._logarithmic_decrement(time, amplitude)
        else:
            return self._peak_decrement(time, amplitude)
    
    def _logarithmic_decrement(
        self,
        time: np.ndarray,
        amplitude: np.ndarray
    ) -> Optional[float]:
        """
        Estima amortiguamiento usando el decremento logarítmico.
        
        δ = (1/n) * ln(a_i / a_{i+n})
        ζ = δ / sqrt(4π² + δ²)
        """
        # Encontrar picos sucesivos
        peaks, _ = self.detect_peaks(amplitude)
        
        if len(peaks) < 2:
            return None
        
        # Calcular decremento logarítmico entre picos
        n = min(len(peaks) - 1, 5)  # Usar hasta 5 ciclos
        
        # Usar promedio de decrementos
        deltas = []
        for i in range(n):
            if peaks[i] > 0 and peaks[i + 1] > 0:
                delta = np.log(peaks[i] / peaks[i + 1])
                if delta > 0:
                    deltas.append(delta)
        
        if not deltas:
            return None
        
        avg_delta = np.mean(deltas)
        
        # Calcular ratio de amortiguamiento
        denominator = np.sqrt(4 * np.pi**2 + avg_delta**2)
        if denominator == 0:
            return None
        
        zeta = avg_delta / denominator
        
        return float(zeta)
    
    def _peak_decrement(
        self,
        time: np.ndarray,
        amplitude: np.ndarray
    ) -> Optional[float]:
        """
        Estima amortiguamiento usando峰值递减法.
        """
        peaks, _ = self.detect_peaks(amplitude)
        
        if len(peaks) < 2:
            return None
        
        # Ajuste lineal de ln(peaks) vs tiempo
        peak_times = []
        for i in range(len(peaks) - 1):
            idx_i = np.argmax(amplitude == peaks[i])
            idx_next = np.argmax(amplitude == peaks[i + 1])
            t_i = time[idx_i]
            t_next = time[idx_next]
            peak_times.append((t_i + t_next) / 2)
        
        peak_times = np.array(peak_times)
        
        if len(peak_times) < 2:
            return None
        
        ln_peaks = np.log(peaks[:-1])
        
        # Ajuste lineal
        coef = np.polyfit(peak_times, ln_peaks, 1)
        alpha = coef[0]  # Tasa de decaimiento
        
        # ζ = α / ω_d, donde ω_d ≈ ω_n para amortiguamiento bajo
        # Asumimos ω_n ≈ 2π * f Dominante
        # Para estimar f, contar oscilaciones en el tiempo
        crossings = self.count_zero_crossings(amplitude)
        if crossings > 0:
            duration = time[-1] - time[0]
            f = crossings / (2 * duration)
            omega_n = 2 * np.pi * f
            zeta = alpha / omega_n
            return float(zeta)
        
        return None
    
    def get_rms_by_bands(
        self,
        time: np.ndarray,
        amplitude: np.ndarray,
        band_edges: List[Tuple[float, float]],
    ) -> List[float]:
        """
        Calcula RMS en bandas de frecuencia específicas.
        
        Args:
            time: Array de tiempo
            amplitude: Array de amplitudes
            band_edges: Lista de tuples (f_low, f_high) para cada banda
            
        Returns:
            Lista de valores RMS por banda
        """
        from scipy import signal
        
        rms_values = []
        for f_low, f_high in band_edges:
            # Diseñar filtro butterworth para la banda
            nyquist = self.fs / 2
            if f_low >= nyquist or f_high > nyquist:
                rms_values.append(0.0)
                continue
            
            low_norm = f_low / nyquist
            high_norm = f_high / nyquist
            
            if low_norm == 0:
                # Filtro pasa-bajos
                b, a = signal.butter(4, high_norm, btype='low')
            elif high_norm >= 1:
                # Filtro pasa-altos
                b, a = signal.butter(4, low_norm, btype='high')
            else:
                # Filtro bandpass
                b, a = signal.butter(4, [low_norm, high_norm], btype='band')
            
            filtered = signal.filtfilt(b, a, amplitude)
            rms = np.sqrt(np.mean(filtered**2))
            rms_values.append(float(rms))
        
        return rms_values
    
    def calculate_instantaneous_amplitude(
        self,
        amplitude: np.ndarray,
        window_size: int = 101,
    ) -> np.ndarray:
        """
        Calcula la amplitud instantánea usando envolvente.
        
        Args:
            amplitude: Array de amplitudes
            window_size: Tamaño de la ventana para el suavizado
            
        Returns:
            Array de amplitudes instantáneas (envolvente)
        """
        # Hilbert transform
        analytic_signal = signal.hilbert(amplitude)
        envelope = np.abs(analytic_signal)
        
        # Suavizar con media móvil
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            envelope = np.convolve(envelope, kernel, mode='same')
        
        return envelope
    
    def calculate_rms_envelope(
        self,
        time: np.ndarray,
        amplitude: np.ndarray,
        window_duration: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula el RMS en una ventana móvil.
        
        Args:
            time: Array de tiempo
            amplitude: Array de amplitudes
            window_duration: Duración de la ventana en segundos
            
        Returns:
            Tuple de (tiempos, rms_values)
        """
        dt = time[1] - time[0] if len(time) > 1 else 0
        window_samples = max(1, int(window_duration / dt))
        
        rms_values = []
        center_times = []
        
        half_window = window_samples // 2
        
        for i in range(half_window, len(amplitude) - half_window):
            segment = amplitude[i - half_window:i + half_window]
            rms = np.sqrt(np.mean(segment**2))
            rms_values.append(rms)
            center_times.append(time[i])
        
        return np.array(center_times), np.array(rms_values)