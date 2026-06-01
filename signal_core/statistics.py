"""
statistics.py - Estadísticas y Métricas de Señales
===================================================
Estadísticas completas, RMS, crest factor, Kurtosis, y más.
Incluye métricas específicas para puentes y análisis estructural.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, List, Tuple
import numpy as np
from scipy import stats


@dataclass
class SignalStatistics:
    """
    Estadísticas completas de una señal.
    
    Attributes:
        mean: Media
        std: Desviación estándar
        rms: Valor RMS (Root Mean Square)
        peak: Pico máximo (absoluto)
        peak_to_peak: Rango pico a pico
        crest_factor: Factor de cresta (peak/RMS)
        skewness: Asimetría
        kurtosis: Kurtosis
        median: Mediana
        iqr: Rango intercuartil
    """
    mean: float
    std: float
    rms: float
    peak: float
    peak_to_peak: float
    crest_factor: float
    skewness: float
    kurtosis: float
    median: float
    iqr: float
    
    def to_dict(self) -> dict:
        """Convierte a diccionario serializable."""
        return {
            'mean': self.mean,
            'std': self.std,
            'rms': self.rms,
            'peak': self.peak,
            'peak_to_peak': self.peak_to_peak,
            'crest_factor': self.crest_factor,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'median': self.median,
            'iqr': self.iqr,
        }


@dataclass
class FrequencyStatistics:
    """
    Estadísticas en el dominio de la frecuencia.
    
    Attributes:
        dominant_frequencies: Frecuencias dominantes
        bandwidth: Ancho de banda efectivo
        spectral_centroid: Centroide espectral
        spectral_rms: RMS espectral
    """
    dominant_frequencies: List[float]
    dominant_amplitudes: List[float]
    bandwidth_hz: float
    spectral_centroid_hz: float
    spectral_rms_hz: float
    total_spectral_energy: float
    
    def to_dict(self) -> dict:
        """Convierte a diccionario serializable."""
        return {
            'dominant_frequencies': self.dominant_frequencies,
            'dominant_amplitudes': self.dominant_amplitudes,
            'bandwidth_hz': self.bandwidth_hz,
            'spectral_centroid_hz': self.spectral_centroid_hz,
            'spectral_rms_hz': self.spectral_rms_hz,
            'total_spectral_energy': self.total_spectral_energy,
        }


@dataclass
class VibrationSeverity:
    """
    Evaluación de severidad de vibración para puentes.
    
    Basado en normativas como ISO 10816, DIN 4150, etc.
    
    Attributes:
        overall_rms: RMS global
        severity_class: Clase de severidad (1-4)
        severity_description: Descripción textual
        meets_standard: Si cumple con los límites
        recommendation: Recomendación
    """
    overall_rms: float
    severity_class: int
    severity_description: str
    meets_standard: bool
    recommendation: str
    
    def to_dict(self) -> dict:
        """Convierte a diccionario serializable."""
        return {
            'overall_rms_mm_s': self.overall_rms,
            'severity_class': self.severity_class,
            'severity_description': self.severity_description,
            'meets_standard': self.meets_standard,
            'recommendation': self.recommendation,
        }


class SignalStatisticsCalculator:
    """
    Calculador de estadísticas para señales de vibración.
    """
    
    def __init__(self):
        """Inicializa el calculador."""
        pass
    
    def calculate(
        self,
        amplitude: np.ndarray,
    ) -> SignalStatistics:
        """
        Calcula estadísticas completas de la señal.
        
        Args:
            amplitude: Array de amplitudes
            
        Returns:
            SignalStatistics
        """
        mean = np.mean(amplitude)
        std = np.std(amplitude)
        rms = np.sqrt(np.mean(amplitude**2))
        peak = np.max(np.abs(amplitude))
        peak_to_peak = np.max(amplitude) - np.min(amplitude)
        
        crest_factor = peak / rms if rms > 0 else 0
        
        skew = stats.skew(amplitude)
        kurt = stats.kurtosis(amplitude)
        
        median = np.median(amplitude)
        q75, q25 = np.percentile(amplitude, [75, 25])
        iqr = q75 - q25
        
        return SignalStatistics(
            mean=mean,
            std=std,
            rms=rms,
            peak=peak,
            peak_to_peak=peak_to_peak,
            crest_factor=crest_factor,
            skewness=skew,
            kurtosis=kurt,
            median=median,
            iqr=iqr,
        )
    
    def get_all_metrics(self, amplitude: np.ndarray) -> dict:
        """
        Obtiene todas las métricas en un diccionario.
        
        Args:
            amplitude: Array de amplitudes
            
        Returns:
            Diccionario con todas las métricas
        """
        stats_obj = self.calculate(amplitude)
        return stats_obj.to_dict()
    
    def calculate_rms_by_segment(
        self,
        amplitude: np.ndarray,
        n_segments: int = 10,
    ) -> List[float]:
        """
        Calcula RMS por segmentos.
        
        Args:
            amplitude: Array de amplitudes
            n_segments: Número de segmentos
            
        Returns:
            Lista de RMS por segmento
        """
        segment_size = len(amplitude) // n_segments
        rms_values = []
        
        for i in range(n_segments):
            start = i * segment_size
            end = start + segment_size if i < n_segments - 1 else len(amplitude)
            segment = amplitude[start:end]
            rms = np.sqrt(np.mean(segment**2))
            rms_values.append(rms)
        
        return rms_values
    
    def calculate_peak_by_segment(
        self,
        amplitude: np.ndarray,
        n_segments: int = 10,
    ) -> List[float]:
        """
        Calcula pico máximo por segmento.
        
        Args:
            amplitude: Array de amplitudes
            n_segments: Número de segmentos
            
        Returns:
            Lista de picos por segmento
        """
        segment_size = len(amplitude) // n_segments
        peak_values = []
        
        for i in range(n_segments):
            start = i * segment_size
            end = start + segment_size if i < n_segments - 1 else len(amplitude)
            segment = amplitude[start:end]
            peak = np.max(np.abs(segment))
            peak_values.append(peak)
        
        return peak_values
    
    def calculate_running_rms(
        self,
        amplitude: np.ndarray,
        window_samples: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula RMS móvil (running RMS).
        
        Args:
            amplitude: Array de amplitudes
            window_samples: Tamaño de la ventana
            
        Returns:
            Tuple de (tiempos, rms_values) - approximation
        """
        rms_values = []
        half_window = window_samples // 2
        
        for i in range(len(amplitude)):
            start = max(0, i - half_window)
            end = min(len(amplitude), i + half_window)
            segment = amplitude[start:end]
            rms = np.sqrt(np.mean(segment**2))
            rms_values.append(rms)
        
        times = np.arange(len(amplitude))
        return np.array(times), np.array(rms_values)
    
    def calculate_running_peak(
        self,
        amplitude: np.ndarray,
        window_samples: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula pico móvil.
        
        Args:
            amplitude: Array de amplitudes
            window_samples: Tamaño de la ventana
            
        Returns:
            Tuple de (tiempos, peak_values)
        """
        peak_values = []
        half_window = window_samples // 2
        
        for i in range(len(amplitude)):
            start = max(0, i - half_window)
            end = min(len(amplitude), i + half_window)
            segment = amplitude[start:end]
            peak = np.max(np.abs(segment))
            peak_values.append(peak)
        
        times = np.arange(len(amplitude))
        return np.array(times), np.array(peak_values)
    
    def estimate_noise_floor(
        self,
        amplitude: np.ndarray,
        percentile: float = 10,
    ) -> float:
        """
        Estima el piso de ruido (noise floor).
        
        Args:
            amplitude: Array de amplitudes
            percentile: Percentil para estimar ruido
            
        Returns:
            Piso de ruido estimado
        """
        # Usar el percentil inferior como estimado del piso de ruido
        return float(np.percentile(np.abs(amplitude), percentile))
    
    def calculate_signal_to_noise(
        self,
        amplitude: np.ndarray,
        noise_percentile: float = 10,
    ) -> float:
        """
        Calcula la relación señal/ruido.
        
        Args:
            amplitude: Array de amplitudes
            noise_percentile: Percentil para estimar ruido
            
        Returns:
            SNR en dB
        """
        signal_rms = np.sqrt(np.mean(amplitude**2))
        noise_floor = self.estimate_noise_floor(amplitude, noise_percentile)
        
        if noise_floor > 0:
            snr = 20 * np.log10(signal_rms / noise_floor)
        else:
            snr = float('inf')
        
        return snr
    
    def detect_saturation(
        self,
        amplitude: np.ndarray,
        threshold_ratio: float = 0.95,
    ) -> dict:
        """
        Detecta saturación en la señal.
        
        Args:
            amplitude: Array de amplitudes
            threshold_ratio: Ratio para considerar saturación
            
        Returns:
            Diccionario con resultado
        """
        max_abs = np.max(np.abs(amplitude))
        
        # Contar muestras cerca del máximo
        saturated_count = np.sum(np.abs(amplitude) > threshold_ratio * max_abs)
        saturation_ratio = saturated_count / len(amplitude)
        
        return {
            'is_saturated': saturation_ratio > 0.01,
            'saturation_ratio': saturation_ratio,
            'max_amplitude': max_abs,
            'warning': 'Posible saturación' if saturation_ratio > 0.01 else 'OK',
        }


class FrequencyStatisticsCalculator:
    """
    Calculador de estadísticas en el dominio de la frecuencia.
    """
    
    def __init__(self, fs: float):
        """
        Inicializa el calculador.
        
        Args:
            fs: Frecuencia de muestreo en Hz
        """
        self.fs = fs
    
    def calculate(
        self,
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
        n_dominant: int = 5,
    ) -> FrequencyStatistics:
        """
        Calcula estadísticas espectrales.
        
        Args:
            frequencies: Array de frecuencias
            amplitudes: Array de amplitudes
            n_dominant: Número de picos dominantes
            
        Returns:
            FrequencyStatistics
        """
        # Encontrar picos dominantes
        from scipy import signal
        peaks, _ = signal.find_peaks(amplitudes, height=0.1 * np.max(amplitudes))
        
        if len(peaks) == 0:
            return FrequencyStatistics(
                dominant_frequencies=[],
                dominant_amplitudes=[],
                bandwidth_hz=0.0,
                spectral_centroid_hz=0.0,
                spectral_rms_hz=0.0,
                total_spectral_energy=0.0,
            )
        
        # Ordenar por amplitud
        peak_amplitudes = amplitudes[peaks]
        order = np.argsort(peak_amplitudes)[::-1][:n_dominant]
        peaks = peaks[order]
        
        dominant_freqs = frequencies[peaks].tolist()
        dominant_amps = amplitudes[peaks].tolist()
        
        # Ancho de banda
        bandwidth = np.max(frequencies) - np.min(frequencies)
        
        # Centroide espectral
        spectral_sum = np.sum(amplitudes)
        if spectral_sum > 0:
            centroid = np.sum(frequencies * amplitudes) / spectral_sum
        else:
            centroid = 0
        
        # RMS espectral
        spectral_rms = np.sqrt(np.sum(amplitudes ** 2) / len(amplitudes))
        
        # Energía total
        total_energy = np.sum(amplitudes ** 2)
        
        return FrequencyStatistics(
            dominant_frequencies=dominant_freqs,
            dominant_amplitudes=dominant_amps,
            bandwidth_hz=float(bandwidth),
            spectral_centroid_hz=float(centroid),
            spectral_rms_hz=float(spectral_rms),
            total_spectral_energy=float(total_energy),
        )
    
    def calculate_band_energy(
        self,
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
        band_edges: List[Tuple[float, float]],
    ) -> List[float]:
        """
        Calcula energía en bandas de frecuencia.
        
        Args:
            frequencies: Array de frecuencias
            amplitudes: Array de amplitudes
            band_edges: Lista de tuples (f_low, f_high)
            
        Returns:
            Lista de energías por banda
        """
        energies = []
        
        for f_low, f_high in band_edges:
            mask = (frequencies >= f_low) & (frequencies <= f_high)
            band_energy = np.sum(amplitudes[mask] ** 2)
            energies.append(float(band_energy))
        
        return energies


class VibrationSeverityEvaluator:
    """
    Evaluador de severidad de vibración para puentes.
    
    Implementa criterios de normas como ISO 10816 para
    evaluación de vibración en máquinas y estructuras.
    """
    
    def __init__(self, fs: float):
        """
        Inicializa el evaluador.
        
        Args:
            fs: Frecuencia de muestreo en Hz
        """
        self.fs = fs
    
    def evaluate(
        self,
        amplitude: np.ndarray,
        unit: str = 'mm/s',
        standard: str = 'iso_10816',
        structure_type: str = 'bridge',
    ) -> VibrationSeverity:
        """
        Evalúa la severidad de vibración.
        
        Args:
            amplitude: Array de amplitudes (velocidad en mm/s)
            unit: Unidad de la señal
            standard: Norma a aplicar
            structure_type: Tipo de estructura
            
        Returns:
            VibrationSeverity
        """
        # Calcular RMS
        rms = np.sqrt(np.mean(amplitude**2))
        
        # Convertir si es necesario
        if unit == 'm/s²':
            # Convertir aceleración a velocidad RMS aproximada
            # v_rms ≈ a_rms / (2πf)
            # Asumiendo f ≈ 10 Hz como frecuencia característica
            f_char = 10.0
            rms = rms / (2 * np.pi * f_char)
        elif unit == 'g':
            rms = rms * 9.81 / (2 * np.pi * 10)
        
        # Clasificar según ISO 10816 (adaptado para puentes)
        if structure_type == 'bridge':
            return self._evaluate_bridge(rms)
        else:
            return self._evaluate_generic(rms)
    
    def _evaluate_bridge(self, rms: float) -> VibrationSeverity:
        """
        Evalúa severidad para puentes.
        
        Valores típicos:
        - Zone A (Good): < 3.5 mm/s RMS
        - Zone B (Acceptable): 3.5 - 7 mm/s RMS
        - Zone C (Unsatisfactory): 7 - 18 mm/s RMS
        - Zone D (Unacceptable): > 18 mm/s RMS
        """
        if rms < 3.5:
            severity_class = 1
            severity_desc = "Excelente - Vibración muy baja"
            meets_standard = True
            recommendation = "Continuar monitoreo regular"
        elif rms < 7.0:
            severity_class = 2
            severity_desc = "Aceptable - Vibración moderada"
            meets_standard = True
            recommendation = "Monitoreo periódico recomendado"
        elif rms < 18.0:
            severity_class = 3
            severity_desc = "Inadecuado - Vibración significativa"
            meets_standard = False
            recommendation = "Inspección detallada recomendada"
        else:
            severity_class = 4
            severity_desc = "Inaceptable - Vibración excesiva"
            meets_standard = False
            recommendation = "Acción correctiva inmediata requerida"
        
        return VibrationSeverity(
            overall_rms=rms,
            severity_class=severity_class,
            severity_description=severity_desc,
            meets_standard=meets_standard,
            recommendation=recommendation,
        )
    
    def _evaluate_generic(self, rms: float) -> VibrationSeverity:
        """
        Evalúa severidad genérica.
        """
        if rms < 2.0:
            severity_class = 1
            severity_desc = "Excelente"
            meets_standard = True
            recommendation = "OK"
        elif rms < 4.5:
            severity_class = 2
            severity_desc = "Aceptable"
            meets_standard = True
            recommendation = "Monitorear"
        elif rms < 11.0:
            severity_class = 3
            severity_desc = "Inadecuado"
            meets_standard = False
            recommendation = "Inspeccionar"
        else:
            severity_class = 4
            severity_desc = "Inaceptable"
            meets_standard = False
            recommendation = "Acción requerida"
        
        return VibrationSeverity(
            overall_rms=rms,
            severity_class=severity_class,
            severity_description=severity_desc,
            meets_standard=meets_standard,
            recommendation=recommendation,
        )
    
    def evaluate_acceleration(
        self,
        acceleration: np.ndarray,
        duration: float,
    ) -> VibrationSeverity:
        """
        Evalúa severidad desde aceleración.
        
        Args:
            acceleration: Array de aceleración (m/s² o g)
            duration: Duración en segundos
            
        Returns:
            VibrationSeverity
        """
        # RMS de aceleración
        rms_acc = np.sqrt(np.mean(acceleration**2))
        
        # Convertir a velocidad RMS (estimación simplificada)
        # Asumiendo contenido de frecuencia característico
        f_char = 5.0  # Hz característico para puentes
        
        # v_rms ≈ a_rms / (2πf)
        rms_vel = rms_acc / (2 * np.pi * f_char)
        
        # Convertir a mm/s
        rms_vel_mm_s = rms_vel * 1000
        
        return self._evaluate_bridge(rms_vel_mm_s)