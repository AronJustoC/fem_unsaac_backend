"""
filters.py - Diseño y Aplicación de Filtros Digitales
=====================================================
Filtros pasa-bajos, pasa-altos, bandpass, Butterworth, Chebyshev, etc.
Esenciales para limpiar señales antes de integración y análisis.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple, List
import numpy as np
from scipy import signal


@dataclass
class FilterSpec:
    """
    Especificación de un filtro.
    
    Attributes:
        filter_type: Tipo ('lowpass', 'highpass', 'bandpass', 'bandstop')
        order: Orden del filtro
        cutoff_freq: Frecuencia(s) de corte
        ripple: Rizado (para Chebyshev)
        sampling_rate: Frecuencia de muestreo
        description: Descripción del filtro
    """
    filter_type: str
    order: int
    cutoff_freq: Tuple[float, float] | float
    ripple: float = 0.5  # dB para Chebyshev
    sampling_rate: float = 100.0
    description: str = ""
    
    def get_critical_freq(self) -> Tuple[float, float] | float:
        """Retorna la frecuencia crítica normalizada."""
        if isinstance(self.cutoff_freq, tuple):
            return tuple(f / (self.sampling_rate / 2) for f in self.cutoff_freq)
        return self.cutoff_freq / (self.sampling_rate / 2)


@dataclass
class FilterResult:
    """
    Resultado de aplicar un filtro.
    
    Attributes:
        filtered_signal: Señal filtrada
        filter_spec: Especificación del filtro usado
        frequency_response: Respuesta en frecuencia del filtro
        frequencies: Array de frecuencias
        magnitude: Magnitud de la respuesta
        phase: Fase de la respuesta
    """
    filtered_signal: np.ndarray
    filter_spec: FilterSpec
    frequency_response: Optional[np.ndarray] = None
    frequencies: Optional[np.ndarray] = None
    magnitude: Optional[np.ndarray] = None
    phase: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        """Convierte a diccionario serializable."""
        return {
            'filter_type': self.filter_spec.filter_type,
            'order': self.filter_spec.order,
            'cutoff_freq': (self.filter_spec.cutoff_freq 
                          if isinstance(self.filter_spec.cutoff_freq, list) 
                          else [self.filter_spec.cutoff_freq]),
            'sampling_rate': self.filter_spec.sampling_rate,
            'filtered_signal_length': len(self.filtered_signal),
        }


class FilterDesign:
    """
    Diseño de filtros digitales.
    
    Proporciona métodos para diseñar filtros Butterworth, Chebyshev,
    Bessel y otros tipos comunes en análisis de señales de vibración.
    """
    
    def __init__(self, fs: float):
        """
        Inicializa el diseñador.
        
        Args:
            fs: Frecuencia de muestreo en Hz
        """
        self.fs = fs
        self.nyquist = fs / 2.0
    
    def butterworth_lowpass(
        self,
        cutoff: float,
        order: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diseña un filtro Butterworth pasa-bajos.
        
        Args:
            cutoff: Frecuencia de corte en Hz
            order: Orden del filtro (mayor orden = más pendiente)
            
        Returns:
            Tuple de (coeficientes b, a)
        """
        normalized_cutoff = cutoff / self.nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        
        return b, a
    
    def butterworth_highpass(
        self,
        cutoff: float,
        order: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diseña un filtro Butterworth pasa-altos.
        
        Args:
            cutoff: Frecuencia de corte en Hz
            order: Orden del filtro
            
        Returns:
            Tuple de (coeficientes b, a)
        """
        normalized_cutoff = cutoff / self.nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='high')
        
        return b, a
    
    def butterworth_bandpass(
        self,
        low_cutoff: float,
        high_cutoff: float,
        order: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diseña un filtro Butterworth pasa-banda.
        
        Args:
            low_cutoff: Frecuencia de corte baja en Hz
            high_cutoff: Frecuencia de corte alta en Hz
            order: Orden del filtro
            
        Returns:
            Tuple de (coeficientes b, a)
        """
        # Validar frecuencias
        low = max(0.001, min(low_cutoff, self.nyquist - 0.001))
        high = max(low + 0.001, min(high_cutoff, self.nyquist - 0.001))
        normalized = [low / self.nyquist, high / self.nyquist]
        b, a = signal.butter(order, normalized, btype='band')
        
        return b, a
    
    def butterworth_bandstop(
        self,
        low_cutoff: float,
        high_cutoff: float,
        order: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diseña un filtro Butterworth rechaza-banda (notch).
        
        Args:
            low_cutoff: Frecuencia de corte baja en Hz
            high_cutoff: Frecuencia de corte alta en Hz
            order: Orden del filtro
            
        Returns:
            Tuple de (coeficientes b, a)
        """
        normalized = [low_cutoff / self.nyquist, high_cutoff / self.nyquist]
        b, a = signal.butter(order, normalized, btype='bandstop')
        
        return b, a
    
    def chebyshev1_lowpass(
        self,
        cutoff: float,
        order: int = 4,
        ripple_db: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diseña un filtro Chebyshev Tipo I pasa-bajos.
        
        Ofrece mayor pendiente que Butterworth a costa de rizado.
        
        Args:
            cutoff: Frecuencia de corte en Hz
            order: Orden del filtro
            ripple_db: Rizado en la banda de paso (dB)
            
        Returns:
            Tuple de (coeficientes b, a)
        """
        normalized_cutoff = cutoff / self.nyquist
        b, a = signal.cheby1(order, ripple_db, normalized_cutoff, btype='low')
        
        return b, a
    
    def chebyshev2_lowpass(
        self,
        cutoff: float,
        stop_freq: float,
        order: int = 4,
        ripple_db: float = 40.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diseña un filtro Chebyshev Tipo II (inverse) pasa-bajos.
        
        Args:
            cutoff: Frecuencia de corte en Hz
            stop_freq: Frecuencia de corte del stopband
            order: Orden del filtro
            ripple_db: Atenuación en la banda de detenida (dB)
            
        Returns:
            Tuple de (coeficientes b, a)
        """
        normalized_cutoff = cutoff / self.nyquist
        normalized_stop = stop_freq / self.nyquist
        b, a = signal.cheby2(order, ripple_db, normalized_stop, btype='low')
        
        return b, a
    
    def elliptic_lowpass(
        self,
        cutoff: float,
        order: int = 4,
        ripple_db: float = 0.5,
        stopband_attenuation: float = 40.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diseña un filtro Elíptico (Cauer) pasa-bajos.
        
        Ofrece la máxima pendiente con rizado en ambas bandas.
        
        Args:
            cutoff: Frecuencia de corte en Hz
            order: Orden del filtro
            ripple_db: Rizado en la banda de paso (dB)
            stopband_attenuation: Atenuación en la banda de detenida (dB)
            
        Returns:
            Tuple de (coeficientes b, a)
        """
        normalized_cutoff = cutoff / self.nyquist
        b, a = signal.ellip(order, ripple_db, stopband_attenuation, normalized_cutoff, btype='low')
        
        return b, a
    
    def bessel_lowpass(
        self,
        cutoff: float,
        order: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diseña un filtro Bessel pasa-bajos.
        
        Mantiene fase lineal (retardo constante), ideal para señales一旦.
        
        Args:
            cutoff: Frecuencia de corte en Hz
            order: Orden del filtro
            
        Returns:
            Tuple de (coeficientes b, a)
        """
        normalized_cutoff = cutoff / self.nyquist
        b, a = signal.bessel(order, normalized_cutoff, btype='low')
        
        return b, a
    
    def notch_filter(
        self,
        freq: float,
        Q: float = 30.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diseña un filtro notch (rechaza-banda angosto).
        
        Ideal para eliminar ruido eléctrico (50/60 Hz).
        
        Args:
            freq: Frecuencia a eliminar en Hz
            Q: Factor Q (mayor = más angosto)
            
        Returns:
            Tuple de (coeficientes b, a)
        """
        w0 = freq / self.nyquist
        b, a = signal.iirnotch(w0, Q)
        
        return b, a
    
    def comb_filter(
        self,
        freq: float,
        n_harmonics: int = 5,
        bandwidth: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diseña un filtro peine (comb filter).
        
        Elimina una frecuencia fundamental y sus armónicos.
        
        Args:
            freq: Frecuencia fundamental en Hz
            n_harmonics: Número de armónicos a rechazar
            bandwidth: Ancho de banda por armónico
            
        Returns:
            Tuple de (coeficientes b, a)
        """
        b_total = np.array([1.0])
        a_total = np.array([1.0])
        
        for h in range(1, n_harmonics + 1):
            freq_h = freq * h
            if freq_h >= self.nyquist:
                break
            
            w0 = freq_h / self.nyquist
            bw = bandwidth / self.nyquist
            b, a = signal.iirnotch(w0, bw * self.nyquist / freq_h * 30)
            
            # Convolucionar coeficientes
            b_total = np.convolve(b_total, b)
            a_total = np.convolve(a_total, a)
        
        return b_total, a_total
    
    def moving_average_filter(
        self,
        window_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diseña un filtro de media móvil.
        
        Filtro FIR simple para suavizado.
        
        Args:
            window_samples: Número de muestras en la ventana
            
        Returns:
            Tuple de (coeficientes b, a)
        """
        b = np.ones(window_samples) / window_samples
        a = np.array([1.0])
        
        return b, a
    
    def polynomial_lowpass(
        self,
        order: int = 2,
        cutoff_ratio: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diseña un filtro polinomial (Savitzky-Golay).
        
        Suaviza mientras preserva picos y valles.
        
        Args:
            order: Orden del polinomio
            cutoff_ratio: Ratio de cutoff (0.0 a 1.0)
            
        Returns:
            Tuple de (coeficientes b, a) - approximation
        """
        # Para Savitzky-Golay necesitamos nperseg
        # Aquí usamos un aproximación IIR simple
        cutoff = cutoff_ratio * self.nyquist
        return self.butterworth_lowpass(cutoff, order)
    
    def get_frequency_response(
        self,
        b: np.ndarray,
        a: np.ndarray,
        n_points: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula la respuesta en frecuencia de un filtro.
        
        Args:
            b: Coeficientes del numerador
            a: Coeficientes del denominador
            n_points: Número de puntos de evaluación
            
        Returns:
            Tuple de (frecuencias, magnitud, fase)
        """
        w, h = signal.freqz(b, a, worN=n_points)
        frequencies = w * self.nyquist / np.pi
        magnitude = np.abs(h)
        phase = np.angle(h)
        
        return frequencies, magnitude, phase


class SignalFilter:
    """
    Aplicador de filtros a señales.
    
    Maneja la aplicación práctica de filtros con filtfilt
    (filtrado en ambas direcciones para evitar desplazamiento de fase).
    """
    
    def __init__(self, fs: float):
        """
        Inicializa el filtro.
        
        Args:
            fs a Frecuencia de muestreo en Hz
        """
        self.fs = fs
        self.designer = FilterDesign(fs)
    
    def apply(
        self,
        amplitude: np.ndarray,
        filter_type: Literal[
            'lowpass', 'highpass', 'bandpass', 'bandstop',
            'notch', 'comb', 'moving_average'
        ] = 'lowpass',
        order: int = 4,
        cutoff_freq: float | Tuple[float, float] = 10.0,
        **kwargs,
    ) -> np.ndarray:
        """
        Aplica un filtro a la señal.
        
        Args:
            amplitude: Array de entrada
            filter_type: Tipo de filtro
            order: Orden del filtro (para IIR)
            cutoff_freq: Frecuencia(s) de corte
            **kwargs: Parámetros adicionales
            
        Returns:
            Señal filtrada
        """
        # Convertir cutoff_freq a la forma correcta
        if isinstance(cutoff_freq, (int, float)):
            cutoff = float(cutoff_freq)
        else:
            cutoff = cutoff_freq
        
        # Diseñar filtro
        if filter_type == 'lowpass':
            b, a = self.designer.butterworth_lowpass(cutoff, order)
        elif filter_type == 'highpass':
            b, a = self.designer.butterworth_highpass(cutoff, order)
        elif filter_type == 'bandpass':
            if not isinstance(cutoff, tuple):
                raise ValueError("bandpass requiere tuple (low, high)")
            b, a = self.designer.butterworth_bandpass(cutoff[0], cutoff[1], order)
        elif filter_type == 'bandstop':
            if not isinstance(cutoff, tuple):
                raise ValueError("bandstop requiere tuple (low, high)")
            b, a = self.designer.butterworth_bandstop(cutoff[0], cutoff[1], order)
        elif filter_type == 'notch':
            b, a = self.designer.notch_filter(cutoff, kwargs.get('Q', 30.0))
        elif filter_type == 'comb':
            b, a = self.designer.comb_filter(
                cutoff,
                kwargs.get('n_harmonics', 5),
                kwargs.get('bandwidth', 1.0),
            )
        elif filter_type == 'moving_average':
            b, a = self.designer.moving_average_filter(int(cutoff))
        else:
            # Default: butterworth lowpass
            b, a = self.designer.butterworth_lowpass(cutoff, order)
        
        # Aplicar con filtfilt (bidireccional para fase cero)
        try:
            filtered = signal.filtfilt(b, a, amplitude)
        except Exception as e:
            print(f"Warning: filtfilt failed, using lfilter: {e}")
            filtered = signal.lfilter(b, a, amplitude)
        
        return filtered
    
    def apply_cascade(
        self,
        amplitude: np.ndarray,
        filters: List[dict],
    ) -> np.ndarray:
        """
        Aplica una cascada de filtros.
        
        Args:
            amplitude: Array de entrada
            filters: Lista de diccionarios con especificaciones de filtros
            
        Returns:
            Señal filtrada después de todos los filtros
        """
        result = amplitude.copy()
        
        for f in filters:
            result = self.apply(
                result,
                filter_type=f.get('type', 'lowpass'),
                order=f.get('order', 4),
                cutoff_freq=f.get('cutoff', 10.0),
                **f.get('kwargs', {}),
            )
        
        return result
    
    def remove_dc_offset(
        self,
        amplitude: np.ndarray,
        highpass_freq: float = 0.1,
    ) -> np.ndarray:
        """
        Elimina el offset DC usando un filtro pasa-altos.
        
        Args:
            amplitude: Array de entrada
            highpass_freq: Frecuencia de corte para eliminar DC
            
        Returns:
            Señal sin offset DC
        """
        return self.apply(
            amplitude,
            filter_type='highpass',
            order=2,
            cutoff_freq=highpass_freq,
        )
    
    def remove_powerline_noise(
        self,
        amplitude: np.ndarray,
        line_freq: float = 50.0,
        n_harmonics: int = 3,
    ) -> np.ndarray:
        """
        Elimina ruido de línea eléctrica.
        
        Args:
            amplitude: Array de entrada
            line_freq: Frecuencia de línea (50 o 60 Hz)
            n_harmonics: Número de armónicos a eliminar
            
        Returns:
            Señal sin ruido de línea
        """
        return self.apply(
            amplitude,
            filter_type='comb',
            cutoff_freq=line_freq,
            n_harmonics=n_harmonics,
            bandwidth=0.5,
        )
    
    def bandpass_for_modal_analysis(
        self,
        amplitude: np.ndarray,
        low_freq: float = 0.5,
        high_freq: float = 20.0,
        order: int = 4,
    ) -> np.ndarray:
        """
        Aplica bandpass optimizado para análisis modal de puentes.
        
        Args:
            amplitude: Array de entrada
            low_freq: Frecuencia baja (para remover drift)
            high_freq: Frecuencia alta (para remover ruido)
            order: Orden del filtro
            
        Returns:
            Señal filtrada
        """
        return self.apply(
            amplitude,
            filter_type='bandpass',
            order=order,
            cutoff_freq=(low_freq, high_freq),
        )
    
    def lowpass_for_noiseReduction(
        self,
        amplitude: np.ndarray,
        cutoff_freq: float = 50.0,
        order: int = 4,
    ) -> np.ndarray:
        """
        Pasa-bajos para reducción de ruido de alta frecuencia.
        
        Args:
            amplitude: Array de entrada
            cutoff_freq: Frecuencia de corte máxima
            order: Orden del filtro
            
        Returns:
            Señal filtrada
        """
        return self.apply(
            amplitude,
            filter_type='lowpass',
            order=order,
            cutoff_freq=cutoff_freq,
        )
    
    def get_filter_specification(
        self,
        filter_type: str,
        cutoff_freq: float | Tuple[float, float],
        order: int = 4,
    ) -> FilterSpec:
        """
        Genera una especificación legible del filtro.
        
        Args:
            filter_type: Tipo de filtro
            cutoff_freq: Frecuencia(s) de corte
            order: Orden del filtro
            
        Returns:
            FilterSpec con descripción
        """
        if isinstance(cutoff_freq, tuple):
            cutoff_str = f"{cutoff_freq[0]}-{cutoff_freq[1]} Hz"
        else:
            cutoff_str = f"{cutoff_freq} Hz"
        
        description = f"{filter_type} Butterworth orden {order}, fc={cutoff_str}, fs={self.fs} Hz"
        
        return FilterSpec(
            filter_type=filter_type,
            order=order,
            cutoff_freq=cutoff_freq,
            sampling_rate=self.fs,
            description=description,
        )