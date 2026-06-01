"""
integration.py - Integración y Diferenciación de Señales
========================================================
Integración para obtener velocidad y desplazamiento desde aceleración.
Incluye filtros para evitar drift y méthodes de integración numérica.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
from scipy import integrate, signal


@dataclass
class IntegrationResult:
    """
    Resultado de la integración.
    
    Attributes:
        time: Array de tiempo
        integrated_signal: Señal integrada
        drift_warning: Advertencia sobre drift
        filter_applied: Filtro aplicado antes de integración
        baseline_correction: Si se aplicó corrección de línea base
    """
    time: np.ndarray
    integrated_signal: np.ndarray
    drift_warning: bool = False
    filter_applied: Optional[str] = None
    baseline_correction: bool = False
    
    def to_dict(self) -> dict:
        """Convierte a diccionario serializable."""
        return {
            'time': self.time.tolist(),
            'n_samples': len(self.integrated_signal),
            'drift_warning': self.drift_warning,
            'filter_applied': self.filter_applied,
            'baseline_corrected': self.baseline_correction,
            'initial_value': float(self.integrated_signal[0]) if len(self.integrated_signal) > 0 else 0,
            'final_value': float(self.integrated_signal[-1]) if len(self.integrated_signal) > 0 else 0,
            'peak_value': float(np.max(np.abs(self.integrated_signal))) if len(self.integrated_signal) > 0 else 0,
        }


class SignalIntegrator:
    """
    Integrador de señales de vibración.
    
    Integra aceleración -> velocidad -> desplazamiento
    con filtrado para minimizar drift.
    """
    
    def __init__(self, fs: float, highpass_freq: float = 0.5):
        """
        Inicializa el integrador.
        
        Args:
            fs: Frecuencia de muestreo en Hz
            highpass_freq: Frecuencia de corte pasa-altos para evitar drift
        """
        self.fs = fs
        self.highpass_freq = highpass_freq
        self.nyquist = fs / 2.0
        
        # Verificar frecuencia de corte válida
        if highpass_freq >= self.nyquist:
            self.highpass_freq = 0.1
    
    def integrate_acceleration(
        self,
        acceleration: np.ndarray,
        time: Optional[np.ndarray] = None,
        double_integrate: bool = False,
        remove_initial_velocity: bool = True,
        baseline_correct: bool = True,
    ) -> np.ndarray:
        """
        Integra aceleración para obtener velocidad o desplazamiento.
        
        Args:
            acceleration: Array de aceleración
            time: Array de tiempo (opcional, usa fs si no se provee)
            double_integrate: Si hacer doble integración (a->v->d)
            remove_initial_velocity: Si intentar remover velocidad inicial
            baseline_correct: Si aplicar corrección de línea base
            
        Returns:
            Array de velocidad o desplazamiento
        """
        # Calcular dt
        if time is not None and len(time) > 1:
            dt = np.mean(np.diff(time))
        else:
            dt = 1.0 / self.fs
        
        # Aplicar filtro pasa-altos antes de integración
        acc_filtered = self._apply_highpass(acceleration)
        
        # Primera integración: a -> v
        velocity = self._cumtrapz(acc_filtered, dt)
        
        if double_integrate:
            # Aplicar filtro pasa-altos nuevamente para reducir drift
            velocity = self._apply_highpass(velocity)
            
            # Segunda integración: v -> d
            displacement = self._cumtrapz(velocity, dt)
            
            # Corrección de línea base si se solicita
            if baseline_correct:
                displacement = self._remove_baseline(displacement)
            
            return displacement
        else:
            # Corrección de línea base
            if baseline_correct:
                velocity = self._remove_baseline(velocity)
            
            return velocity
    
    def differentiate_signal(
        self,
        displacement: np.ndarray,
        time: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Diferencia una señal para obtener velocidad o aceleración.
        
        Args:
            displacement: Array de desplazamiento
            time: Array de tiempo (opcional)
            
        Returns:
            Array de velocidad
        """
        if time is not None and len(time) > 1:
            dt = np.mean(np.diff(time))
        else:
            dt = 1.0 / self.fs
        
        # Diferenciación usando gradient
        derivative = np.gradient(displacement, dt)
        
        return derivative
    
    def integrate_with_trapezoid(
        self,
        amplitude: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Integración usando la regla trapezoidal.
        
        Args:
            amplitude: Array de entrada
            dt: Incremento de tiempo
            
        Returns:
            Array integrado
        """
        return integrate.cumulative_trapezoid(amplitude, dx=dt, initial=0)
    
    def integrate_with_simpson(
        self,
        amplitude: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Integración usando la regla de Simpson.
        
        Args:
            amplitude: Array de entrada
            dt: Incremento de tiempo
            
        Returns:
            Array integrado
        """
        n = len(amplitude)
        
        if n < 3:
            return self.integrate_with_trapezoid(amplitude, dt)
        
        result = np.zeros(n)
        
        # Simpson para múltiples de 2 puntos
        m = (n - 1) // 2
        
        # Acumular usando Simpson
        for i in range(m):
            idx_h = 2 * i
            idx_m = 2 * i + 1
            idx_l = 2 * i + 2
            
            h = dt
            
            # Simpson: h/6 * (y0 + 4*y1 + y2)
            if idx_l < n:
                result[idx_l] = result[idx_h] + (h / 6) * (
                    amplitude[idx_h] + 4 * amplitude[idx_m] + amplitude[idx_l]
                )
        
        # Completar últimos puntos si n es impar
        if (n - 1) % 2 != 0 and n > 2:
            result[-1] = result[-2] + (dt / 2) * (
                amplitude[-2] + amplitude[-1]
            )
        
        return result
    
    def _cumtrapz(
        self,
        amplitude: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Integración acumulativa trapezoidal con valor inicial cero.
        
        Args:
            amplitude: Array de entrada
            dt: Incremento de tiempo
            
        Returns:
            Array integrado
        """
        return integrate.cumulative_trapezoid(amplitude, dx=dt, initial=0)
    
    def _apply_highpass(self, amplitude: np.ndarray) -> np.ndarray:
        """
        Aplica filtro pasa-altos para minimizar drift.
        
        Args:
            amplitude: Array de entrada
            
        Returns:
            Array filtrado
        """
        if self.highpass_freq <= 0:
            return amplitude
        
        # Diseñar filtro pasa-altos
        nyquist = self.fs / 2
        cutoff_norm = self.highpass_freq / nyquist
        
        # Usar filtro de primer orden simple
        b, a = signal.butter(2, max(0.001, cutoff_norm), btype='high')
        
        # Aplicar bidireccional
        try:
            filtered = signal.filtfilt(b, a, amplitude)
        except Exception:
            filtered = amplitude - np.mean(amplitude)
        
        return filtered
    
    def _remove_baseline(self, amplitude: np.ndarray) -> np.ndarray:
        """
        Elimina tendencia de línea base.
        
        Args:
            amplitude: Array de entrada
            
        Returns:
            Array con tendencia removida
        """
        # Remover media
        amplitude = amplitude - np.mean(amplitude)
        
        # Detrend lineal si hay estructura
        if len(amplitude) > 10:
            x = np.arange(len(amplitude))
            coef = np.polyfit(x, amplitude, 1)
            trend = np.polyval(coef, x)
            amplitude = amplitude - trend
        
        return amplitude
    
    def check_drift(
        self,
        displacement: np.ndarray,
        threshold_ratio: float = 10.0,
    ) -> dict:
        """
        Verifica si hay drift excesivo en el desplazamiento.
        
        Args:
            displacement: Array de desplazamiento
            threshold_ratio: Ratio máximo de Drifts/RMS
            
        Returns:
            Diccionario con resultado de verificación
        """
        rms = np.sqrt(np.mean(displacement**2))
        
        # Calcular drift como diferencia entre valores finales e iniciales
        if len(displacement) > 1:
            # Drift final - inicial
            drift_1 = displacement[-1] - displacement[0]
            
            # Tendencia lineal
            x = np.arange(len(displacement))
            coef = np.polyfit(x, displacement, 1)
            trend_slope = coef[0]
            
            # Drift estimado al final
            drift_trend = trend_slope * (len(displacement) - 1)
        else:
            drift_1 = 0
            drift_trend = 0
        
        drift_value = max(abs(drift_1), abs(drift_trend))
        
        # Verificar si el drift es excesivo
        exceed_factor = drift_value / (rms + 1e-10)
        
        return {
            'has_excessive_drift': exceed_factor > threshold_ratio,
            'drift_value': float(drift_value),
            'rms': float(rms),
            'drift_ratio': float(exceed_factor),
            'warning': "Desplazamiento puede tener drift" if exceed_factor > threshold_ratio else "OK",
        }
    
    def integrate_segment(
        self,
        acceleration: np.ndarray,
        time: np.ndarray,
        segment_start: float,
        segment_end: float,
        double_integrate: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Integra un segmento específico de la señal.
        
        Args:
            acceleration: Array de aceleración
            time: Array de tiempo
            segment_start: Tiempo inicial
            segment_end: Tiempo final
            double_integrate: Si integrar dos veces
            
        Returns:
            Tuple de (tiempo_segmento, señal_integrada)
        """
        # Encontrar índices del segmento
        start_idx = np.argmin(np.abs(time - segment_start))
        end_idx = np.argmin(np.abs(time - segment_end))
        
        # Extraer segmento
        acc_segment = acceleration[start_idx:end_idx + 1]
        time_segment = time[start_idx:end_idx + 1]
        
        # Integrar
        if double_integrate:
            result = self.integrate_acceleration(
                acc_segment,
                time=time_segment,
                double_integrate=True,
            )
        else:
            result = self.integrate_acceleration(
                acc_segment,
                time=time_segment,
                double_integrate=False,
            )
        
        return time_segment, result
    
    def get_velocity_response_spectrum(
        self,
        acceleration: np.ndarray,
        damping_ratio: float = 0.05,
        freq_range: tuple[float, float] = (0.1, 50.0),
        n_freqs: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcula el espectro de respuesta de velocidad (pseudo-velocidad).
        
        Útil para diseño sísmico y análisis de respuesta estructural.
        
        Args:
            acceleration: Array de aceleración
            damping_ratio: Ratio de amortiguamiento
            freq_range: Rango de frecuencias
            n_freqs: Número de frecuencias a evaluar
            
        Returns:
            Tuple de (frecuencias, pseudo-velocidad)
        """
        from scipy import signal
        
        fs = self.fs
        dt = 1.0 / fs
        
        frequencies = np.logspace(
            np.log10(freq_range[0]),
            np.log10(freq_range[1]),
            n_freqs,
        )
        
        pseudo_velocity = np.zeros(n_freqs)
        
        for i, f in enumerate(frequencies):
            omega = 2 * np.pi * f
            
            # Diseñar filtro pasa-bajos para SD a w
            # Usar la transformada de Fourier
            n = len(acceleration)
            freq_axis = np.fft.fftfreq(n, dt)
            
            acc_fft = np.fft.fft(acceleration)
            
            # SD = a(w) / (w^2) para una frecuencia
            sd = np.zeros(n, dtype=complex)
            for j in range(n):
                w_squared = (2 * np.pi * freq_axis[j]) ** 2
                if w_squared > 0.01:
                    sd[j] = acc_fft[j] / w_squared
            
            # Pseudo-velocidad = w * SD
            sv_complex = np.zeros(n, dtype=complex)
            for j in range(n):
                w = abs(2 * np.pi * freq_axis[j])
                if w > 0.01 and not np.isinf(w_squared):
                    idx = j if freq_axis[j] >= 0 else n - j
                    sv_complex[idx] = w * abs(sd[idx]) if j == idx else sv_complex[idx]
            
            # Buscar la respuesta máxima
            pseudo_velocity[i] = np.max(np.abs(sv_complex))
        
        return frequencies, pseudo_velocity


class DisplacementEstimator:
    """
    Estimador de desplazamiento por doble integración.
    
    Incluye validación y advertencias sobre limitaciones.
    """
    
    def __init__(self, fs: float, highpass_freq: float = 0.5):
        self.fs = fs
        self.integrator = SignalIntegrator(fs, highpass_freq)
    
    def estimate(
        self,
        acceleration: np.ndarray,
        time: Optional[np.ndarray] = None,
        validate: bool = True,
    ) -> dict:
        """
        Estima el desplazamiento por doble integración.
        
        Args:
            acceleration: Array de aceleración
            time: Array de tiempo (opcional)
            validate: Si validar los resultados
            
        Returns:
            Diccionario con desplazamiento y advertencias
        """
        displacement = self.integrator.integrate_acceleration(
            acceleration,
            time=time,
            double_integrate=True,
        )
        
        result = {
            'displacement': displacement,
            'status': 'estimated',
        }
        
        if validate:
            drift_check = self.integrator.check_drift(displacement)
            result['drift_check'] = drift_check
            
            if drift_check['has_excessive_drift']:
                result['status'] = 'warning'
                result['warning'] = (
                    "El desplazamiento estimado puede tener drift. "
                    "Para mayor precisión, usar láser, LVDT, GPS diferencial o fotogrametría."
                )
        
        # Estadísticas del desplazamiento
        result['statistics'] = {
            'max': float(np.max(np.abs(displacement))),
            'mean': float(np.mean(displacement)),
            'rms': float(np.sqrt(np.mean(displacement**2))),
        }
        
        return result