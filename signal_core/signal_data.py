"""
signal_data.py - Core Data Structures for Signal Processing
============================================================
Define las estructuras de datos fundamentales para el procesamiento de señales.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal, Any
import numpy as np
from datetime import datetime
from enum import Enum


class AmplitudeUnit(Enum):
    """Unidades de amplitud de aceleración."""
    G = "g"                    # Aceleración en g (9.81 m/s²)
    M_S2 = "m/s²"              # Metros por segundo al cuadrado
    MM_S2 = "mm/s²"            # Milímetros por segundo al cuadrado
    CM_S2 = "cm/s²"            # Centímetros por segundo al cuadrado
    IN_S2 = "in/s²"            # Pulgadas por segundo al cuadrado
    VEL_M_S = "m/s"            # Velocidad en m/s
    VEL_MM_S = "mm/s"          # Velocidad en mm/s
    DISP_M = "m"               # Desplazamiento en metros
    DISP_MM = "mm"             # Desplazamiento en milímetros
    DISP_UM = "µm"             # Desplazamiento en micras


class SignalType(Enum):
    """Tipos de señal."""
    ACCELERATION = "acceleration"
    VELOCITY = "velocity"
    DISPLACEMENT = "displacement"
    FORCE = "force"
    PRESSURE = "pressure"
    STRAIN = "strain"
    UNKNOWN = "unknown"


class Orientation(Enum):
    """Orientación del sensor."""
    X = "x"                    # Longitudinal
    Y = "y"                    # Transversal
    Z = "z"                    # Vertical
    RESULTANT = "resultant"    # Resultante 3D
    UNKNOWN = "unknown"


@dataclass
class SignalMetadata:
    """
    Metadatos de una señal de vibración.
    
    Attributes:
        name: Nombre identificador de la señal
        unit: Unidad de medida
        signal_type: Tipo de señal (aceleración, velocidad, etc.)
        orientation: Orientación del sensor (X, Y, Z)
        sampling_rate: Frecuencia de muestreo en Hz
        duration: Duración total de la señal en segundos
        start_time: Tiempo de inicio de la adquisición
        sensor_id: Identificador del sensor
        location: Ubicación física del sensor
        notes: Notas adicionales
    """
    name: str
    unit: AmplitudeUnit = AmplitudeUnit.G
    signal_type: SignalType = SignalType.ACCELERATION
    orientation: Orientation = Orientation.UNKNOWN
    sampling_rate: float = 0.0
    duration: float = 0.0
    start_time: Optional[datetime] = None
    sensor_id: Optional[str] = None
    location: Optional[str] = None
    notes: str = ""
    # Estadísticas precalculadas
    _mean: Optional[float] = None
    _std: Optional[float] = None
    _rms: Optional[float] = None
    _peak: Optional[float] = None
    _peak_to_peak: Optional[float] = None
    
    @property
    def dt(self) -> float:
        """Incremento de tiempo entre muestras."""
        if self.sampling_rate > 0:
            return 1.0 / self.sampling_rate
        return 0.0
    
    @property
    def n_samples(self) -> int:
        """Número total de muestras."""
        if self.dt > 0 and self.duration > 0:
            return int(self.duration / self.dt)
        return 0
    
    @property
    def nyquist_freq(self) -> float:
        """Frecuencia de Nyquist."""
        return self.sampling_rate / 2.0 if self.sampling_rate > 0 else 0.0


@dataclass
class SignalChannel:
    """
    Canal individual de señal con datos temporales.
    
    Attributes:
        time: Array de tiempo
        amplitude: Array de amplitudes
        metadata: Metadatos del canal
    """
    time: np.ndarray
    amplitude: np.ndarray
    metadata: SignalMetadata
    
    def __post_init__(self):
        """Valida y sincroniza los datos."""
        if len(self.time) != len(self.amplitude):
            raise ValueError(
                f"Time ({len(self.time)}) and amplitude ({len(self.amplitude)}) "
                f"arrays must have the same length"
            )
    
    @property
    def fs(self) -> float:
        """Frecuencia de muestreo."""
        return self.metadata.sampling_rate
    
    @property
    def n(self) -> int:
        """Número de muestras."""
        return len(self.amplitude)
    
    @property
    def t_end(self) -> float:
        """Tiempo final."""
        return self.time[-1] if len(self.time) > 0 else 0.0
    
    @property
    def dt(self) -> float:
        """Incremento de tiempo."""
        if len(self.time) > 1:
            return self.time[1] - self.time[0]
        return 0.0
    
    def get_segment(self, t_start: float, t_end: float) -> 'SignalChannel':
        """
        Obtiene un segmento de la señal.
        
        Args:
            t_start: Tiempo inicial del segmento
            t_end: Tiempo final del segmento
            
        Returns:
            Nuevo SignalChannel con el segmento
        """
        mask = (self.time >= t_start) & (self.time <= t_end)
        return SignalChannel(
            time=self.time[mask],
            amplitude=self.amplitude[mask],
            metadata=self.metadata
        )
    
    def detrend(self, type: Literal['linear', 'constant'] = 'constant') -> 'SignalChannel':
        """
        Elimina la tendencia de la señal.
        
        Args:
            type: Tipo de detrend ('linear' o 'constant')
            
        Returns:
            Nuevo SignalChannel con la tendencia eliminada
        """
        if type == 'constant':
            trend = np.mean(self.amplitude)
        else:
            # Detrend lineal
            x = np.arange(len(self.amplitude))
            coef = np.polyfit(x, self.amplitude, 1)
            trend = np.polyval(coef, x)
        
        new_amplitude = self.amplitude - trend
        new_metadata = SignalMetadata(
            name=self.metadata.name,
            unit=self.metadata.unit,
            signal_type=self.metadata.signal_type,
            orientation=self.metadata.orientation,
            sampling_rate=self.metadata.sampling_rate,
            duration=self.metadata.duration,
            start_time=self.metadata.start_time,
            notes=f"{self.metadata.notes} | Detrended ({type})"
        )
        return SignalChannel(self.time.copy(), new_amplitude, new_metadata)
    
    def apply_filter(self, filter_type: str, **kwargs) -> 'SignalChannel':
        """
        Aplica un filtro a la señal.
        
        Args:
            filter_type: Tipo de filtro
            **kwargs: Parámetros del filtro
            
        Returns:
            Nuevo SignalChannel con la señal filtrada
        """
        from .filters import SignalFilter
        signal_filter = SignalFilter(self.fs)
        
        filtered = signal_filter.apply(
            self.amplitude,
            filter_type=filter_type,
            **kwargs
        )
        
        new_metadata = SignalMetadata(
            name=f"{self.metadata.name}_filtered",
            unit=self.metadata.unit,
            signal_type=self.metadata.signal_type,
            orientation=self.metadata.orientation,
            sampling_rate=self.metadata.sampling_rate,
            duration=self.metadata.duration,
            notes=f"Filter: {filter_type} | {kwargs}"
        )
        return SignalChannel(self.time.copy(), filtered, new_metadata)
    
    def to_velocity(self, highpass_freq: float = 0.5) -> 'SignalChannel':
        """
        Integra la señal para obtener velocidad.
        
        Args:
            highpass_freq: Frecuencia de corte pasa-altos para evitar drift
            
        Returns:
            Nuevo SignalChannel con velocidad
        """
        from .integration import SignalIntegrator
        integrator = SignalIntegrator(self.fs, highpass_freq=highpass_freq)
        velocity = integrator.integrate_acceleration(self.amplitude)
        
        unit_map = {
            AmplitudeUnit.G: AmplitudeUnit.VEL_M_S,
            AmplitudeUnit.M_S2: AmplitudeUnit.VEL_M_S,
            AmplitudeUnit.MM_S2: AmplitudeUnit.VEL_MM_S,
        }
        new_unit = unit_map.get(self.metadata.unit, AmplitudeUnit.VEL_M_S)
        
        new_metadata = SignalMetadata(
            name=f"{self.metadata.name}_vel",
            unit=new_unit,
            signal_type=SignalType.VELOCITY,
            orientation=self.metadata.orientation,
            sampling_rate=self.metadata.sampling_rate,
            duration=self.metadata.duration,
            notes=f"Integrated from {self.metadata.name}"
        )
        return SignalChannel(self.time.copy(), velocity, new_metadata)
    
    def to_displacement(self, highpass_freq: float = 0.5) -> 'SignalChannel':
        """
        Integra dos veces para obtener desplazamiento.
        
        Args:
            highpass_freq: Frecuencia de corte pasa-altos para evitar drift
            
        Returns:
            Nuevo SignalChannel con desplazamiento
        """
        from .integration import SignalIntegrator
        integrator = SignalIntegrator(self.fs, highpass_freq=highpass_freq)
        displacement = integrator.integrate_acceleration(
            self.amplitude, 
            double_integrate=True
        )
        
        unit_map = {
            AmplitudeUnit.G: AmplitudeUnit.DISP_M,
            AmplitudeUnit.M_S2: AmplitudeUnit.DISP_M,
            AmplitudeUnit.MM_S2: AmplitudeUnit.DISP_MM,
        }
        new_unit = unit_map.get(self.metadata.unit, AmplitudeUnit.DISP_M)
        
        new_metadata = SignalMetadata(
            name=f"{self.metadata.name}_disp",
            unit=new_unit,
            signal_type=SignalType.DISPLACEMENT,
            orientation=self.metadata.orientation,
            sampling_rate=self.metadata.sampling_rate,
            duration=self.metadata.duration,
            notes=f"Double integrated from {self.metadata.name}"
        )
        return SignalChannel(self.time.copy(), displacement, new_metadata)


@dataclass
class SignalData:
    """
    Contenedor principal para datos de señales multicanal.
    
    Diseñado para datos de puentes con 4 columnas: Tiempo, AccX, AccY, AccZ.
    
    Attributes:
        channels: Diccionario de canales por nombre
        metadata: Metadatos generales de la medición
    """
    channels: dict[str, SignalChannel] = field(default_factory=dict)
    metadata: Optional[SignalMetadata] = None
    
    def add_channel(self, name: str, channel: SignalChannel) -> None:
        """Añade un canal al contenedor."""
        self.channels[name] = channel
    
    def get_channel(self, name: str) -> Optional[SignalChannel]:
        """Obtiene un canal por nombre."""
        return self.channels.get(name)
    
    @property
    def time(self) -> np.ndarray:
        """Array de tiempo (del primer canal)."""
        if self.channels:
            first_channel = next(iter(self.channels.values()))
            return first_channel.time
        return np.array([])
    
    @property
    def fs(self) -> float:
        """Frecuencia de muestreo (del primer canal)."""
        if self.channels:
            first_channel = next(iter(self.channels.values()))
            return first_channel.fs
        return 0.0
    
    @property
    def duration(self) -> float:
        """Duración total."""
        if self.channels:
            first_channel = next(iter(self.channels.values()))
            return first_channel.metadata.duration
        return 0.0
    
    def get_resultant(self) -> SignalChannel:
        """
        Calcula la resultante 3D de las componentes X, Y, Z.
        
        Returns:
            Canal con la resultante sqrt(X² + Y² + Z²)
        """
        if not all(k in self.channels for k in ['acc_x', 'acc_y', 'acc_z']):
            raise ValueError(
                "Se requieren canales acc_x, acc_y, acc_z para calcular la resultante"
            )
        
        x = self.channels['acc_x'].amplitude
        y = self.channels['acc_y'].amplitude
        z = self.channels['acc_z'].amplitude
        
        resultant = np.sqrt(x**2 + y**2 + z**2)
        
        time = self.time
        metadata = SignalMetadata(
            name="resultant_3d",
            unit=self.channels['acc_x'].metadata.unit,
            signal_type=SignalType.ACCELERATION,
            orientation=Orientation.RESULTANT,
            sampling_rate=self.fs,
            duration=self.duration,
            notes="Resultante 3D: sqrt(X² + Y² + Z²)"
        )
        
        return SignalChannel(time, resultant, metadata)
    
    def get_statistics(self) -> dict[str, dict[str, float]]:
        """
        Calcula estadísticas para todos los canales.
        
        Returns:
            Diccionario con estadísticas por canal
        """
        stats = {}
        for name, channel in self.channels.items():
            from .statistics import SignalStatistics
            analyzer = SignalStatistics(channel.amplitude)
            stats[name] = analyzer.get_all_metrics()
        return stats
    
    def split_by_axes(self) -> dict[str, SignalChannel]:
        """
        Separa los datos en canales por eje.
        
        Returns:
            Diccionario con canales X, Y, Z
        """
        result = {}
        for axis in ['x', 'y', 'z']:
            for prefix in ['acc_', 'vel_', 'disp_']:
                key = f"{prefix}{axis}"
                if key in self.channels:
                    result[key] = self.channels[key]
        return result
    
    def validate(self) -> dict[str, Any]:
        """
        Valida la integridad de los datos.
        
        Returns:
            Diccionario con resultados de validación
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        if not self.channels:
            validation['valid'] = False
            validation['errors'].append("No hay canales en los datos")
            return validation
        
        # Verificarlongitudes
        lengths = [len(ch.time) for ch in self.channels.values()]
        if len(set(lengths)) > 1:
            validation['warnings'].append(
                f"Canales con diferentes longitudes: {lengths}"
            )
        
        # Verificar muestreo constante
        for name, ch in self.channels.items():
            if len(ch.time) > 1:
                dt_check = np.diff(ch.time)
                if not np.allclose(dt_check, dt_check[0], rtol=0.01):
                    validation['warnings'].append(
                        f"Canal '{name}' tiene muestreo irregular"
                    )
        
        # Verificar offset DC
        for name, ch in self.channels.items():
            mean = np.mean(ch.amplitude)
            if abs(mean) > 0.1 * np.std(ch.amplitude):
                validation['warnings'].append(
                    f"Canal '{name}' tiene offset DC significativo: {mean:.4f}"
                )
        
        # Info de muestreo
        validation['info']['n_channels'] = len(self.channels)
        validation['info']['n_samples'] = lengths[0] if lengths else 0
        validation['info']['sampling_rate'] = self.fs
        validation['info']['duration'] = self.duration
        
        return validation