"""
SignalCore - Signal Processing Module
=====================================
Módulo completo para procesamiento de señales de vibración en puentes y estructuras.
Basado en VibrationData Toolbox / vibrationdata_App v13.3

Funcionalidades:
- Importación de datos (CSV, columnas tiempo-acelx-acely-acelz)
- Historial temporal (Time History)
- FFT y análisis espectral
- PSD (Power Spectral Density)
- Waterfall/Spectrogram
- Filtros (pasa-bajos, pasa-altos, bandpass, Butterworth, etc.)
- Integración (aceleración -> velocidad -> desplazamiento)
- Envelope e Hilbert Transform
- Cepstrum
- Resultante 3D
- Estadísticas y métricas
- Exportación de resultados
- API REST para integración con frontend

Autor: Tesis UNSAAC
Versión: 1.0.0
"""

from .signal_data import (
    SignalData, SignalChannel, SignalMetadata,
    SignalType, AmplitudeUnit, Orientation,
)
from .time_domain import TimeDomainAnalyzer
from .frequency_domain import FrequencyDomainAnalyzer
from .spectral_analysis import SpectralAnalyzer, WaterfallAnalyzer
from .filters import SignalFilter, FilterDesign
from .integration import SignalIntegrator
from .envelope import EnvelopeAnalyzer
from .cepstrum import CepstrumAnalyzer
from .statistics import SignalStatistics
from .importer import SignalImporter, BridgeDataImporter
from .exporter import SignalExporter, ReportGenerator
from .signal_api import router as signal_api_router

__all__ = [
    # Core data structures
    'SignalData',
    'SignalChannel', 
    'SignalMetadata',
    'SignalType',
    'AmplitudeUnit',
    'Orientation',
    
    # Analyzers
    'TimeDomainAnalyzer',
    'FrequencyDomainAnalyzer',
    'SpectralAnalyzer',
    'WaterfallAnalyzer',
    'SignalFilter',
    'FilterDesign',
    'SignalIntegrator',
    'EnvelopeAnalyzer',
    'CepstrumAnalyzer',
    'SignalStatistics',
    
    # I/O
    'SignalImporter',
    'BridgeDataImporter',
    'SignalExporter',
    'ReportGenerator',
    
    # API
    'signal_api_router',
]

__version__ = "1.0.0"