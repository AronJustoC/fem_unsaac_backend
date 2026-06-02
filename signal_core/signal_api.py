"""
signal_api.py - API de Procesamiento de Señales de Vibración
=============================================================
API REST para análisis de señales de puentes y estructuras.
Expone todas las funcionalidades del SignalCore.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
import numpy as np
from io import StringIO
import tempfile
import os

# Importar módulos del signal_core
from signal_core import (
    SignalData,
    SignalChannel,
    TimeDomainAnalyzer,
    FrequencyDomainAnalyzer,
    SpectralAnalyzer,
    WaterfallAnalyzer,
    SignalFilter,
    SignalIntegrator,
    EnvelopeAnalyzer,
    CepstrumAnalyzer,
    SignalStatistics,
    BridgeDataImporter,
    SignalExporter,
    ReportGenerator,
)
from signal_core.vibrationdata_compat import analyze_vibrationdata_compat

router = APIRouter(prefix="/api/signal", tags=["Signal Processing"])


# ============ Schemas ============

class SignalImportRequest(BaseModel):
    """Request para importar datos de señal."""
    time_data: List[float]
    channels: Dict[str, List[float]]
    unit: str = "g"
    sampling_rate: Optional[float] = None
    channel_names: List[str] = ["acc_x", "acc_y", "acc_z"]
    metadata: Optional[Dict[str, Any]] = None


class FFTRequest(BaseModel):
    """Request para análisis FFT."""
    amplitude: List[float]
    sampling_rate: float
    window_type: str = "hanning"
    detrend: bool = True
    freq_range: Optional[tuple[float, float]] = None
    max_peaks: int = 10


class PSDRequest(BaseModel):
    """Request para análisis PSD."""
    amplitude: List[float]
    sampling_rate: float
    nperseg: Optional[int] = None
    noverlap: Optional[int] = None
    window: str = "hann"


class FilterRequest(BaseModel):
    """Request para aplicar filtros."""
    amplitude: List[float]
    sampling_rate: float
    filter_type: str = "bandpass"
    order: int = 4
    cutoff_low: Optional[float] = 0.5
    cutoff_high: Optional[float] = 20.0
    notch_freq: Optional[float] = 60.0
    quality_factor: float = 30.0
    n_harmonics: int = 3
    bandwidth: float = 0.5
    window_size: int = 11
    sigma: float = 3.0
    mad_threshold: float = 6.0
    alpha: float = 0.2
    polyorder: int = 2


class WaterfallRequest(BaseModel):
    """Request para análisis Waterfall."""
    amplitude: List[float]
    sampling_rate: float
    segment_length: int = 1024
    overlap_ratio: float = 0.75
    max_freq: Optional[float] = None


class EnvelopeRequest(BaseModel):
    """Request para análisis de envolvente."""
    amplitude: List[float]
    sampling_rate: float
    time: Optional[List[float]] = None
    low_freq: float = 10.0
    high_freq: float = 100.0


class IntegrationRequest(BaseModel):
    """Request para integración de señales."""
    acceleration: List[float]
    sampling_rate: float
    time: Optional[List[float]] = None
    double_integrate: bool = False
    highpass_freq: float = 0.5


class VibrationDataAnalysisRequest(BaseModel):
    """Request para cálculos estilo VibrationData/enDAQ."""
    acceleration: List[float]
    sampling_rate: Optional[float] = Field(default=None, gt=0)
    time: Optional[List[float]] = None
    unit: str = "g"
    bin_width: float = Field(default=1.0, gt=0)
    window: str = "hann"
    overlap: float = Field(default=0.5, ge=0.0, lt=0.96)
    highpass_hz: float = Field(default=0.5, ge=0.0)
    freq_range: Optional[tuple[float, float]] = None
    zero_low_frequency_bins: int = Field(default=0, ge=0, le=10)


class CepstrumRequest(BaseModel):
    """Request para análisis de Cepstrum."""
    amplitude: List[float]
    sampling_rate: float
    min_quefrency: float = 0.01
    max_quefrency: float = 10.0


class FullAnalysisRequest(BaseModel):
    """Request para análisis completo de puente."""
    time_data: List[float]
    acc_x: List[float]
    acc_y: List[float]
    acc_z: List[float]
    sampling_rate: float
    unit: str = "g"
    file_name: Optional[str] = None
    sensor_location: Optional[str] = None
    window_type: str = "hann"
    detrend: bool = False


# ============ Endpoints ============

@router.post("/import", summary="Importar datos de vibración")
async def import_signal_data(request: SignalImportRequest) -> Dict[str, Any]:
    """
    Importa datos de señales de vibración.
    
    Formato esperado:
    - time_data: Array de tiempos
    - channels: Diccionario con arrays por canal (acc_x, acc_y, acc_z)
    """
    try:
        signal_data = SignalData()
        
        time_array = np.array(request.time_data)
        fs = request.sampling_rate if request.sampling_rate else (1.0 / np.mean(np.diff(time_array)))
        
        from signal_core.signal_data import SignalMetadata, SignalType, AmplitudeUnit, Orientation
        
        for i, (channel_name, channel_data) in enumerate(request.channels.items()):
            metadata = SignalMetadata(
                name=channel_name,
                unit=AmplitudeUnit(request.unit),
                signal_type=SignalType.ACCELERATION,
                orientation=Orientation.X if 'x' in channel_name else 
                          Orientation.Y if 'y' in channel_name else
                          Orientation.Z if 'z' in channel_name else Orientation.UNKNOWN,
                sampling_rate=fs,
                duration=time_array[-1] - time_array[0] if len(time_array) > 1 else 0,
            )
            
            channel = SignalChannel(
                time=time_array.copy(),
                amplitude=np.array(channel_data),
                metadata=metadata,
            )
            signal_data.add_channel(channel_name, channel)
        
        return {
            'success': True,
            'n_channels': len(signal_data.channels),
            'n_samples': len(time_array),
            'sampling_rate': fs,
            'duration': signal_data.duration,
            'channels': list(signal_data.channels.keys()),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fft", summary="Análisis FFT")
async def compute_fft(request: FFTRequest) -> Dict[str, Any]:
    """
    Computa la FFT de la señal.
    
    Parámetros:
    - amplitude: Array de amplitudes
    - sampling_rate: Frecuencia de muestreo en Hz
    - window_type: Tipo de ventana (hanning, hamming, blackman)
    - detrend: Si remover tendencia antes de FFT
    """
    try:
        analyzer = FrequencyDomainAnalyzer(request.sampling_rate)
        
        amplitude = np.asarray(request.amplitude, dtype=np.float64)
        result = analyzer.compute_fft(
            amplitude,
            window_type=request.window_type,
            detrend=request.detrend,
        )
        
        # Encontrar picos
        min_height = 0.1 * np.max(result.amplitude_spectrum)
        result = analyzer.find_peaks(
            result,
            min_height=min_height,
            max_peaks=request.max_peaks,
            freq_range=request.freq_range,
        )
        
        return {
            'success': True,
            'frequencies': result.frequencies.tolist(),
            'amplitude_spectrum': result.amplitude_spectrum.tolist(),
            'peak_frequencies': result.peak_frequencies.tolist() if result.peak_frequencies is not None else [],
            'peak_amplitudes': result.peak_amplitudes.tolist() if result.peak_amplitudes is not None else [],
            'window_type': result.window_type,
            'nyquist_freq': request.sampling_rate / 2,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/psd", summary="Análisis de Densidad Espectral de Potencia")
async def compute_psd(request: PSDRequest) -> Dict[str, Any]:
    """
    Computa la PSD usando el método de Welch.
    
    Parámetros:
    - amplitude: Array de amplitudes
    - sampling_rate: Frecuencia de muestreo en Hz
    - nperseg: Muestras por segmento
    - noverlap: Solapamiento entre segmentos
    """
    try:
        analyzer = SpectralAnalyzer(request.sampling_rate)
        
        amplitude = np.array(request.amplitude)
        result = analyzer.compute_welch_psd(
            amplitude,
            nperseg=request.nperseg,
            noverlap=request.noverlap,
            window=request.window,
        )
        
        # Obtener picos principales
        peaks = analyzer.get_spectral_peaks(result, n_peaks=10)
        
        return {
            'success': True,
            'frequencies': result.frequencies.tolist(),
            'psd': result.psd.tolist(),
            'method': result.method,
            'spectral_peaks': peaks,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vibrationdata-analysis", summary="Análisis VibrationData/enDAQ compatible")
async def compute_vibrationdata_analysis(request: VibrationDataAnalysisRequest) -> Dict[str, Any]:
    """
    Calcula A/V/D, FFT one-sided, Aggregate FFT y PSD Welch con metadatos.

    Este endpoint está pensado para que el frontend renderice gráficas parecidas
    a VibrationData de Tom Irvine/enDAQ sin duplicar cálculos aproximados en JS.
    """
    try:
        return analyze_vibrationdata_compat(
            acceleration=request.acceleration,
            sampling_rate=request.sampling_rate,
            time=request.time,
            unit=request.unit,
            bin_width=request.bin_width,
            window=request.window,
            overlap=request.overlap,
            highpass_hz=request.highpass_hz,
            freq_range=request.freq_range,
            zero_low_frequency_bins=request.zero_low_frequency_bins,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/filter", summary="Aplicar filtros")
async def apply_filter(request: FilterRequest) -> Dict[str, Any]:
    """
    Aplica filtros a la señal.
    
    Tipos disponibles:
    - lowpass: Pasa-bajos
    - highpass: Pasa-altos
    - bandpass: Pasa-banda
    - bandstop: Rechaza-banda
    - notch: Notch para ruido de línea
    - harmonic_notch: Notch para frecuencia de línea y armónicos
    - demean/detrend: Correcciones de base
    - hampel/mad_despike/impact_guard: Anti-golpes
    - median/moving_average/exponential/savgol: Suavizados
    - anti_ski_slope: Detrend + anti-golpes + pasa alto
    """
    try:
        signal_filter = SignalFilter(request.sampling_rate)
        
        amplitude = np.array(request.amplitude)
        
        if request.filter_type in {'bandpass', 'bandstop'}:
            cutoff = (request.cutoff_low, request.cutoff_high)
        elif request.filter_type in {'highpass', 'anti_ski_slope'}:
            cutoff = request.cutoff_low if request.cutoff_low is not None else 0.5
        elif request.filter_type in {'notch', 'comb', 'harmonic_notch'}:
            cutoff = request.notch_freq or request.cutoff_high or 60.0
        elif request.filter_type in {'moving_average'}:
            cutoff = float(request.window_size)
        else:
            cutoff = request.cutoff_high or 10.0
        
        filtered = signal_filter.apply(
            amplitude,
            filter_type=request.filter_type,
            order=request.order,
            cutoff_freq=cutoff,
            Q=request.quality_factor,
            n_harmonics=request.n_harmonics,
            bandwidth=request.bandwidth,
            window_size=request.window_size,
            sigma=request.sigma,
            threshold=request.mad_threshold,
            alpha=request.alpha,
            polyorder=request.polyorder,
        )
        
        return {
            'success': True,
            'filtered_amplitude': filtered.tolist(),
            'filter_type': request.filter_type,
            'order': request.order,
            'cutoff': cutoff,
            'parameters': {
                'notch_freq': request.notch_freq,
                'quality_factor': request.quality_factor,
                'n_harmonics': request.n_harmonics,
                'bandwidth': request.bandwidth,
                'window_size': request.window_size,
                'sigma': request.sigma,
                'mad_threshold': request.mad_threshold,
                'alpha': request.alpha,
                'polyorder': request.polyorder,
            },
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/integrate", summary="Integrar señal (a→v o a→d)")
async def integrate_signal(request: IntegrationRequest) -> Dict[str, Any]:
    """
    Integra la señal de aceleración para obtener velocidad o desplazamiento.
    
    Parámetros:
    - acceleration: Array de aceleración
    - sampling_rate: Frecuencia de muestreo
    - double_integrate: Si hacer doble integración (a→v→d)
    - highpass_freq: Frecuencia de corte para evitar drift
    """
    try:
        integrator = SignalIntegrator(
            request.sampling_rate,
            highpass_freq=request.highpass_freq,
        )
        
        acceleration = np.array(request.acceleration)
        time_array = np.array(request.time) if request.time else None
        
        if request.double_integrate:
            result = integrator.integrate_acceleration(
                acceleration,
                time=time_array,
                double_integrate=True,
            )
            result_type = "displacement"
        else:
            result = integrator.integrate_acceleration(
                acceleration,
                time=time_array,
                double_integrate=False,
            )
            result_type = "velocity"
        
        # Verificar drift
        drift_check = integrator.check_drift(result)
        
        return {
            'success': True,
            'integrated_signal': result.tolist(),
            'result_type': result_type,
            'drift_warning': drift_check['has_excessive_drift'],
            'drift_ratio': drift_check['drift_ratio'],
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/envelope", summary="Análisis de envolvente e impactos")
async def compute_envelope(request: EnvelopeRequest) -> Dict[str, Any]:
    """
    Computa la envolvente de la señal para detectar impactos.
    
    Parámetros:
    - amplitude: Array de amplitudes
    - sampling_rate: Frecuencia de muestreo
    - low_freq: Frecuencia baja del bandpass
    - high_freq: Frecuencia alta del bandpass
    """
    try:
        analyzer = EnvelopeAnalyzer(request.sampling_rate)
        
        amplitude = np.array(request.amplitude)
        time_array = np.array(request.time) if request.time else None
        
        # Calcular envolvente
        envelope_result = analyzer.compute_envelope_from_filtered(
            amplitude,
            time_array,
            low_freq=request.low_freq,
            high_freq=request.high_freq,
        )
        
        # Detectar impactos
        impact_result = analyzer.detect_impacts(envelope_result)
        
        return {
            'success': True,
            'envelope_time': envelope_result.time.tolist(),
            'envelope_amplitude': envelope_result.envelope.tolist(),
            'impact_times': impact_result.impact_times.tolist(),
            'impact_amplitudes': impact_result.impact_amplitudes.tolist(),
            'avg_interval_s': impact_result.avg_interval,
            'periodicity_score': impact_result.periodicity_score,
            'probable_source': impact_result.probable_source,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cepstrum", summary="Análisis de Cepstrum")
async def compute_cepstrum(request: CepstrumRequest) -> Dict[str, Any]:
    """
    Computa el Cepstrum para detectar periodicidad en el espectro.
    
    Parámetros:
    - amplitude: Array de amplitudes
    - sampling_rate: Frecuencia de muestreo
    - min_quefrency: Quefrency mínima
    - max_quefrency: Quefrency máxima
    """
    try:
        analyzer = CepstrumAnalyzer(request.sampling_rate)
        
        amplitude = np.array(request.amplitude)
        
        # Computar cepstrum
        result = analyzer.compute_power_cepstrum(amplitude)
        
        # Encontrar quefrencys dominantes
        result = analyzer.find_dominant_quefrencies(
            result,
            min_quefrency=request.min_quefrency,
            max_quefrency=request.max_quefrency,
        )
        
        # Interpretar picos
        peaks = analyzer.interpret_peaks(result)
        
        return {
            'success': True,
            'quefrency': result.quefrency.tolist(),
            'cepstrum': result.cepstrum.tolist(),
            'dominant_quefrencies': result.dominant_quefrencies,
            'dominant_amplitudes': result.dominant_amplitudes,
            'interpreted_peaks': [
                {
                    'quefrency_s': p.quefrency_s,
                    'period_hz': p.period_hz,
                    'classification': p.classification,
                }
                for p in peaks
            ],
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/waterfall", summary="Análisis Waterfall 3D")
async def compute_waterfall(request: WaterfallRequest) -> Dict[str, Any]:
    """
    Computa el waterfall (stack de FFTs) para análisis tiempo-frecuencia.
    
    Parámetros:
    - amplitude: Array de amplitudes
    - sampling_rate: Frecuencia de muestreo
    - segment_length: Longitud de cada segmento
    - overlap_ratio: Ratio de solapamiento (0-1)
    """
    try:
        analyzer = WaterfallAnalyzer(request.sampling_rate)
        
        amplitude = np.array(request.amplitude)
        
        result = analyzer.compute_waterfall(
            amplitude,
            segment_length=request.segment_length,
            overlap_ratio=request.overlap_ratio,
            max_freq=request.max_freq,
        )
        
        # Identificar modos constantes
        constant_modes = analyzer.identify_constant_modes(result)
        
        return {
            'success': True,
            'frequencies': result.frequencies.tolist(),
            'times': result.times.tolist(),
            'n_slices': len(result.times),
            'constant_modes': constant_modes,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/full", summary="Análisis completo de puente")
async def full_bridge_analysis(request: FullAnalysisRequest) -> Dict[str, Any]:
    """
    Realiza un análisis completo de datos de puente con 4 columnas.
    
    Incluye:
    - Validación de datos
    - Historial temporal y estadísticas
    - FFT por canal
    - PSD por canal
    - Resultante 3D
    - Detección de frecuencias naturales
    """
    try:
        time_array = np.asarray(request.time_data, dtype=np.float64)
        acc_x = np.asarray(request.acc_x, dtype=np.float64)
        acc_y = np.asarray(request.acc_y, dtype=np.float64)
        acc_z = np.asarray(request.acc_z, dtype=np.float64)
        
        fs = request.sampling_rate
        n_samples = len(time_array)
        duration = time_array[-1] - time_array[0] if n_samples > 1 else 0
        
        # Crear SignalData
        from signal_core.signal_data import (
            SignalData, SignalChannel, SignalMetadata,
            SignalType, AmplitudeUnit, Orientation
        )
        
        signal_data = SignalData()
        
        unit = AmplitudeUnit(request.unit)
        
        for name, data, orientation in [
            ('acc_x', acc_x, Orientation.X),
            ('acc_y', acc_y, Orientation.Y),
            ('acc_z', acc_z, Orientation.Z),
        ]:
            metadata = SignalMetadata(
                name=name,
                unit=unit,
                signal_type=SignalType.ACCELERATION,
                orientation=orientation,
                sampling_rate=fs,
                duration=duration,
            )
            channel = SignalChannel(time_array.copy(), data, metadata)
            signal_data.add_channel(name, channel)
        
        # Calcular resultante
        try:
            resultant = signal_data.get_resultant()
        except:
            resultant = None
        
        # Analizadores
        time_analyzer = TimeDomainAnalyzer(fs)
        freq_analyzer = FrequencyDomainAnalyzer(fs)
        spectral_analyzer = SpectralAnalyzer(fs)
        
        results = {
            'success': True,
            'file_info': {
                'file_name': request.file_name or 'unknown',
                'sampling_rate_hz': fs,
                'duration_s': duration,
                'n_samples': n_samples,
                'unit': request.unit,
                'sensor_location': request.sensor_location or 'not specified',
            },
            'validation': signal_data.validate(),
            'time_domain': {},
            'frequency_domain': {},
            'spectral': {},
            'statistics': {},
        }
        
        # Análisis por canal
        for channel_name in ['acc_x', 'acc_y', 'acc_z']:
            channel = signal_data.get_channel(channel_name)
            if channel is None:
                continue
            
            amplitude = channel.amplitude
            
            # Análisis temporal
            time_result = time_analyzer.analyze(time_array, amplitude)
            results['time_domain'][channel_name] = {
                'rms': time_result.statistics['rms'],
                'peak': time_result.statistics['peak'],
                'peak_to_peak': time_result.statistics['peak_to_peak'],
                'crest_factor': time_result.crest_factor,
                'n_peaks': len(time_result.peaks),
            }
            
            # FFT
            fft_result = freq_analyzer.compute_fft(
                amplitude,
                window_type=request.window_type,
                detrend=request.detrend,
            )
            fft_result = freq_analyzer.find_peaks(
                fft_result,
                min_height=0.1 * np.max(fft_result.amplitude_spectrum),
                max_peaks=10,
            )
            
            results['frequency_domain'][channel_name] = {
                'peak_frequencies': (fft_result.peak_frequencies.tolist() 
                                    if fft_result.peak_frequencies is not None else []),
                'peak_amplitudes': (fft_result.peak_amplitudes.tolist() 
                                  if fft_result.peak_amplitudes is not None else []),
            }
            
            # PSD
            if request.window_type in ["hann", "hanning"]:
                psd_window = "hann"
            elif request.window_type == "rectangular":
                psd_window = "boxcar"
            else:
                psd_window = request.window_type
            psd_result = spectral_analyzer.compute_welch_psd(amplitude, window=psd_window)
            peaks = spectral_analyzer.get_spectral_peaks(psd_result, n_peaks=5)
            
            results['spectral'][channel_name] = {
                'psd_method': psd_result.method,
                'spectral_peaks': peaks,
            }
            
            # Estadísticas
            results['statistics'][channel_name] = time_result.statistics
        
        # Análisis de resultant si disponible
        if resultant is not None:
            results['resultant'] = {
                'n_samples': len(resultant.amplitude),
                'rms': float(np.sqrt(np.mean(resultant.amplitude**2))),
                'peak': float(np.max(np.abs(resultant.amplitude))),
            }
        
        # Identificar frecuencias naturales dominantes (canal Z - vertical)
        if 'acc_z' in results['frequency_domain']:
            z_freqs = results['frequency_domain']['acc_z']['peak_frequencies']
            if z_freqs:
                results['natural_frequencies'] = {
                    'vertical_modes': z_freqs[:5],
                    'fundamental_freq_hz': z_freqs[0] if z_freqs else None,
                }
        
        # Generar observaciones automáticas
        observations = []
        
        if results['statistics'].get('acc_z', {}).get('mean', 0) > 0.1:
            observations.append("Offset DC significativo detectado - considerar aplicar filtro pasa-altos")
        
        for channel in ['acc_x', 'acc_y', 'acc_z']:
            fft_data = results['frequency_domain'].get(channel, {})
            peaks = fft_data.get('peak_frequencies', [])
            
            for power_freq in [50, 60]:
                for peak in peaks:
                    if abs(peak - power_freq) < 2:
                        observations.append(f"Posible ruido eléctrico {power_freq} Hz detectado en {channel}")
        
        results['observations'] = observations
        
        # Recomendaciones
        recommendations = [
            "Revisar las frecuencias naturales identificadas en el informe.",
            "Comparar con modelos FEM para validación estructural.",
            "Monitorear cambios en frecuencias naturales a lo largo del tiempo.",
        ]
        results['recommendations'] = recommendations
        
        return results
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/vehicle-pass", summary="Analizar paso vehicular")
async def analyze_vehicle_pass(
    acceleration: List[float] = Form(...),
    time_data: List[float] = Form(...),
    sampling_rate: float = Form(...),
    trigger_threshold: float = Form(0.1),
) -> Dict[str, Any]:
    """
    Analiza específicamente el paso de un vehículo.
    
    Detecta:
    - Tiempo de paso
    - Pico de excitación
    - Vibración libre
    - Frecuencias naturales después del paso
    - Amortiguamiento estimado
    """
    try:
        spectral_analyzer = SpectralAnalyzer(sampling_rate)
        
        amplitude = np.array(acceleration)
        time_array = np.array(time_data)
        
        result = spectral_analyzer.analyze_vehicle_pass(
            amplitude,
            time_array,
            trigger_threshold=trigger_threshold,
            segment_duration=10.0,
        )
        
        return {
            'success': True,
            'analysis': result,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def signal_health():
    """Verificación de salud del módulo."""
    return {
        'status': 'healthy',
        'module': 'signal_core',
        'version': '1.0.0',
    }


# Función para registrar el router en la app principal
def register_signal_routes(app):
    """Registra las rutas de señales en la aplicación FastAPI."""
    app.include_router(router)
    return router
