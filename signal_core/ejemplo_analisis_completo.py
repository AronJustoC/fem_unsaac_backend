"""
ejemplo_analisis_completo.py
============================
Ejemplo completo de análisis de datos de puente usando SignalCore.
Basado en VibrationData Toolbox / vibrationdata_App v13.3

Este script demuestra el flujo completo de análisis para datos de puente:
1. Importación de datos
2. Análisis temporal
3. Análisis FFT
4. Análisis PSD
5. Waterfall
6. Filtros e integración
7. Envolvente y detección de impactos
8. Cepstrum
9. Generación de reporte
"""

from signal_core import (
    SignalData,
    SignalChannel,
    SignalMetadata,
    TimeDomainAnalyzer,
    FrequencyDomainAnalyzer,
    SpectralAnalyzer,
    WaterfallAnalyzer,
    SignalFilter,
    SignalIntegrator,
    EnvelopeAnalyzer,
    CepstrumAnalyzer,
    BridgeDataImporter,
    ReportGenerator,
    SignalType,
    AmplitudeUnit,
    Orientation,
)
import numpy as np


def generar_datos_puente_ejemplo(
    fs: float = 200.0,
    duration: float = 30.0,
    vehiculo_frec: float = 2.5,
    ruido_nivel: float = 0.01,
) -> SignalData:
    """
    Genera datos sintéticos de ejemplo para un puente.
    
    Simula:
    - Paso de vehículos cada 4 segundos
    - Frecuencia fundamental del puente a 2.5 Hz
    - Modo secundario a 5.0 Hz
    - Ruido gaussiano
    
    Args:
        fs: Frecuencia de muestreo (Hz)
        duration: Duración (s)
        vehiculo_frec: Frecuencia característica del vehículo
        ruido_nivel: Nivel de ruido
        
    Returns:
        SignalData con los datos simulados
    """
    n_samples = int(fs * duration)
    time = np.linspace(0, duration, n_samples)
    
    # Crear eventos de paso vehicular
    pass_interval = 4.0  # segundos entre pasos
    n_passes = int(duration / pass_interval)
    
    acc_x = np.zeros(n_samples)
    acc_y = np.zeros(n_samples)
    acc_z = np.zeros(n_samples)
    
    for i in range(n_passes):
        # Tiempo con variación aleatoria
        pass_time = pass_interval * i + np.random.uniform(-0.2, 0.2)
        pass_idx = int(pass_time * fs)
        
        if pass_idx >= n_samples or pass_idx < 0:
            continue
        
        # Duración del impacto
        max_impulse_len = n_samples - pass_idx
        if max_impulse_len < 10:
            continue
        
        impulse_len = min(int(1.5 * fs), max_impulse_len)
        t_impulse = np.linspace(0, impulse_len/fs, impulse_len)
        
        # Envolvente exponencial
        envelope = np.exp(-2 * t_impulse)
        
        # Oscilación con frecuencia del puente
        oscillation_z = np.sin(2 * np.pi * vehiculo_frec * t_impulse)
        oscillation_x = 0.2 * np.sin(2 * np.pi * vehiculo_frec * 0.8 * t_impulse)
        oscillation_y = 0.1 * np.sin(2 * np.pi * vehiculo_frec * 0.6 * t_impulse)
        
        # Impulso principal en Z (vertical)
        impulse_z = 0.4 * envelope * oscillation_z
        impulse_x = 0.08 * envelope * oscillation_x
        impulse_y = 0.04 * envelope * oscillation_y
        
        end_idx = pass_idx + impulse_len
        
        acc_z[pass_idx:end_idx] += impulse_z
        acc_x[pass_idx:end_idx] += impulse_x
        acc_y[pass_idx:end_idx] += impulse_y
    
    # Agregar segundo modo (5 Hz)
    mode_2 = 0.15 * np.sin(2 * np.pi * 5.0 * time)
    acc_z += mode_2
    
    # Agregar ruido gaussiano
    acc_x += np.random.normal(0, ruido_nivel, n_samples)
    acc_y += np.random.normal(0, ruido_nivel, n_samples)
    acc_z += np.random.normal(0, ruido_nivel, n_samples)
    
    # Crear SignalData
    signal_data = SignalData()
    
    for name, data, orientation in [
        ('acc_x', acc_x, Orientation.X),
        ('acc_y', acc_y, Orientation.Y),
        ('acc_z', acc_z, Orientation.Z),
    ]:
        metadata = SignalMetadata(
            name=name,
            unit=AmplitudeUnit.G,
            signal_type=SignalType.ACCELERATION,
            orientation=orientation,
            sampling_rate=fs,
            duration=duration,
            notes=f"Datos sintéticos de puente - fs={fs}Hz",
        )
        channel = SignalChannel(time.copy(), data, metadata)
        signal_data.add_channel(name, channel)
    
    return signal_data


def analisis_temporal(signal_data: SignalData) -> dict:
    """
    Realiza análisis en el dominio del tiempo.
    
    Returns:
        Diccionario con resultados
    """
    print("\n" + "=" * 60)
    print("ANÁLISIS EN EL DOMINIO DEL TIEMPO")
    print("=" * 60)
    
    analyzer = TimeDomainAnalyzer(signal_data.fs)
    results = {}
    
    for channel_name in ['acc_x', 'acc_y', 'acc_z']:
        channel = signal_data.get_channel(channel_name)
        if channel is None:
            continue
        
        time_result = analyzer.analyze(channel.time, channel.amplitude)
        
        results[channel_name] = {
            'rms': time_result.statistics['rms'],
            'peak': time_result.statistics['peak'],
            'peak_to_peak': time_result.statistics['peak_to_peak'],
            'crest_factor': time_result.crest_factor,
            'n_peaks': len(time_result.peaks),
            'skewness': time_result.statistics['skewness'],
            'kurtosis': time_result.statistics['kurtosis'],
        }
        
        print(f"\n{channel_name.upper()}:")
        print(f"  RMS: {time_result.statistics['rms']:.6f}")
        print(f"  Pico: {time_result.statistics['peak']:.6f}")
        print(f"  Pico-pico: {time_result.statistics['peak_to_peak']:.6f}")
        print(f"  Factor de cresta: {time_result.crest_factor:.2f}")
        print(f"  Picos detectados: {len(time_result.peaks)}")
    
    return results


def analisis_fft(signal_data: SignalData, freq_range: tuple = (0, 50)) -> dict:
    """
    Realiza análisis FFT y detecta frecuencias naturales.
    
    Returns:
        Diccionario con resultados
    """
    print("\n" + "=" * 60)
    print("ANÁLISIS EN EL DOMINIO DE LA FRECUENCIA (FFT)")
    print("=" * 60)
    
    analyzer = FrequencyDomainAnalyzer(signal_data.fs)
    results = {}
    
    for channel_name in ['acc_x', 'acc_y', 'acc_z']:
        channel = signal_data.get_channel(channel_name)
        if channel is None:
            continue
        
        fft_result = analyzer.compute_fft(channel.amplitude, window_type='hanning')
        fft_result = analyzer.find_peaks(
            fft_result,
            min_height=0.1 * np.max(fft_result.amplitude_spectrum),
            max_peaks=10,
            freq_range=freq_range,
        )
        
        results[channel_name] = {
            'peak_frequencies': fft_result.peak_frequencies.tolist() if fft_result.peak_frequencies is not None else [],
            'peak_amplitudes': fft_result.peak_amplitudes.tolist() if fft_result.peak_amplitudes is not None else [],
        }
        
        print(f"\n{channel_name.upper()} - Frecuencias naturales:")
        if fft_result.peak_frequencies is not None:
            for i, (f, a) in enumerate(zip(fft_result.peak_frequencies[:5], fft_result.peak_amplitudes[:5])):
                period = 1.0 / f if f > 0 else 0
                print(f"  Modo {i+1}: f = {f:.3f} Hz (T = {period:.3f} s), amp = {a:.6f}")
        else:
            print("  No se detectaron picos")
    
    # Análisis específico para canal Z (vertical)
    if 'acc_z' in results and results['acc_z']['peak_frequencies']:
        z_freqs = results['acc_z']['peak_frequencies']
        print(f"\n📊 FRECUENCIA FUNDAMENTAL VERTICAL: {z_freqs[0]:.3f} Hz")
        print(f"   Período: {1.0/z_freqs[0]:.3f} s")
        
        if z_freqs[0] < 2.0:
            print("   → Estructura flexible (posible puente de luz considerable)")
        elif z_freqs[0] > 8.0:
            print("   → Estructura rígida")
        else:
            print("   → Estructura con rigidez moderada")
    
    return results


def analisis_psd(signal_data: SignalData) -> dict:
    """
    Realiza análisis de Densidad Espectral de Potencia.
    
    Returns:
        Diccionario con resultados
    """
    print("\n" + "=" * 60)
    print("ANÁLISIS DE DENSIDAD ESPECTRAL DE POTENCIA (PSD)")
    print("=" * 60)
    
    analyzer = SpectralAnalyzer(signal_data.fs)
    results = {}
    
    # Definir bandas de frecuencia típicas para puentes
    band_edges = [
        (0.0, 1.0),   # Banda muy baja
        (1.0, 3.0),   # Primer modo
        (3.0, 6.0),   # Segundo modo
        (6.0, 10.0),  # Modos superiores
        (10.0, 20.0), # Alta frecuencia
    ]
    
    for channel_name in ['acc_x', 'acc_y', 'acc_z']:
        channel = signal_data.get_channel(channel_name)
        if channel is None:
            continue
        
        # PSD usando Welch
        psd_result = analyzer.compute_welch_psd(channel.amplitude)
        
        # Obtener picos principales
        spectral_peaks = analyzer.get_spectral_peaks(psd_result, n_peaks=5)
        
        # Calcular RMS por bandas
        band_rms = analyzer.compute_band_rms(psd_result, band_edges)
        
        results[channel_name] = {
            'method': psd_result.method,
            'spectral_peaks': spectral_peaks,
            'band_rms': dict(zip([f"{b[0]}-{b[1]} Hz" for b in band_edges], band_rms)),
        }
        
        print(f"\n{channel_name.upper()}:")
        print(f"  Método: {psd_result.method}")
        print(f"  Picos espectrales:")
        for peak in spectral_peaks[:3]:
            print(f"    f = {peak['frequency_hz']:.3f} Hz, Q = {peak['quality_factor']:.1f}")
        print(f"  RMS por bandas:")
        for band, rms in results[channel_name]['band_rms'].items():
            print(f"    {band}: {rms:.6f}")
    
    return results


def analisis_waterfall(signal_data: SignalData, segment_length: int = 1024) -> dict:
    """
    Realiza análisis Waterfall 3D para ver evolución tiempo-frecuencia.
    
    Returns:
        Diccionario con resultados
    """
    print("\n" + "=" * 60)
    print("ANÁLISIS WATERFALL 3D (STACK DE FFTs)")
    print("=" * 60)
    
    analyzer = WaterfallAnalyzer(signal_data.fs)
    
    channel = signal_data.get_channel('acc_z')
    if channel is None:
        return {}
    
    # Calcular waterfall
    wf_result = analyzer.compute_waterfall(
        channel.amplitude,
        segment_length=segment_length,
        overlap_ratio=0.75,
        max_freq=20.0,
    )
    
    print(f"  Segmentos calculados: {len(wf_result.times)}")
    print(f"  Rango de frecuencias: 0 - {wf_result.frequencies[-1]:.1f} Hz")
    
    # Identificar modos constantes
    constant_modes = analyzer.identify_constant_modes(wf_result, freq_tolerance=0.5)
    
    print(f"\n  Modos estructurales identificados:")
    for i, mode in enumerate(constant_modes[:5]):
        print(f"    Modo {i+1}: f = {mode['average_frequency_hz']:.3f} Hz")
        print(f"             Estabilidad: ±{mode['frequency_stability_hz']:.3f} Hz")
        print(f"             Presencia: {mode['presence_ratio']*100:.1f}%")
    
    return {
        'n_slices': len(wf_result.times),
        'constant_modes': constant_modes[:5],
    }


def aplicar_filtros_e_integracion(signal_data: SignalData) -> dict:
    """
    Aplica filtros e integración.
    
    Returns:
        Diccionario con resultados
    """
    print("\n" + "=" * 60)
    print("FILTRADO E INTEGRACIÓN")
    print("=" * 60)
    
    results = {}
    
    # Filtro pasa-banda para análisis modal
    signal_filter = SignalFilter(signal_data.fs)
    
    channel = signal_data.get_channel('acc_z')
    if channel is None:
        return {}
    
    # Aplicar bandpass (0.5-20 Hz)
    filtered = signal_filter.apply(
        channel.amplitude,
        filter_type='bandpass',
        order=4,
        cutoff_freq=(0.5, 20.0),
    )
    
    print(f"  Filtro bandpass: 0.5-20 Hz, orden 4")
    print(f"  Longitud señal filtrada: {len(filtered)}")
    
    # Integrar a velocidad
    integrator = SignalIntegrator(signal_data.fs, highpass_freq=0.5)
    velocity = integrator.integrate_acceleration(channel.amplitude, double_integrate=False)
    
    print(f"\n  Integración a velocidad:")
    print(f"    RMS de velocidad: {np.sqrt(np.mean(velocity**2)):.6f} m/s")
    
    # Integrar a desplazamiento
    displacement = integrator.integrate_acceleration(channel.amplitude, double_integrate=True)
    
    # Verificar drift
    drift_check = integrator.check_drift(displacement)
    
    print(f"\n  Integración a desplazamiento:")
    print(f"    RMS de desplazamiento: {np.sqrt(np.mean(displacement**2)):.6f} m")
    print(f"    Drift: {drift_check['drift_ratio']:.2f}x")
    
    if drift_check['has_excessive_drift']:
        print("    ⚠ ADVERTENCIA: Drift excesivo detectado")
        print("      Considerar validar con láser/LVDT/GPS")
    
    results['filtered'] = {'length': len(filtered)}
    results['velocity'] = {'rms': float(np.sqrt(np.mean(velocity**2)))}
    results['displacement'] = {
        'rms': float(np.sqrt(np.mean(displacement**2))),
        'drift_warning': drift_check['has_excessive_drift'],
    }
    
    return results


def analisis_envolvente(signal_data: SignalData) -> dict:
    """
    Analiza envolvente y detecta impactos.
    
    Returns:
        Diccionario con resultados
    """
    print("\n" + "=" * 60)
    print("ANÁLISIS DE ENVOLVENTE (HILBERT)")
    print("=" * 60)
    
    analyzer = EnvelopeAnalyzer(signal_data.fs)
    
    channel = signal_data.get_channel('acc_z')
    if channel is None:
        return {}
    
    # Calcular envolvente
    env_result = analyzer.compute_envelope_from_filtered(
        channel.amplitude,
        channel.time,
        low_freq=10.0,
        high_freq=100.0,
    )
    
    # Detectar impactos
    impact_result = analyzer.detect_impacts(env_result)
    
    print(f"  Picos detectados: {len(impact_result.impact_times)}")
    print(f"  Intervalo promedio: {impact_result.avg_interval:.3f} s")
    print(f"  Score de periodicidad: {impact_result.periodicity_score:.2f}")
    print(f"  Fuente probable: {impact_result.probable_source}")
    
    # FFT de la envolvente
    env_fft = analyzer.get_envelope_fft(env_result, max_freq=10.0)
    
    print(f"\n  FFT de envolvente - Picos principales:")
    from scipy import signal
    peaks, _ = signal.find_peaks(env_fft[1], height=0.1 * np.max(env_fft[1]))
    for i, peak_idx in enumerate(peaks[:3]):
        print(f"    Frecu: {env_fft[0][peak_idx]:.3f} Hz")
    
    return {
        'n_impacts': len(impact_result.impact_times),
        'avg_interval_s': impact_result.avg_interval,
        'periodicity_score': impact_result.periodicity_score,
        'probable_source': impact_result.probable_source,
    }


def analisis_cepstrum(signal_data: SignalData) -> dict:
    """
    Realiza análisis de Cepstrum.
    
    Returns:
        Diccionario con resultados
    """
    print("\n" + "=" * 60)
    print("ANÁLISIS DE CEPSTRUM")
    print("=" * 60)
    
    analyzer = CepstrumAnalyzer(signal_data.fs)
    
    channel = signal_data.get_channel('acc_z')
    if channel is None:
        return {}
    
    # Calcular power cepstrum
    cep_result = analyzer.compute_power_cepstrum(channel.amplitude)
    
    # Encontrar quefrencys dominantes
    cep_result = analyzer.find_dominant_quefrencies(
        cep_result,
        min_quefrency=0.01,
        max_quefrency=5.0,
    )
    
    print(f"  Quefrencys dominantes:")
    for i, (q, amp) in enumerate(zip(cep_result.dominant_quefrencies[:5], cep_result.dominant_amplitudes[:5])):
        period_hz = 1.0 / q if q > 0 else 0
        classification = analyzer._classify_quefrency(q)
        print(f"    Q = {q:.4f} s, f = {period_hz:.2f} Hz - {classification}")
    
    # Interpretar picos
    peaks = analyzer.interpret_peaks(cep_result)
    
    return {
        'dominant_quefrencies': cep_result.dominant_quefrencies[:5],
        'interpreted_peaks': [
            {'quefrency_s': p.quefrency_s, 'period_hz': p.period_hz, 'classification': p.classification}
            for p in peaks
        ],
    }


def generar_reporte_completo(signal_data: SignalData, resultados: dict) -> None:
    """
    Genera un reporte completo del análisis.
    """
    print("\n" + "=" * 60)
    print("REPORTE DE ANÁLISIS DE VIBRACIÓN DE PUENTE")
    print("=" * 60)
    
    print(f"\n📁 INFORMACIÓN DEL ARCHIVO:")
    print(f"   Frecuencia de muestreo: {signal_data.fs} Hz")
    print(f"   Duración: {signal_data.duration:.2f} s")
    print(f"   Muestras: {len(signal_data.time)}")
    print(f"   Canales: {list(signal_data.channels.keys())}")
    
    if 'fft' in resultados:
        print(f"\n📊 FRECUENCIAS NATURALES IDENTIFICADAS:")
        if 'acc_z' in resultados['fft']:
            z_freqs = resultados['fft']['acc_z']['peak_frequencies']
            if z_freqs:
                print(f"   Frecuencia fundamental: {z_freqs[0]:.3f} Hz")
                print(f"   Período: {1.0/z_freqs[0]:.3f} s")
                if len(z_freqs) > 1:
                    print(f"   Segundo modo: {z_freqs[1]:.3f} Hz")
    
    if 'envolvente' in resultados:
        print(f"\n🔔 ANÁLISIS DE IMPACTOS:")
        print(f"   Impactos detectados: {resultados['envolvente']['n_impacts']}")
        print(f"   Periodicidad: {resultados['envolvente']['periodicity_score']:.2f}")
        print(f"   Fuente: {resultados['envolvente']['probable_source']}")
    
    print("\n" + "=" * 60)
    print("FIN DEL REPORTE")
    print("=" * 60)


def main():
    """
    Función principal que ejecuta el análisis completo.
    """
    print("=" * 60)
    print("SEÑALCORE v1.0.0 - ANÁLISIS COMPLETO DE PUENTE")
    print("=" * 60)
    print("\nBasado en VibrationData Toolbox / vibrationdata_App v13.3")
    print("=" * 60)
    
    # Generar datos de ejemplo
    print("\nGenerando datos sintéticos de puente...")
    signal_data = generar_datos_puente_ejemplo(
        fs=200.0,
        duration=30.0,
        vehiculo_frec=2.5,
    )
    print(f"✓ Datos generados: {len(signal_data.time)} muestras, {signal_data.duration:.1f} s")
    
    # Ejecutar análisis
    resultados = {}
    
    # 1. Análisis temporal
    resultados['temporal'] = analisis_temporal(signal_data)
    
    # 2. Análisis FFT
    resultados['fft'] = analisis_fft(signal_data)
    
    # 3. Análisis PSD
    resultados['psd'] = analisis_psd(signal_data)
    
    # 4. Análisis Waterfall
    resultados['waterfall'] = analisis_waterfall(signal_data)
    
    # 5. Filtros e integración
    resultados['filtros'] = aplicar_filtros_e_integracion(signal_data)
    
    # 6. Análisis de envolvente
    resultados['envolvente'] = analisis_envolvente(signal_data)
    
    # 7. Análisis de Cepstrum
    resultados['cepstrum'] = analisis_cepstrum(signal_data)
    
    # Generar reporte
    generar_reporte_completo(signal_data, resultados)
    
    return resultados


if __name__ == '__main__':
    resultados = main()