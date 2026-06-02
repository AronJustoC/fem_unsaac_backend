# SignalCore - Procesamiento de Señales de Vibración para Puentes

## Descripción

**SignalCore** es un módulo completo de procesamiento de señales de vibración con rutinas compatibles con VibrationData/enDAQ. Diseñado específicamente para el análisis de datos de puentes con acelerómetros de 4 columnas (Tiempo, AccX, AccY, AccZ).

## Características

### Análisis en Dominio del Tiempo
- Historial temporal (Time History)
- Cálculo de estadísticas (RMS, pico, pico-pico, crest factor)
- Detección de picos y valles
- Segmentación por eventos
- Extracción de vibración libre
- Estimación de amortiguamiento

### Análisis en Dominio de la Frecuencia
- FFT con múltiples tipos de ventanas (Hanning, Hamming, Blackman)
- Detección automática de frecuencias naturales/picos dominantes
- Identificación de modos estructurales (vertical, lateral, longitudinal)
- Detección de ruido eléctrico (50/60 Hz)
- Análisis de coherencia entre canales

### Análisis Espectral
- PSD (Power Spectral Density) - Método de Welch
- Waterfall 3D (stack de FFTs para evolución tiempo-frecuencia
- Spectrograma 2D
- Análisis de paso vehicular
- RMS por bandas de frecuencia

### Filtros Digitales
- Pasa-bajos (Butterworth, Chebyshev, Bessel, Elíptico)
- Pasa-altos
- Pasa-banda
- Rechaza-banda (Notch)
- Filtro peine (comb) para eliminar armónicos de 50/60 Hz
- Filtros FIR (media móvil, Savitzky-Golay)

### Integración
- Aceleración → Velocidad (integración simple)
- Aceleración → Desplazamiento (doble integración)
- Filtros para minimizar drift
- Detección de drift excesivo

### Análisis de Envolvente
- Transformada de Hilbert
- Detección de impactos
- Periodicidad de eventos
- Análisis de sidebands

### Cepstrum
- Power Cepstrum y Real Cepstrum
- Detección de periodicidad en el espectro
- Identificación de ecos y reflexiones
- Análisis de quefrency

## Instalación

```bash
pip install numpy scipy pandas fastapi pydantic
```

## Uso Rápido

```python
from signal_core import *

import numpy as np

# Crear datos de ejemplo
fs = 200.0  # Frecuencia de muestreo
t = np.linspace(0, 10, int(fs * 10))  # 10 segundos
acc_z = 0.5 * np.sin(2 * np.pi * 2.5 * t)  # Frecuencia fundamental

# Crear SignalData
from signal_core.signal_data import SignalData, SignalChannel, SignalMetadata

metadata = SignalMetadata(name='acc_z', sampling_rate=fs)
channel = SignalChannel(time=t, amplitude=acc_z, metadata=metadata)
signal_data = SignalData()
signal_data.add_channel('acc_z', channel)

# FFT
fft_analyzer = FrequencyDomainAnalyzer(fs)
result = fft_analyzer.compute_fft(channel.amplitude)
result = fft_analyzer.find_peaks(result)
print(f"Frecuencias naturales: {result.peak_frequencies}")

# PSD
spectral = SpectralAnalyzer(fs)
psd = spectral.compute_welch_psd(channel.amplitude)
peaks = spectral.get_spectral_peaks(psd)
print(f"Picos espectrales: {peaks}")

# Waterfall 3D
waterfall = WaterfallAnalyzer(fs)
wf_result = waterfall.compute_waterfall(channel.amplitude, segment_length=1024)
constant_modes = waterfall.identify_constant_modes(wf_result)
print(f"Picos estables: {constant_modes}")
```

## API REST

El módulo expone endpoints REST para integración con el frontend:

```
POST /api/signal/import          - Importar datos
POST /api/signal/fft            - Análisis FFT
POST /api/signal/psd            - Análisis PSD
POST /api/signal/vibrationdata-analysis - A/V/D, FFT, Aggregate FFT y PSD compatible
POST /api/signal/filter         - Aplicar filtros
POST /api/signal/integrate      - Integrar señal
POST /api/signal/envelope       - Análisis de envolvente
POST /api/signal/cepstrum       - Análisis de Cepstrum
POST /api/signal/waterfall      - Análisis Waterfall
POST /api/signal/analyze/full   - Análisis completo de puente
POST /api/signal/analyze/vehicle-pass - Análisis de paso vehicular
```

## Estructura de Datos

### SignalData
Contenedor principal para datos de múltiples canales.

### SignalChannel
Un canal individual con:
- `time`: Array de tiempo
- `amplitude`: Array de amplitudes
- `metadata`: Metadatos (unidad, tipo de señal, orientación, fs)

## Formato de Archivo Esperado

```
Archivo CSV con 4 columnas:
tiempo,acelx,acely,acelz
0.0000,0.0012,-0.0008,0.0035
0.0050,0.0015,-0.0009,0.0037
...
```

- Tiempo en segundos
- Incremento constante dt
- Unidad: g, m/s², o mm/s²
- Separador: coma (CSV)
- Encabezado: en la primera fila

## Frecuencias de Muestreo Típicas para Puentes

| Tipo de Análisis | Frecuencia de Muestreo |
|-------------------|------------------------|
| Modos bajos (<10Hz) | 100-200 Hz |
| Análisis general | 200-500 Hz |
| Impactos/transitorios | 500-1000 Hz |
| Ruido de maquinaria | >1000 Hz |

## Resultados Típicos para Puentes

### Frecuencias Naturales
- **Primer modo vertical**: 1-5 Hz (puentes grandes), 3-10 Hz (puentes pequeños)
- **Modos laterales**: 0.5-2 Hz
- **Modos torsionales**: 2-6 Hz

### Análisis de Paso Vehicular
- Excitaciones transitorias durante el paso
- Vibración libre después del paso
- Decaimiento exponencial con frecuencias naturales
- Amortiguamiento: típicamente 1-5%

## Autores

- Desarrollado para la tesis de la UNSAAC
- Compatible con convenciones VibrationData Toolbox / enDAQ

## Licencia

MIT License

## Versión

1.0.0
