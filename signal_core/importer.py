"""
importer.py - Importación de Datos de Señales de Vibración
=========================================================
Importación de archivos CSV, TXT, Excel para datos de puentes.
 Incluye validación, detección de formato, y procesamiento automático.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import io


@dataclass
class ImportConfig:
    """
    Configuración para la importación.
    
    Attributes:
        has_header: Si el archivo tiene encabezado
        time_column: Índice de la columna de tiempo (0-based)
        data_columns: Índices de las columnas de datos
        delimiter: Delimitador (coma, punto y coma, tab, etc.)
        decimal_separator: Separador decimal ('.' o ',')
        unit: Unidad de los datos
        sampling_rate: Frecuencia de muestreo (calculada o fija)
        skip_rows: Filas a saltar al inicio
    """
    has_header: bool = True
    time_column: int = 0
    data_columns: List[int] = None
    delimiter: str = ','
    decimal_separator: str = '.'
    unit: str = 'g'
    sampling_rate: Optional[float] = None
    skip_rows: int = 0
    
    def __post_init__(self):
        if self.data_columns is None:
            self.data_columns = [1, 2, 3]


@dataclass
class ImportResult:
    """
    Resultado de la importación.
    
    Attributes:
        success: Si la importación fue exitosa
        signal_data: SignalData importado
        file_name: Nombre del archivo
        errors: Lista de errores
        warnings: Lista de advertencias
        metadata: Metadatos de la importación
    """
    success: bool
    signal_data: Optional[any] = None  # Forward reference
    file_name: str = ""
    errors: List[str] = None
    warnings: List[str] = None
    metadata: dict = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> dict:
        """Convierte a diccionario serializable."""
        return {
            'success': self.success,
            'file_name': self.file_name,
            'errors': self.errors,
            'warnings': self.warnings,
            'metadata': self.metadata,
        }


class SignalImporter:
    """
    Importador genérico de señales de vibración.
    """
    
    def __init__(self):
        """Inicializa el importador."""
        self.config = ImportConfig()
    
    def import_csv(
        self,
        file_path: str,
        config: Optional[ImportConfig] = None,
    ) -> ImportResult:
        """
        Importa un archivo CSV.
        
        Args:
            file_path: Ruta del archivo
            config: Configuración opcional
            
        Returns:
            ImportResult con los datos importados
        """
        if config is None:
            config = self.config
        
        result = ImportResult(success=False, file_name=file_path)
        
        try:
            # Leer el archivo
            df = self._read_csv(file_path, config)
            
            if df is None or len(df) == 0:
                result.errors.append("No se pudieron leer datos del archivo")
                return result
            
            # Extraer columnas
            time_data = df.iloc[:, config.time_column].values
            
            if config.delimiter == ';':
                time_data = np.array([float(str(x).replace(',', '.')) for x in time_data])
            
            # Verificar que el tiempo sea numérico
            try:
                time_data = np.array([float(x) for x in time_data])
            except ValueError:
                result.errors.append("Columna de tiempo contiene valores no numéricos")
                return result
            
            # Extraer datos
            channels = {}
            for i, col_idx in enumerate(config.data_columns):
                if col_idx < len(df.columns):
                    col_data = df.iloc[:, col_idx].values
                    
                    # Convertir separador decimal si es necesario
                    if config.delimiter == ';':
                        col_data = np.array([float(str(x).replace(',', '.')) for x in col_data])
                    else:
                        col_data = np.array([float(x) for x in col_data])
                    
                    channel_names = ['acc_x', 'acc_y', 'acc_z', 'vel_x', 'vel_y', 'vel_z', 'disp_x', 'disp_y', 'disp_z']
                    channel_name = channel_names[i] if i < len(channel_names) else f'channel_{i}'
                    
                    channels[channel_name] = col_data
                else:
                    result.warnings.append(f"Índice de columna {col_idx} fuera de rango")
            
            # Verificar muestreo constante
            if len(time_data) > 1:
                dt_values = np.diff(time_data)
                if not np.allclose(dt_values, dt_values[0], rtol=0.01):
                    result.warnings.append("Muestreo irregular detectado")
                    dt_avg = np.mean(dt_values)
                else:
                    dt_avg = dt_values[0]
                
                # Calcular frecuencia de muestreo
                fs = 1.0 / dt_avg if dt_avg > 0 else 0
            else:
                fs = 0
                result.warnings.append("Datos insuficientes")
            
            # Calcular duración
            duration = time_data[-1] - time_data[0] if len(time_data) > 1 else 0
            
            # Crear SignalData
            from .signal_data import SignalData, SignalChannel, SignalMetadata, SignalType, AmplitudeUnit, Orientation
            
            signal_data = SignalData()
            
            for name, amplitude in channels.items():
                metadata = SignalMetadata(
                    name=name,
                    unit=AmplitudeUnit(config.unit),
                    signal_type=SignalType.ACCELERATION if 'acc' in name else SignalType.UNKNOWN,
                    orientation=self._get_orientation(name),
                    sampling_rate=fs,
                    duration=duration,
                )
                
                channel = SignalChannel(
                    time=time_data.copy(),
                    amplitude=amplitude,
                    metadata=metadata,
                )
                
                signal_data.add_channel(name, channel)
            
            result.success = True
            result.signal_data = signal_data
            result.metadata = {
                'n_rows': len(df),
                'n_channels': len(channels),
                'sampling_rate_hz': fs,
                'duration_s': duration,
                'dt_s': 1.0 / fs if fs > 0 else 0,
            }
            
        except FileNotFoundError:
            result.errors.append(f"Archivo no encontrado: {file_path}")
        except Exception as e:
            result.errors.append(f"Error al importar: {str(e)}")
        
        return result
    
    def import_numpy(
        self,
        time: np.ndarray,
        amplitude: np.ndarray,
        name: str = "signal",
        unit: str = "g",
    ) -> ImportResult:
        """
        Importa datos desde arrays de numpy.
        
        Args:
            time: Array de tiempo
            amplitude: Array de amplitudes
            name: Nombre de la señal
            unit: Unidad
            
        Returns:
            ImportResult con los datos importados
        """
        from .signal_data import SignalData, SignalChannel, SignalMetadata, SignalType, AmplitudeUnit
        
        result = ImportResult(success=True, file_name="numpy_array")
        
        # Calcular fs
        if len(time) > 1:
            dt = np.mean(np.diff(time))
            fs = 1.0 / dt if dt > 0 else 0
            duration = time[-1] - time[0]
        else:
            fs = 0
            duration = 0
        
        metadata = SignalMetadata(
            name=name,
            unit=AmplitudeUnit(unit),
            signal_type=SignalType.ACCELERATION,
            sampling_rate=fs,
            duration=duration,
        )
        
        channel = SignalChannel(time.copy(), amplitude.copy(), metadata)
        
        signal_data = SignalData()
        signal_data.add_channel(name, channel)
        
        result.signal_data = signal_data
        result.metadata = {
            'n_samples': len(time),
            'sampling_rate_hz': fs,
            'duration_s': duration,
        }
        
        return result
    
    def import_dataframe(
        self,
        df: pd.DataFrame,
        config: Optional[ImportConfig] = None,
    ) -> ImportResult:
        """
        Importa datos desde un DataFrame de pandas.
        
        Args:
            df: DataFrame con los datos
            config: Configuración opcional
            
        Returns:
            ImportResult
        """
        # Guardar temporalmente
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        
        # Guardar en archivo temporal
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(buffer.getvalue())
            temp_path = f.name
        
        return self.import_csv(temp_path, config)
    
    def _read_csv(
        self,
        file_path: str,
        config: ImportConfig,
    ) -> Optional[pd.DataFrame]:
        """
        Lee un archivo CSV con manejo de diferentes formatos.
        
        Args:
            file_path: Ruta del archivo
            config: Configuración
            
        Returns:
            DataFrame o None si falla
        """
        # Probar diferentes separadores
        delimiters = [config.delimiter]
        if config.delimiter == ',':
            delimiters.extend([';', '\t'])
        
        for delim in delimiters:
            try:
                # Detectar encoding
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(
                            file_path,
                            delimiter=delim,
                            skiprows=config.skip_rows,
                            header=0 if config.has_header else None,
                            encoding=encoding,
                            decimal=config.decimal_separator if delim != ';' else '.',
                        )
                        
                        # Verificar que hay columnas numéricas
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) >= 2:
                            return df
                    except Exception:
                        continue
                        
            except Exception:
                continue
        
        return None
    
    def _get_orientation(self, name: str) -> str:
        """Determina la orientación del canal."""
        name_lower = name.lower()
        
        if 'x' in name_lower:
            return 'x'
        elif 'y' in name_lower:
            return 'y'
        elif 'z' in name_lower:
            return 'z'
        elif 'resultant' in name_lower:
            return 'resultant'
        else:
            return 'unknown'
    
    def auto_detect_format(
        self,
        file_path: str,
    ) -> ImportConfig:
        """
        Detecta automáticamente el formato del archivo.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            ImportConfig con formato detectado
        """
        config = ImportConfig()
        
        try:
            # Leer primeras líneas para análisis
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_lines = [f.readline() for _ in range(10)]
            
            first_content = ''.join(first_lines)
            
            # Detectar separador
            comma_count = first_content.count(',')
            semicolon_count = first_content.count(';')
            tab_count = first_content.count('\t')
            
            if comma_count >= semicolon_count and comma_count >= tab_count:
                config.delimiter = ','
            elif semicolon_count > tab_count:
                config.delimiter = ';'
            else:
                config.delimiter = '\t'
            
            # Detectar encabezado
            first_line = first_lines[0].strip().lower()
            delimiters = config.delimiter
            
            # Analizar primera línea
            parts = first_line.split(delimiters)
            
            # Verificar si son nombres o números
            numeric_count = 0
            for part in parts[:4]:
                part = part.strip()
                try:
                    float(part.replace(',', '.'))
                    numeric_count += 1
                except ValueError:
                    continue
            
            if numeric_count <= 1:
                config.has_header = True
            else:
                config.has_header = False
            
            # Detectar filas iniciales no numéricas
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i > 10:
                        break
                    parts = line.strip().split(delimiters)
                    numeric_parts = sum(1 for p in parts[:4] if self._is_numeric(p))
                    if numeric_parts <= 1:
                        config.skip_rows = i
                    else:
                        break
            
            # Detectar columnas de datos
            config.data_columns = list(range(1, 4))  # Default: columnas 1, 2, 3
            
        except Exception:
            pass
        
        return config
    
    def _is_numeric(self, s: str) -> bool:
        """Verifica si un string es numérico."""
        s = s.strip()
        try:
            float(s.replace(',', '.'))
            return True
        except ValueError:
            return False


class BridgeDataImporter(SignalImporter):
    """
    Importador especializado para datos de puentes (4 columnas).
    
    Formato esperado:
    - Columna 0: Tiempo (s)
    - Columna 1: AccX (aceleración X)
    - Columna 2: AccY (aceleración Y)
    - Columna 3: AccZ (aceleración Z)
    """
    
    def import_bridge_data(
        self,
        file_path: str,
        unit: str = 'g',
        verify_sampling_rate: bool = True,
    ) -> ImportResult:
        """
        Importa datos específicos de puente con 4 columnas.
        
        Args:
            file_path: Ruta del archivo
            unit: Unidad de aceleración
            verify_sampling_rate: Si verificar la frecuencia de muestreo
            
        Returns:
            ImportResult con datos del puente
        """
        # Auto-detectar formato
        config = self.auto_detect_format(file_path)
        config.data_columns = [1, 2, 3]  # AccX, AccY, AccZ
        config.unit = unit
        
        result = self.import_csv(file_path, config)
        
        if result.success:
            # Verificaciones específicas para puentes
            signal_data = result.signal_data
            
            # Verificar傅立叶 de muestreo
            fs = signal_data.fs
            if verify_sampling_rate and fs > 0:
                expected_fs_ranges = [
                    (50, 60),    # 50-60 Hz típico
                    (100, 110),  # 100 Hz
                    (200, 210),  # 200 Hz
                    (500, 510),  # 500 Hz
                ]
                
                is_typical_fs = False
                for fs_min, fs_max in expected_fs_ranges:
                    if fs_min <= fs <= fs_max:
                        is_typical_fs = True
                        break
                
                if not is_typical_fs:
                    result.warnings.append(
                        f"Frecuencia de muestreo {fs:.1f} Hz es inusual. Verificar datos."
                    )
            
            # Verificar unidades consistentes
            units = set()
            for channel in signal_data.channels.values():
                units.add(channel.metadata.unit.value)
            
            if len(units) > 1:
                result.warnings.append("Advertencia: canales con diferentes unidades detectadas")
            
            # Agregar metadata específica del puente
            result.metadata['data_type'] = 'bridge_4columns'
            result.metadata['orientation'] = 'x,y,z'
        
        return result
    
    def create_sample_bridge_data(
        self,
        fs: float = 200.0,
        duration: float = 30.0,
        vehicle_freq: float = 2.5,
        noise_level: float = 0.01,
    ) -> ImportResult:
        """
        Crea datos de ejemplo para puente con paso vehicular.
        
        Args:
            fs: Frecuencia de muestreo
            duration: Duración en segundos
            vehicle_freq: Frecuencia característica del vehículo (Hz)
            noise_level: Nivel de ruido
            
        Returns:
            ImportResult con datos sintéticos
        """
        n_samples = int(fs * duration)
        time = np.linspace(0, duration, n_samples)
        
        # Generar excitación de vehículo (simulada)
        # Frecuencia de passo (tiempo entre ejes ~ 0.4s para vehículo a 60 km/h)
        pass_interval = 0.4  # segundos
        n_passes = int(duration / pass_interval)
        
        # Crear eventos de paso
        acc_x = np.zeros(n_samples)
        acc_y = np.zeros(n_samples)
        acc_z = np.zeros(n_samples)
        
        for i in range(n_passes):
            pass_time = pass_interval * i + np.random.uniform(0, 0.1)
            pass_idx = int(pass_time * fs)
            
            if pass_idx < n_samples:
                # Impulso en Z (vertical)
                t_impulse = np.arange(0, min(1.0, duration - pass_time), 1.0/fs)
                envelope = np.exp(-3 * t_impulse)  # Decaimiento exponencial
                oscillation = np.sin(2 * np.pi * vehicle_freq * t_impulse)
                impulse = 0.5 * envelope * oscillation
                
                end_idx = min(pass_idx + len(impulse), n_samples)
                acc_z[pass_idx:end_idx] += impulse[:end_idx - pass_idx]
                
                # Movimiento lateral más pequeño en X e Y
                acc_x[pass_idx:end_idx] += 0.1 * envelope * np.sin(2 * np.pi * vehicle_freq * 0.8 * t_impulse)
                acc_y[pass_idx:end_idx] += 0.05 * envelope * np.sin(2 * np.pi * vehicle_freq * 0.6 * t_impulse)
        
        # Agregar ruido gaussiano
        acc_x += np.random.normal(0, noise_level, n_samples)
        acc_y += np.random.normal(0, noise_level, n_samples)
        acc_z += np.random.normal(0, noise_level, n_samples)
        
        # Crear SignalData
        from .signal_data import SignalData, SignalChannel, SignalMetadata, SignalType, AmplitudeUnit
        
        signal_data = SignalData()
        
        for name, data, orientation in [
            ('acc_x', acc_x, Orientation.X),
            ('acc_y', acc_y, Orientation.Y),
            ('acc_z', acc_z, Orientation.Z),
        ]:
            metadata = SignalMetadata(
                name=name,
                unit=AmplitudeUnit(unit),
                signal_type=SignalType.ACCELERATION,
                orientation=orientation,
                sampling_rate=fs,
                duration=duration,
                notes=f"Sample data - fs={fs}Hz, duration={duration}s",
            )
            channel = SignalChannel(time.copy(), data, metadata)
            signal_data.add_channel(name, channel)
        
        result = ImportResult(success=True, file_name="sample_bridge_data")
        result.signal_data = signal_data
        result.metadata = {
            'sampling_rate_hz': fs,
            'duration_s': duration,
            'data_type': 'bridge_4columns_synthetic',
        }
        
        return result