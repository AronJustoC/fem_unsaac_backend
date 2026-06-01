"""
exporter.py - Exportación y Reportes
=====================================
Exportación de resultados, generación de reportes técnicos.
Incluye exportación a CSV, JSON, y generación de reportes completos.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal, List, Dict, Any
import numpy as np
import json
import csv
from pathlib import Path
from datetime import datetime


@dataclass
class ExportOptions:
    """
    Opciones para la exportación.
    
    Attributes:
        format: Formato de salida ('csv', 'json', 'txt')
        include_metadata: Si incluir metadatos
        precision: Precisión decimal
        compress: Si comprimir archivos
    """
    format: str = 'csv'
    include_metadata: bool = True
    precision: int = 6
    compress: bool = False


@dataclass
class AnalysisReport:
    """
    Reporte completo de análisis de señales.
    
    Attributes:
        title: Título del reporte
        date: Fecha de generación
        file_info: Información del archivo
        signal_summary: Resumen de la señal
        time_domain_results: Resultados en tiempo
        frequency_domain_results: Resultados en frecuencia
        spectral_results: Resultados espectrales
        filter_info: Información de filtros aplicados
        statistics: Estadísticas calculadas
        observations: Observaciones y conclusiones
        recommendations: Recomendaciones
    """
    title: str
    date: datetime = field(default_factory=datetime.now)
    file_info: Dict[str, Any] = field(default_factory=dict)
    signal_summary: Dict[str, Any] = field(default_factory=dict)
    time_domain_results: Dict[str, Any] = field(default_factory=dict)
    frequency_domain_results: Dict[str, Any] = field(default_factory=dict)
    spectral_results: Dict[str, Any] = field(default_factory=dict)
    filter_info: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    observations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convierte a diccionario serializable."""
        return {
            'title': self.title,
            'date': self.date.isoformat(),
            'file_info': self.file_info,
            'signal_summary': self.signal_summary,
            'time_domain_results': self.time_domain_results,
            'frequency_domain_results': self.frequency_domain_results,
            'spectral_results': self.spectral_results,
            'filter_info': self.filter_info,
            'statistics': self.statistics,
            'observations': self.observations,
            'recommendations': self.recommendations,
        }


class SignalExporter:
    """
    Exportador de señales y resultados.
    """
    
    def __init__(self, precision: int = 6):
        """
        Inicializa el exportador.
        
        Args:
            precision: Número de decimales para exportar
        """
        self.precision = precision
    
    def export_to_csv(
        self,
        data: Dict[str, np.ndarray],
        file_path: str,
        include_time: bool = True,
        time_array: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Exporta datos a CSV.
        
        Args:
            data: Diccionario con arrays de datos
            file_path: Ruta del archivo
            include_time: Si incluir columna de tiempo
            time_array: Array de tiempo (opcional)
            
        Returns:
            True si exitoso
        """
        try:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                headers = []
                if include_time:
                    headers.append('time')
                headers.extend(data.keys())
                writer.writerow(headers)
                
                # Data
                n_rows = len(next(iter(data.values())))
                
                for i in range(n_rows):
                    row = []
                    if include_time and time_array is not None:
                        row.append(f"{time_array[i]:.{self.precision}f}")
                    for key in data.keys():
                        row.append(f"{data[key][i]:.{self.precision}f}")
                    writer.writerow(row)
            
            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
    
    def export_to_json(
        self,
        data: Dict[str, Any],
        file_path: str,
        pretty: bool = True,
    ) -> bool:
        """
        Exporta datos a JSON.
        
        Args:
            data: Diccionario con datos
            file_path: Ruta del archivo
            pretty: Si usar formato legible
            
        Returns:
            True si exitoso
        """
        try:
            def numpy_converter(obj):
                """Convierte objetos numpy a tipos serializables."""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                return obj
            
            json_data = self._to_serializable(data, numpy_converter)
            
            with open(file_path, 'w') as f:
                if pretty:
                    json.dump(json_data, f, indent=2)
                else:
                    json.dump(json_data, f)
            
            return True
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False
    
    def export_fft_result(
        self,
        fft_result: 'FFTResult',
        file_path: str,
    ) -> bool:
        """
        Exporta resultados de FFT.
        
        Args:
            fft_result: FFTResult a exportar
            file_path: Ruta del archivo
            
        Returns:
            True si exitoso
        """
        data = {
            'frequency': fft_result.frequencies,
            'amplitude': fft_result.amplitude_spectrum,
        }
        
        if fft_result.phase_spectrum is not None:
            data['phase'] = fft_result.phase_spectrum
        
        if fft_result.peak_frequencies is not None:
            data['peak_frequencies'] = fft_result.peak_frequencies
            data['peak_amplitudes'] = fft_result.peak_amplitudes
        
        return self.export_to_csv(data, file_path)
    
    def export_psd_result(
        self,
        psd_result: 'PSDResult',
        file_path: str,
    ) -> bool:
        """
        Exporta resultados de PSD.
        
        Args:
            psd_result: PSDResult a exportar
            file_path: Ruta del archivo
            
        Returns:
            True si exitoso
        """
        data = {
            'frequency': psd_result.frequencies,
            'psd': psd_result.psd,
        }
        
        return self.export_to_csv(data, file_path)
    
    def export_waterfall(
        self,
        waterfall_result: 'WaterfallResult',
        base_path: str,
    ) -> List[str]:
        """
        Exporta resultados de waterfall.
        
        Args:
            waterfall_result: WaterfallResult
            base_path: Ruta base para los archivos
            
        Returns:
            Lista de archivos exportados
        """
        exported = []
        
        # Exportar matriz 3D como CSV
        matrix_path = f"{base_path}_matrix.csv"
        n_freqs = waterfall_result.amplitude_matrix.shape[1]
        
        with open(matrix_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header: time, freq1, freq2, ...
            header = ['time']
            header.extend([f'f{i}' for i in range(n_freqs)])
            writer.writerow(header)
            
            # Data rows
            for t_idx in range(len(waterfall_result.times)):
                row = [f"{waterfall_result.times[t_idx]:.{self.precision}f}"]
                row.extend([f"{amp:.{self.precision}e}" 
                           for amp in waterfall_result.amplitude_matrix[t_idx, :]])
                writer.writerow(row)
        
        exported.append(matrix_path)
        
        # Exportar metadata
        meta_path = f"{base_path}_metadata.json"
        metadata = {
            'frequencies': waterfall_result.frequencies.tolist(),
            'n_slices': len(waterfall_result.times),
            'n_frequencies': n_freqs,
            'times': waterfall_result.times.tolist(),
        }
        self.export_to_json(metadata, meta_path)
        exported.append(meta_path)
        
        return exported
    
    def _to_serializable(self, obj: Any, converter: callable) -> Any:
        """Convierte objetos a tipos serializables."""
        if isinstance(obj, dict):
            return {k: self._to_serializable(v, converter) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_serializable(item, converter) for item in obj]
        elif isinstance(obj, (np.ndarray, np.integer, np.floating, np.bool_)):
            return converter(obj)
        else:
            return obj


class ReportGenerator:
    """
    Generador de reportes técnicos completos.
    
    Genera reportes en formato estructurado para análisis
    de señales de puentes y estructuras.
    """
    
    def __init__(self):
        """Inicializa el generador de reportes."""
        pass
    
    def generate_bridge_analysis_report(
        self,
        signal_data: 'SignalData',
        time_results: Optional[dict] = None,
        fft_results: Optional[Dict[str, 'FFTResult']] = None,
        psd_results: Optional[Dict[str, 'PSDResult']] = None,
        filter_info: Optional[dict] = None,
    ) -> AnalysisReport:
        """
        Genera reporte completo para análisis de puente.
        
        Args:
            signal_data: SignalData con los datos
            time_results: Resultados del análisis temporal
            fft_results: Resultados de FFT por canal
            psd_results: Resultados de PSD por canal
            filter_info: Información de filtros aplicados
            
        Returns:
            AnalysisReport completo
        """
        report = AnalysisReport(
            title="Reporte de Análisis de Vibración de Puente",
        )
        
        # Información del archivo
        report.file_info = {
            'data_type': 'bridge_4columns',
            'sampling_rate_hz': signal_data.fs,
            'duration_s': signal_data.duration,
            'n_samples': len(signal_data.time),
            'channels': list(signal_data.channels.keys()),
            'orientation': 'X, Y, Z (longitudinal, transversal, vertical)',
        }
        
        # Resumen de la señal
        report.signal_summary = {
            'fs': signal_data.fs,
            'dt': 1.0 / signal_data.fs if signal_data.fs > 0 else 0,
            'duration': signal_data.duration,
            'nyquist_freq': signal_data.fs / 2 if signal_data.fs > 0 else 0,
        }
        
        # Resultados temporales
        if time_results is not None:
            report.time_domain_results = time_results
        
        # Resultados de frecuencia
        if fft_results is not None:
            report.frequency_domain_results = {}
            for channel_name, fft_result in fft_results.items():
                report.frequency_domain_results[channel_name] = {
                    'window_type': fft_result.window_type,
                    'peak_frequencies': (fft_result.peak_frequencies.tolist() 
                                       if fft_result.peak_frequencies is not None else []),
                    'peak_amplitudes': (fft_result.peak_amplitudes.tolist() 
                                       if fft_result.peak_amplitudes is not None else []),
                }
        
        # Resultados espectrales
        if psd_results is not None:
            report.spectral_results = {}
            for channel_name, psd_result in psd_results.items():
                report.spectral_results[channel_name] = {
                    'method': psd_result.method,
                    'dominant_bands': [],
                }
        
        # Información de filtros
        if filter_info is not None:
            report.filter_info = filter_info
        
        # Estadísticas
        stats = signal_data.get_statistics()
        report.statistics = stats
        
        # Observaciones
        report.observations = self._generate_observations(
            signal_data, fft_results, psd_results
        )
        
        # Recomendaciones
        report.recommendations = self._generate_recommendations(
            signal_data, fft_results, psd_results
        )
        
        return report
    
    def _generate_observations(
        self,
        signal_data: 'SignalData',
        fft_results: Optional[dict],
        psd_results: Optional[dict],
    ) -> List[str]:
        """
        Genera observaciones basadas en los resultados.
        
        Args:
            signal_data: Datos de la señal
            fft_results: Resultados de FFT
            psd_results: Resultados de PSD
            
        Returns:
            Lista de observaciones
        """
        observations = []
        
        # Verificar offset DC
        for name, channel in signal_data.channels.items():
            mean = np.mean(channel.amplitude)
            if abs(mean) > 0.1:
                observations.append(
                    f"Canal {name}: Offset DC significativo detectado (mean={mean:.4f}). "
                    "Considerar aplicar detrend o filtro pasa-altos."
                )
        
        # Verificar frecuencias naturales
        if fft_results is not None:
            for channel_name, fft_result in fft_results.items():
                if fft_result.peak_frequencies is not None:
                    n_modes = len(fft_result.peak_frequencies)
                    if n_modes > 0:
                        observations.append(
                            f"Canal {channel_name}: Se identificaron {n_modes} picos "
                            f"dominantes. Frecuencia fundamental: {fft_result.peak_frequencies[0]:.2f} Hz"
                        )
        
        # Verificar ruido eléctrico
        if fft_results is not None:
            for channel_name, fft_result in fft_results.items():
                freqs = fft_result.frequencies
                amps = fft_result.amplitude_spectrum
                
                # Buscar picos cerca de 50/60 Hz
                for power_freq in [50, 60]:
                    mask = np.abs(freqs - power_freq) < 2
                    if np.any(mask):
                        peak_amp = np.max(amps[mask])
                        observations.append(
                            f"Canal {channel_name}: Posible ruido eléctrico "
                            f"detectado cerca de {power_freq} Hz (amp={peak_amp:.4f})"
                        )
        
        return observations
    
    def _generate_recommendations(
        self,
        signal_data: 'SignalData',
        fft_results: Optional[dict],
        psd_results: Optional[dict],
    ) -> List[str]:
        """
        Genera recomendaciones basadas en los resultados.
        
        Args:
            signal_data: Datos de la señal
            fft_results: Resultados de FFT
            psd_results: Resultados de PSD
            
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Recomendaciones generales
        recommendations.append(
            "Mantener registro histórico de mediciones para comparar "
            "evolución de frecuencias naturales del puente."
        )
        
        # Recomendaciones específicas según modos
        if fft_results is not None and 'acc_z' in fft_results:
            z_result = fft_results['acc_z']
            if z_result.peak_frequencies is not None:
                fundamental_freq = z_result.peak_frequencies[0]
                if fundamental_freq < 2.0:
                    recommendations.append(
                        "Frecuencia fundamental baja (< 2 Hz) indica estructura "
                        "flexible. Verificar capacidad de amortiguamiento."
                    )
                elif fundamental_freq > 8.0:
                    recommendations.append(
                        "Frecuencia fundamental alta (> 8 Hz) indica estructura "
                        "rígida. Verificar que los sensores captan adecuadamente."
                    )
        
        # Recomendaciones de análisis adicional
        if psd_results is not None:
            recommendations.append(
                "Considerar realizar análisis de Waterfall para observar "
                "evolución de modos durante eventos de carga."
            )
        
        # Recomendaciones de mantenimiento
        recommendations.append(
            "Instalar sensores permanentes si se detectan cambios "
            "significativos en las frecuencias naturales."
        )
        
        return recommendations
    
    def export_report_to_json(
        self,
        report: AnalysisReport,
        file_path: str,
    ) -> bool:
        """
        Exporta el reporte a JSON.
        
        Args:
            report: AnalysisReport a exportar
            file_path: Ruta del archivo
            
        Returns:
            True si exitoso
        """
        exporter = SignalExporter()
        return exporter.export_to_json(report.to_dict(), file_path)
    
    def export_report_to_text(
        self,
        report: AnalysisReport,
        file_path: str,
    ) -> bool:
        """
        Exporta el reporte a texto plano formateado.
        
        Args:
            report: AnalysisReport a exportar
            file_path: Ruta del archivo
            
        Returns:
            True si exitoso
        """
        try:
            with open(file_path, 'w') as f:
                # Título
                f.write("=" * 70 + "\n")
                f.write(f"{report.title.upper()}\n")
                f.write("=" * 70 + "\n")
                f.write(f"Fecha: {report.date.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n")
                
                # Información del archivo
                f.write("-" * 50 + "\n")
                f.write("INFORMACIÓN DEL ARCHIVO\n")
                f.write("-" * 50 + "\n")
                for key, value in report.file_info.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
                
                # Resumen de la señal
                f.write("-" * 50 + "\n")
                f.write("RESUMEN DE LA SEÑAL\n")
                f.write("-" * 50 + "\n")
                for key, value in report.signal_summary.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
                
                # Resultados en frecuencia
                f.write("-" * 50 + "\n")
                f.write("FRECUENCIAS NATURALES IDENTIFICADAS\n")
                f.write("-" * 50 + "\n")
                for channel, data in report.frequency_domain_results.items():
                    f.write(f"\n  Canal: {channel}\n")
                    if 'peak_frequencies' in data:
                        for i, freq in enumerate(data['peak_frequencies'][:5]):
                            f.write(f"    Modo {i+1}: {freq:.3f} Hz\n")
                f.write("\n")
                
                # Observaciones
                f.write("-" * 50 + "\n")
                f.write("OBSERVACIONES\n")
                f.write("-" * 50 + "\n")
                for i, obs in enumerate(report.observations, 1):
                    f.write(f"  {i}. {obs}\n")
                f.write("\n")
                
                # Recomendaciones
                f.write("-" * 50 + "\n")
                f.write("RECOMENDACIONES\n")
                f.write("-" * 50 + "\n")
                for i, rec in enumerate(report.recommendations, 1):
                    f.write(f"  {i}. {rec}\n")
                f.write("\n")
                
                # Estadísticas
                f.write("-" * 50 + "\n")
                f.write("ESTADÍSTICAS POR CANAL\n")
                f.write("-" * 50 + "\n")
                for channel, stats in report.statistics.items():
                    f.write(f"\n  Canal: {channel}\n")
                    for metric, value in stats.items():
                        if isinstance(value, float):
                            f.write(f"    {metric}: {value:.6f}\n")
                        else:
                            f.write(f"    {metric}: {value}\n")
                f.write("\n")
                
                f.write("=" * 70 + "\n")
                f.write("FIN DEL REPORTE\n")
                f.write("=" * 70 + "\n")
            
            return True
        except Exception as e:
            print(f"Error exporting report: {e}")
            return False
    
    def generate_summary_table(
        self,
        report: AnalysisReport,
    ) -> str:
        """
        Genera una tabla resumen de los resultados.
        
        Args:
            report: AnalysisReport
            
        Returns:
            String con la tabla formateada
        """
        lines = []
        
        lines.append("\n" + "=" * 70)
        lines.append("TABLA RESUMEN DE RESULTADOS")
        lines.append("=" * 70)
        lines.append("")
        
        # Header
        lines.append(f"{'Canal':<15} {'Frecuencia (Hz)':<20} {'Amplitud':<15}")
        lines.append("-" * 50)
        
        # Datos
        if report.frequency_domain_results:
            for channel, data in report.frequency_domain_results.items():
                freqs = data.get('peak_frequencies', [])
                amps = data.get('peak_amplitudes', [])
                
                for i, (f, a) in enumerate(zip(freqs, amps)):
                    channel_str = channel if i == 0 else ""
                    lines.append(f"{channel_str:<15} {f:<20.3f} {a:<15.4f}")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)


# Alias para compatibilidad
def export_signal_data(data, filepath, format='csv', precision=6):
    """Función de conveniencia para exportar datos."""
    exporter = SignalExporter(precision)
    
    if format == 'csv':
        return exporter.export_to_csv(data, filepath)
    elif format == 'json':
        return exporter.export_to_json(data, filepath)
    else:
        raise ValueError(f"Format {format} not supported")