"""
vibrationdata_compat.py - Cálculos VibrationData/enDAQ
======================================================

Rutinas numéricas para generar gráficas A/V/D, FFT, Aggregate FFT y PSD con
enDAQ como librería principal de cálculo.

La intención de este módulo es ser la fuente de verdad backend para cálculos
espectrales; el frontend solo debe renderizar/exportar resultados.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

import endaq
import endaq.calc.fft as endaq_fft
import endaq.calc.filters as endaq_filters
import endaq.calc.integrate as endaq_integrate
import endaq.calc.psd as endaq_psd
import numpy as np
import pandas as pd
from scipy import signal


STANDARD_GRAVITY = 9.80665
MM_PER_METER = 1_000.0

WindowName = Literal[
    "rectangular",
    "boxcar",
    "hann",
    "hanning",
    "hamming",
    "blackman",
    "flattop",
]
DetrendMode = Literal["none", "mean", "median", "linear"]


@dataclass(frozen=True)
class PreparedSignal:
    """Señal limpia con unidades de aceleración normalizadas."""

    time: np.ndarray
    acceleration_input: np.ndarray
    acceleration_g: np.ndarray
    acceleration_mps2: np.ndarray
    sampling_rate: float
    unit: str
    duration_s: float
    inferred_sampling_rate: bool = False


def _clean_1d(values: Sequence[float], *, name: str) -> np.ndarray:
    """Convierte una secuencia a float64 1D e interpola valores no finitos."""
    try:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} debe ser un arreglo numérico") from exc

    if array.size == 0:
        raise ValueError(f"{name} no puede estar vacío")

    finite = np.isfinite(array)
    if np.all(finite):
        return array.copy()
    if not np.any(finite):
        raise ValueError(f"{name} no contiene valores finitos")

    indices = np.arange(array.size)
    cleaned = array.copy()
    cleaned[~finite] = np.interp(indices[~finite], indices[finite], array[finite])
    return cleaned


def _acceleration_scale_to_mps2(unit: str) -> float:
    """Factor para convertir la unidad de entrada a m/s²."""
    normalized = unit.strip().lower().replace(" ", "").replace("²", "2")
    if normalized in {"g", "gs", "gee", "gravity", "grav"}:
        return STANDARD_GRAVITY
    if normalized in {"mg", "millig", "milligravity"}:
        return STANDARD_GRAVITY / 1_000.0
    if normalized in {"m/s2", "mps2", "m/s/s", "meter/s2", "meters/s2"}:
        return 1.0
    if normalized in {"cm/s2", "cmps2", "gal", "gals"}:
        return 0.01
    if normalized in {"mm/s2", "mmps2"}:
        return 0.001
    return 1.0


def _coerce_sampling_rate(
    time: Optional[Sequence[float]],
    n_samples: int,
    sampling_rate: Optional[float],
) -> tuple[np.ndarray, float, bool]:
    """Valida tiempo/fs y genera un eje temporal relativo en segundos."""
    fs = float(sampling_rate) if sampling_rate is not None else np.nan
    fs_is_valid = np.isfinite(fs) and fs > 0

    if time is not None:
        time_values = _clean_1d(time, name="time")
        if time_values.size != n_samples:
            raise ValueError("time y acceleration deben tener el mismo número de muestras")

        diffs = np.diff(time_values)
        if diffs.size > 0 and np.all(np.isfinite(diffs)) and np.all(diffs > 0):
            median_dt = float(np.median(diffs))
            if median_dt > 0:
                inferred_fs = 1.0 / median_dt
                if not fs_is_valid:
                    fs = inferred_fs
                    fs_is_valid = True
                    inferred = True
                else:
                    inferred = False
                return time_values - time_values[0], fs, inferred

    if not fs_is_valid:
        raise ValueError("sampling_rate debe ser positivo cuando no hay tiempo válido")

    return np.arange(n_samples, dtype=np.float64) / fs, fs, False


def prepare_signal(
    acceleration: Sequence[float],
    *,
    sampling_rate: Optional[float] = None,
    time: Optional[Sequence[float]] = None,
    unit: str = "g",
) -> PreparedSignal:
    """Limpia la señal, valida tiempo y normaliza aceleración a G y m/s²."""
    acceleration_input = _clean_1d(acceleration, name="acceleration")
    if acceleration_input.size < 2:
        raise ValueError("acceleration requiere al menos 2 muestras")

    time_array, fs, inferred_fs = _coerce_sampling_rate(time, acceleration_input.size, sampling_rate)
    scale_to_mps2 = _acceleration_scale_to_mps2(unit)
    acceleration_mps2 = acceleration_input * scale_to_mps2
    acceleration_g = acceleration_mps2 / STANDARD_GRAVITY
    duration_s = float(time_array[-1] - time_array[0]) if time_array.size > 1 else 0.0

    return PreparedSignal(
        time=time_array,
        acceleration_input=acceleration_input,
        acceleration_g=acceleration_g,
        acceleration_mps2=acceleration_mps2,
        sampling_rate=float(fs),
        unit=unit,
        duration_s=duration_s,
        inferred_sampling_rate=inferred_fs,
    )


def _window_values(window: str, n: int) -> np.ndarray:
    """Devuelve coeficientes de ventana compatibles con scipy/Matlab."""
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.float64)
    normalized = window.lower().strip()
    if normalized in {"rectangular", "boxcar", "none"}:
        return np.ones(n, dtype=np.float64)
    if normalized in {"hann", "hanning"}:
        return signal.windows.hann(n, sym=True)
    if normalized == "hamming":
        return signal.windows.hamming(n, sym=True)
    if normalized == "blackman":
        return signal.windows.blackman(n, sym=True)
    if normalized == "flattop":
        return signal.windows.flattop(n, sym=True)
    raise ValueError(f"Ventana no soportada: {window}")


def _scipy_window_name(window: str) -> str:
    normalized = window.lower().strip()
    if normalized in {"rectangular", "boxcar", "none"}:
        return "boxcar"
    if normalized == "hanning":
        return "hann"
    if normalized in {"hann", "hamming", "blackman", "flattop"}:
        return normalized
    raise ValueError(f"Ventana no soportada: {window}")


def _to_endaq_frame(time: np.ndarray, values: np.ndarray, column: str = "signal") -> pd.DataFrame:
    """Crea el DataFrame que espera `endaq.calc`: índice temporal en segundos."""
    n = min(time.size, values.size)
    if n < 2:
        raise ValueError("enDAQ requiere al menos 2 muestras")
    return pd.DataFrame(
        {column: values[:n].astype(np.float64, copy=False)},
        index=pd.Series(time[:n].astype(np.float64, copy=False), name="timestamp"),
    )


def _time_from_fs(n_samples: int, fs: float) -> np.ndarray:
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs debe ser positivo")
    return np.arange(n_samples, dtype=np.float64) / float(fs)


def _frame_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (
        frame.index.to_numpy(dtype=np.float64),
        frame.iloc[:, 0].to_numpy(dtype=np.float64),
    )


def _endaq_detrend_arg(values: np.ndarray, detrend: DetrendMode) -> tuple[np.ndarray, str | bool]:
    """Mapea nuestro modo detrend a lo que acepta scipy vía `endaq.calc.psd.welch`."""
    if detrend == "median":
        return values - float(np.median(values)), "constant"
    if detrend == "mean":
        return values, "constant"
    if detrend == "linear":
        return values, "linear"
    if detrend == "none":
        return values, False
    raise ValueError(f"Detrend no soportado: {detrend}")


def _endaq_zero_mode(zero: DetrendMode | str) -> str:
    """`endaq.calc.integrate.integrals` soporta start/mean/median."""
    if zero in {"mean", "median", "start"}:
        return str(zero)
    return "mean"


def _apply_detrend(values: np.ndarray, detrend: DetrendMode = "mean") -> np.ndarray:
    if values.size == 0 or detrend == "none":
        return values.copy()
    if detrend == "mean":
        return values - float(np.mean(values))
    if detrend == "median":
        return values - float(np.median(values))
    if detrend == "linear":
        return signal.detrend(values, type="linear")
    raise ValueError(f"Detrend no soportado: {detrend}")


def _frequency_mask(
    frequencies: np.ndarray,
    freq_range: Optional[tuple[float, float]],
) -> np.ndarray:
    if freq_range is None:
        return np.ones_like(frequencies, dtype=bool)
    f_low, f_high = float(freq_range[0]), float(freq_range[1])
    low = min(f_low, f_high)
    high = max(f_low, f_high)
    return (frequencies >= max(0.0, low)) & (frequencies <= high)


def _stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "rms": 0.0, "peak_abs": 0.0, "peak_to_peak": 0.0}
    return {
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "peak_abs": float(np.max(np.abs(values))),
        "peak_to_peak": float(np.max(values) - np.min(values)),
    }


def _spectral_peaks(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    *,
    max_peaks: int = 10,
    min_frequency_hz: float = 0.0,
) -> list[dict[str, float]]:
    if frequencies.size == 0 or amplitudes.size == 0:
        return []

    mask = frequencies >= min_frequency_hz
    local_freq = frequencies[mask]
    local_amp = amplitudes[mask]
    if local_amp.size == 0:
        return []

    max_amp = float(np.max(local_amp))
    if max_amp <= 0:
        return []

    peak_indices, properties = signal.find_peaks(local_amp, height=max_amp * 0.05)
    if peak_indices.size == 0:
        peak_indices = np.array([int(np.argmax(local_amp))])
        heights = local_amp[peak_indices]
    else:
        heights = properties.get("peak_heights", local_amp[peak_indices])

    order = np.argsort(heights)[::-1][:max_peaks]
    peaks = []
    for idx in peak_indices[order]:
        peaks.append(
            {
                "frequency_hz": float(local_freq[idx]),
                "amplitude": float(local_amp[idx]),
            }
        )
    return peaks


def compute_full_amplitude_fft(
    values: Sequence[float],
    fs: float,
    *,
    window: str = "hann",
    detrend: DetrendMode = "mean",
    freq_range: Optional[tuple[float, float]] = None,
    unit_label: str = "",
) -> dict[str, Any]:
    """
    FFT real one-sided usando `endaq.calc.fft.rfft`.

    enDAQ usa `norm="unit"` para conservar unidades físicas y hacer que un seno
    de amplitud A tenga pico cercano a A. La FFT directa de enDAQ no recibe
    ventana; la ventana se aplica en Aggregate FFT/PSD.
    """
    array = _clean_1d(values, name="values")
    clean_array = _apply_detrend(array, detrend)
    frame = _to_endaq_frame(_time_from_fs(clean_array.size, float(fs)), clean_array)
    magnitude_frame = endaq_fft.rfft(frame, output="magnitude", norm="unit", optimize=False)
    phase_frame = endaq_fft.rfft(frame, output="angle", norm="unit", optimize=False)

    frequencies, amplitudes = _frame_arrays(magnitude_frame)
    _, phases_rad = _frame_arrays(phase_frame)
    phases_deg = np.rad2deg(phases_rad)
    mask = _frequency_mask(frequencies, freq_range)
    selected_freqs = frequencies[mask]
    selected_amplitudes = amplitudes[mask]
    selected_phases = phases_deg[mask]

    return {
        "frequencies": selected_freqs.tolist(),
        "amplitudes": selected_amplitudes.tolist(),
        "phases_deg": selected_phases.tolist(),
        "peaks": _spectral_peaks(selected_freqs, selected_amplitudes, max_peaks=10, min_frequency_hz=0.0),
        "unit": unit_label,
        "engine": "endaq.calc.fft.rfft",
        "window": "none",
        "requested_window": window,
        "detrend": detrend,
        "n_samples": int(array.size),
        "frequency_resolution_hz": float(np.median(np.diff(frequencies))) if frequencies.size > 1 else 0.0,
        "overall_rms_time": _stats(_apply_detrend(array, detrend))["rms"],
    }


def _segment_size_from_bin_width(fs: float, bin_width: float, n_samples: int) -> int:
    if not np.isfinite(bin_width) or bin_width <= 0:
        bin_width = 1.0
    segment_size = int(round(float(fs) / float(bin_width)))
    segment_size = max(8, segment_size)
    return max(2, min(int(n_samples), segment_size))


def _segment_starts(n_samples: int, segment_size: int, overlap: float) -> tuple[list[int], int]:
    bounded_overlap = min(0.95, max(0.0, float(overlap)))
    noverlap = int(round(segment_size * bounded_overlap))
    noverlap = min(max(0, noverlap), max(0, segment_size - 1))
    step = max(1, segment_size - noverlap)
    starts = list(range(0, max(1, n_samples - segment_size + 1), step))
    if not starts:
        starts = [0]
    return starts, noverlap


def compute_aggregate_fft(
    values: Sequence[float],
    fs: float,
    *,
    bin_width: float = 1.0,
    window: str = "hann",
    overlap: float = 0.5,
    detrend: DetrendMode = "mean",
    freq_range: Optional[tuple[float, float]] = None,
    unit_label: str = "",
) -> dict[str, Any]:
    """
    Aggregate FFT usando `endaq.calc.fft.aggregate_fft`.

    Mantiene unidades de entrada; es el método enDAQ recomendado para picos
    estrechos/senoidales con ancho de bin controlado.
    """
    array = _clean_1d(values, name="values")
    segment_size = _segment_size_from_bin_width(float(fs), float(bin_width), array.size)
    starts, noverlap = _segment_starts(array.size, segment_size, overlap)
    prepared_values, scipy_detrend = _endaq_detrend_arg(array, detrend)
    frame = _to_endaq_frame(_time_from_fs(prepared_values.size, float(fs)), prepared_values)
    spectrum_frame = endaq_fft.aggregate_fft(
        frame,
        bin_width=float(bin_width),
        window=_scipy_window_name(window),
        noverlap=noverlap,
        detrend=scipy_detrend,
    )

    frequencies, amplitudes = _frame_arrays(spectrum_frame)
    mask = _frequency_mask(frequencies, freq_range)
    selected_freqs = frequencies[mask]
    selected_amplitudes = amplitudes[mask]
    actual_df = float(np.median(np.diff(selected_freqs))) if selected_freqs.size > 1 else 0.0

    return {
        "frequencies": selected_freqs.tolist(),
        "amplitudes": selected_amplitudes.tolist(),
        "phases_deg": [0.0] * int(selected_freqs.size),
        "phase_supported": False,
        "peaks": _spectral_peaks(selected_freqs, selected_amplitudes, max_peaks=10, min_frequency_hz=0.0),
        "unit": unit_label,
        "engine": "endaq.calc.fft.aggregate_fft",
        "window": window,
        "detrend": detrend,
        "bin_width_hz": float(bin_width),
        "actual_bin_width_hz": actual_df,
        "n_segments": int(len(starts)),
        "nperseg": int(segment_size),
        "noverlap": int(noverlap),
    }


def compute_welch_psd(
    values: Sequence[float],
    fs: float,
    *,
    bin_width: float = 1.0,
    window: str = "hann",
    overlap: float = 0.5,
    detrend: DetrendMode = "mean",
    freq_range: Optional[tuple[float, float]] = None,
    unit_label: str = "",
    zero_low_frequency_bins: int = 0,
) -> dict[str, Any]:
    """PSD Welch usando `endaq.calc.psd.welch` con densidad unidad²/Hz."""
    array = _clean_1d(values, name="values")
    segment_size = _segment_size_from_bin_width(float(fs), float(bin_width), array.size)
    _, noverlap = _segment_starts(array.size, segment_size, overlap)
    prepared_values, scipy_detrend = _endaq_detrend_arg(array, detrend)
    frame = _to_endaq_frame(_time_from_fs(prepared_values.size, float(fs)), prepared_values)
    psd_frame = endaq_psd.welch(
        frame,
        bin_width=float(bin_width),
        scaling="density",
        window=_scipy_window_name(window),
        noverlap=noverlap,
        detrend=scipy_detrend,
    )
    frequencies, psd = _frame_arrays(psd_frame)

    bins_to_zero = max(0, int(zero_low_frequency_bins))
    if bins_to_zero:
        psd = psd.copy()
        psd[: min(bins_to_zero, psd.size)] = 0.0

    mask = _frequency_mask(frequencies, freq_range)
    selected_freqs = frequencies[mask]
    selected_psd = psd[mask]

    if selected_freqs.size > 1:
        df = float(np.median(np.diff(selected_freqs)))
        rms_from_psd = float(np.sqrt(max(0.0, np.sum(selected_psd) * df)))
    else:
        df = 0.0
        rms_from_psd = 0.0

    return {
        "frequencies": selected_freqs.tolist(),
        "psd": selected_psd.tolist(),
        "peaks": _spectral_peaks(selected_freqs, selected_psd, max_peaks=10, min_frequency_hz=0.0),
        "unit": f"{unit_label}^2/Hz" if unit_label else "unit^2/Hz",
        "input_unit": unit_label,
        "method": "welch",
        "engine": "endaq.calc.psd.welch",
        "window": window,
        "detrend": detrend,
        "bin_width_hz": float(bin_width),
        "actual_bin_width_hz": df,
        "nperseg": int(segment_size),
        "noverlap": int(noverlap),
        "rms_from_psd": rms_from_psd,
        "zero_low_frequency_bins": bins_to_zero,
    }


def compute_avd_time_history(
    prepared: PreparedSignal,
    *,
    highpass_hz: float = 0.5,
    acceleration_detrend: DetrendMode = "linear",
    integration_zero: DetrendMode = "mean",
) -> dict[str, Any]:
    """
    Calcula A/V/D usando `endaq.calc.integrate.integrals`.

    La aceleración mostrada conserva la escala de la entrada ya convertida a G;
    enDAQ aplica el high-pass antes de cada integración para reducir drift.
    """
    time = prepared.time
    acceleration_mps2 = prepared.acceleration_mps2
    input_acceleration = _apply_detrend(acceleration_mps2, acceleration_detrend)
    frame = _to_endaq_frame(time, input_acceleration, column="acceleration_mps2")
    highpass_cutoff = float(highpass_hz) if np.isfinite(highpass_hz) and highpass_hz > 0 else None
    zero_mode = _endaq_zero_mode(integration_zero)
    integral_frames = endaq_integrate.integrals(
        frame,
        n=2,
        zero=zero_mode,
        highpass_cutoff=highpass_cutoff,
        tukey_percent=0.0,
    )

    if highpass_cutoff is not None:
        conditioned_frame = endaq_filters.butterworth(
            frame,
            low_cutoff=highpass_cutoff,
            high_cutoff=None,
            tukey_percent=0.0,
        )
    else:
        conditioned_frame = frame

    conditioned_acc = conditioned_frame.iloc[:, 0].to_numpy(dtype=np.float64)
    velocity = integral_frames[1].iloc[:, 0].to_numpy(dtype=np.float64)
    displacement = integral_frames[2].iloc[:, 0].to_numpy(dtype=np.float64)

    displacement_stats = _stats(displacement)
    drift = float(abs(displacement[-1] - displacement[0])) if displacement.size > 1 else 0.0
    drift_ratio = drift / max(displacement_stats["peak_abs"], 1e-15)

    velocity_mm_s = velocity * MM_PER_METER
    displacement_mm = displacement * MM_PER_METER
    conditioned_acc_g = conditioned_acc / STANDARD_GRAVITY

    return {
        "time": time.tolist(),
        "acceleration_g": prepared.acceleration_g.tolist(),
        "acceleration_conditioned_g": conditioned_acc_g.tolist(),
        "velocity_mm_s": velocity_mm_s.tolist(),
        "displacement_mm": displacement_mm.tolist(),
        "units": {
            "acceleration": "G",
            "velocity": "mm/s",
            "displacement": "mm",
        },
        "stats": {
            "acceleration_g": _stats(prepared.acceleration_g),
            "acceleration_conditioned_g": _stats(conditioned_acc_g),
            "velocity_mm_s": _stats(velocity_mm_s),
            "displacement_mm": _stats(displacement_mm),
        },
        "method": {
            "integration": "endaq.calc.integrate.integrals",
            "engine": "endaq",
            "endaq_version": getattr(endaq, "__version__", "unknown"),
            "highpass_hz": float(highpass_hz),
            "acceleration_detrend": acceleration_detrend,
            "integration_zero": zero_mode,
        },
        "drift": {
            "raw_displacement_final_minus_initial_m": drift,
            "drift_ratio": float(drift_ratio),
            "warning": bool(drift_ratio > 0.35),
        },
    }


def analyze_vibrationdata_compat(
    *,
    acceleration: Sequence[float],
    sampling_rate: Optional[float],
    time: Optional[Sequence[float]] = None,
    unit: str = "g",
    bin_width: float = 1.0,
    window: str = "hann",
    overlap: float = 0.5,
    highpass_hz: float = 0.5,
    freq_range: Optional[tuple[float, float]] = None,
    zero_low_frequency_bins: int = 0,
) -> dict[str, Any]:
    """Análisis completo para gráficas estilo VibrationData/enDAQ."""
    prepared = prepare_signal(
        acceleration,
        sampling_rate=sampling_rate,
        time=time,
        unit=unit,
    )
    avd = compute_avd_time_history(prepared, highpass_hz=highpass_hz)
    acceleration_g = np.asarray(avd["acceleration_g"], dtype=np.float64)
    velocity_mm_s = np.asarray(avd["velocity_mm_s"], dtype=np.float64)
    displacement_mm = np.asarray(avd["displacement_mm"], dtype=np.float64)

    fft_results = {
        "acceleration": compute_full_amplitude_fft(
            acceleration_g,
            prepared.sampling_rate,
            window=window,
            freq_range=freq_range,
            unit_label="G",
        ),
        "velocity": compute_full_amplitude_fft(
            velocity_mm_s,
            prepared.sampling_rate,
            window=window,
            freq_range=freq_range,
            unit_label="mm/s",
        ),
        "displacement": compute_full_amplitude_fft(
            displacement_mm,
            prepared.sampling_rate,
            window=window,
            freq_range=freq_range,
            unit_label="mm",
        ),
    }
    aggregate_fft_results = {
        "acceleration": compute_aggregate_fft(
            acceleration_g,
            prepared.sampling_rate,
            bin_width=bin_width,
            window=window,
            overlap=overlap,
            freq_range=freq_range,
            unit_label="G",
        ),
        "velocity": compute_aggregate_fft(
            velocity_mm_s,
            prepared.sampling_rate,
            bin_width=bin_width,
            window=window,
            overlap=overlap,
            freq_range=freq_range,
            unit_label="mm/s",
        ),
        "displacement": compute_aggregate_fft(
            displacement_mm,
            prepared.sampling_rate,
            bin_width=bin_width,
            window=window,
            overlap=overlap,
            freq_range=freq_range,
            unit_label="mm",
        ),
    }
    psd_results = {
        "acceleration": compute_welch_psd(
            acceleration_g,
            prepared.sampling_rate,
            bin_width=bin_width,
            window=window,
            overlap=overlap,
            freq_range=freq_range,
            unit_label="G",
            zero_low_frequency_bins=zero_low_frequency_bins,
        ),
        "velocity": compute_welch_psd(
            velocity_mm_s,
            prepared.sampling_rate,
            bin_width=bin_width,
            window=window,
            overlap=overlap,
            freq_range=freq_range,
            unit_label="mm/s",
            zero_low_frequency_bins=zero_low_frequency_bins,
        ),
        "displacement": compute_welch_psd(
            displacement_mm,
            prepared.sampling_rate,
            bin_width=bin_width,
            window=window,
            overlap=overlap,
            freq_range=freq_range,
            unit_label="mm",
            zero_low_frequency_bins=zero_low_frequency_bins,
        ),
    }

    return {
        "success": True,
        "method": "endaq_core_vibrationdata_compatible",
        "engine": {
            "name": "endaq",
            "version": getattr(endaq, "__version__", "unknown"),
            "core_functions": [
                "endaq.calc.fft.rfft",
                "endaq.calc.fft.aggregate_fft",
                "endaq.calc.psd.welch",
                "endaq.calc.integrate.integrals",
            ],
        },
        "input": {
            "n_samples": int(prepared.acceleration_input.size),
            "sampling_rate_hz": prepared.sampling_rate,
            "duration_s": prepared.duration_s,
            "unit": prepared.unit,
            "inferred_sampling_rate": prepared.inferred_sampling_rate,
        },
        "settings": {
            "bin_width_hz": float(bin_width),
            "window": window,
            "overlap": float(overlap),
            "highpass_hz": float(highpass_hz),
            "freq_range": list(freq_range) if freq_range is not None else None,
            "zero_low_frequency_bins": int(zero_low_frequency_bins),
        },
        "time_histories": avd,
        "fft": fft_results,
        "aggregate_fft": aggregate_fft_results,
        "psd": psd_results,
    }
