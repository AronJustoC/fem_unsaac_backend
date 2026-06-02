import numpy as np
import sys
from pathlib import Path

import endaq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signal_core.vibrationdata_compat import (
    STANDARD_GRAVITY,
    analyze_vibrationdata_compat,
    compute_aggregate_fft,
    compute_avd_time_history,
    compute_full_amplitude_fft,
    compute_welch_psd,
    prepare_signal,
)


def test_full_amplitude_fft_recovers_one_g_sine_peak():
    fs = 1_000.0
    duration = 2.0
    frequency = 30.0
    time = np.arange(int(fs * duration)) / fs
    acceleration_g = np.sin(2 * np.pi * frequency * time)

    result = compute_full_amplitude_fft(
        acceleration_g,
        fs,
        window="rectangular",
        detrend="mean",
    )

    freqs = np.asarray(result["frequencies"])
    amplitudes = np.asarray(result["amplitudes"])
    peak_index = int(np.argmax(amplitudes))

    assert abs(freqs[peak_index] - frequency) <= result["frequency_resolution_hz"]
    assert np.isclose(amplitudes[peak_index], 1.0, rtol=0.01, atol=0.01)


def test_aggregate_fft_preserves_sine_amplitude_with_requested_bin_width():
    fs = 1_000.0
    duration = 4.0
    frequency = 25.0
    amplitude = 0.75
    time = np.arange(int(fs * duration)) / fs
    signal_g = amplitude * np.sin(2 * np.pi * frequency * time)

    result = compute_aggregate_fft(
        signal_g,
        fs,
        bin_width=1.0,
        window="rectangular",
        overlap=0.5,
        detrend="mean",
    )

    freqs = np.asarray(result["frequencies"])
    amplitudes = np.asarray(result["amplitudes"])
    peak_index = int(np.argmax(amplitudes))

    assert result["n_segments"] > 1
    assert result["engine"] == "endaq.calc.fft.aggregate_fft"
    assert np.isclose(result["actual_bin_width_hz"], 1.0)
    assert abs(freqs[peak_index] - frequency) <= result["actual_bin_width_hz"]
    assert np.isclose(amplitudes[peak_index], amplitude, rtol=0.02, atol=0.01)


def test_welch_psd_integral_matches_time_domain_rms():
    fs = 1_024.0
    rng = np.random.default_rng(42)
    signal_g = rng.normal(0.0, 0.25, int(fs * 12.0))
    signal_g = signal_g - np.mean(signal_g)

    result = compute_welch_psd(
        signal_g,
        fs,
        bin_width=1.0,
        window="hann",
        overlap=0.5,
        detrend="mean",
        unit_label="G",
    )

    expected_rms = float(np.sqrt(np.mean(signal_g**2)))

    assert result["unit"] == "G^2/Hz"
    assert result["engine"] == "endaq.calc.psd.welch"
    assert np.isclose(result["actual_bin_width_hz"], 1.0)
    assert np.isclose(result["rms_from_psd"], expected_rms, rtol=0.08)


def test_avd_integration_matches_sine_theory_without_highpass():
    fs = 2_000.0
    duration = 3.0
    frequency = 10.0
    time = np.arange(int(fs * duration)) / fs
    acceleration_g = np.sin(2 * np.pi * frequency * time)
    prepared = prepare_signal(acceleration_g, time=time, sampling_rate=fs, unit="g")

    result = compute_avd_time_history(
        prepared,
        highpass_hz=0.0,
        acceleration_detrend="mean",
        integration_zero="mean",
    )

    velocity_mm_s = np.asarray(result["velocity_mm_s"])
    displacement_mm = np.asarray(result["displacement_mm"])
    expected_velocity_peak = STANDARD_GRAVITY / (2 * np.pi * frequency) * 1_000.0
    expected_displacement_peak = STANDARD_GRAVITY / (2 * np.pi * frequency) ** 2 * 1_000.0

    assert np.isclose(np.max(np.abs(velocity_mm_s)), expected_velocity_peak, rtol=0.03)
    assert np.isclose(np.max(np.abs(displacement_mm)), expected_displacement_peak, rtol=0.05)


def test_analyze_vibrationdata_compat_returns_all_backend_sections():
    fs = 500.0
    duration = 2.0
    frequency = 12.5
    time = np.arange(int(fs * duration)) / fs
    acceleration_g = 0.4 * np.sin(2 * np.pi * frequency * time)

    result = analyze_vibrationdata_compat(
        acceleration=acceleration_g,
        sampling_rate=fs,
        time=time,
        unit="g",
        bin_width=2.0,
        window="hann",
        overlap=0.5,
        highpass_hz=0.0,
    )

    assert result["success"] is True
    assert result["time_histories"]["units"]["velocity"] == "mm/s"
    assert result["time_histories"]["method"]["integration"] == "endaq.calc.integrate.integrals"
    assert "acceleration" in result["fft"]
    assert "acceleration" in result["aggregate_fft"]
    assert "acceleration" in result["psd"]
    assert result["method"] == "endaq_core_vibrationdata_compatible"
    assert result["engine"]["name"] == "endaq"
    assert result["engine"]["version"] == getattr(endaq, "__version__", "unknown")
    assert result["psd"]["acceleration"]["unit"] == "G^2/Hz"
