import numpy as np

from analysis_core.api_adapters import run_harmonic_analysis


def simple_cantilever_structure():
    return {
        "nodes": [
            {"id": 1, "coords": [0.0, 0.0, 0.0]},
            {"id": 2, "coords": [1.0, 0.0, 0.0]},
        ],
        "materials": [
            {"id": 1, "E": 2.1e11, "nu": 0.3, "rho": 7850.0},
        ],
        "sections": [
            {"id": 1, "area": 0.01, "Iz": 1e-5, "Iy": 1e-5, "J": 2e-5},
        ],
        "elements": [
            {"id": 1, "node_ids": [1, 2], "material_id": 1, "section_id": 1},
        ],
        "loads": [
            {"id": 1, "node_id": 2, "fx": 0.0, "fy": 1000.0, "fz": 0.0, "mx": 0.0, "my": 0.0, "mz": 0.0},
        ],
        "restraints": {"1": ["ux", "uy", "uz", "rx", "ry", "rz"]},
    }


def test_harmonic_analysis_returns_frequency_sweep_and_peak():
    result = run_harmonic_analysis(
        {
            "structure": simple_cantilever_structure(),
            "freq_start": 1.0,
            "freq_end": 20.0,
            "num_points": 8,
            "damping_ratio": 0.02,
        }
    )

    assert "error" not in result
    assert len(result["frequencies_sweep"]) == 8
    assert 2 in result["response_amplitudes"]
    amplitudes = np.array(result["response_amplitudes"][2])
    assert np.all(np.isfinite(amplitudes))
    assert amplitudes.max() > 0
    assert result["peak_node_id"] == 2
    assert result["peak_frequency"] is not None
    assert result["peak_amplitude"] == amplitudes.max()


def test_harmonic_analysis_rejects_zero_excitation():
    structure = simple_cantilever_structure()
    structure["loads"] = []

    result = run_harmonic_analysis(
        {
            "structure": structure,
            "freq_start": 1.0,
            "freq_end": 20.0,
            "num_points": 8,
            "damping_ratio": 0.02,
        }
    )

    assert "error" in result
    assert "carga nodal" in result["error"]


def test_harmonic_unbalanced_load_uses_selected_node_direction_mass_and_eccentricity():
    structure = simple_cantilever_structure()
    structure["loads"] = []

    result = run_harmonic_analysis(
        {
            "structure": structure,
            "freq_start": 1.0,
            "freq_end": 20.0,
            "num_points": 8,
            "damping_ratio": 0.02,
            "is_unbalanced": True,
            "unbalanced_node_id": 2,
            "unbalanced_direction": [0.0, 1.0, 0.0],
            "unbalanced_mass": 0.2,
            "unbalanced_eccentricity": 0.05,
        }
    )

    assert "error" not in result
    amplitudes = np.array(result["response_amplitudes"][2])
    assert np.all(np.isfinite(amplitudes))
    assert amplitudes.max() > 0
    assert result["peak_node_id"] == 2


def test_harmonic_unbalanced_load_rejects_zero_direction():
    structure = simple_cantilever_structure()
    structure["loads"] = []

    result = run_harmonic_analysis(
        {
            "structure": structure,
            "freq_start": 1.0,
            "freq_end": 20.0,
            "num_points": 8,
            "damping_ratio": 0.02,
            "is_unbalanced": True,
            "unbalanced_node_id": 2,
            "unbalanced_direction": [0.0, 0.0, 0.0],
            "unbalanced_mass": 0.2,
            "unbalanced_eccentricity": 0.05,
        }
    )

    assert "error" in result
    assert "dirección" in result["error"]
