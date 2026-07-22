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
    assert 2 in result["node_response_series"]
    node_series = result["node_response_series"][2]
    assert set(["displacement_m", "velocity_m_s", "acceleration_m_s2", "stress_pa"]).issubset(node_series.keys())
    assert len(node_series["velocity_m_s"]) == 8
    assert np.asarray(node_series["acceleration_m_s2"]).max() > 0
    assert np.asarray(node_series["stress_pa"]).max() > 0
    assert result["node_peak_summary"][2]["velocity_m_s"] > 0
    assert 2 in result["node_displacement_components"]
    components = result["node_displacement_components"][2]
    assert set(["ux_real_m", "uy_real_m", "uz_real_m", "ux_imag_m", "uy_imag_m", "uz_imag_m"]).issubset(components.keys())
    assert len(components["uy_real_m"]) == 8


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


def test_direct_frequency_response_matches_closed_form_sdof_reference():
    """Benchmark tipo SOL 108/ANSYS full method para un GDL.

    El resultado esperado se calcula de forma independiente con la impedancia
    dinámica escalar Z(ω)=k-ω²m+iωc. Es el mismo principio matricial usado por
    solucionadores comerciales de respuesta directa en frecuencia.
    """
    from analysis_core.analisis_modal_3d.analysis.frequency_response import direct_frequency_response

    mass = 3.5
    stiffness = 1200.0
    damping = 8.0
    force = 25.0
    frequencies = np.array([0.5, 1.5, 2.75, 4.0])

    response = direct_frequency_response(
        np.array([[stiffness]]),
        np.array([[mass]]),
        np.array([[damping]]),
        np.array([force]),
        frequencies,
    )[:, 0]

    omega = 2.0 * np.pi * frequencies
    expected = force / ((stiffness - omega**2 * mass) + 1j * omega * damping)

    np.testing.assert_allclose(response.real, expected.real, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(response.imag, expected.imag, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.abs(response), np.abs(expected), rtol=1e-12, atol=1e-12)


def test_harmonic_node_kinematics_follow_frequency_domain_identities():
    """Valida que los reportes nodales sigan v=ωu y a=ω²u.

    Estas identidades son el criterio de comparación más estable entre nuestro
    resultado y cualquier software matricial de respuesta armónica en régimen estacionario.
    """
    result = run_harmonic_analysis(
        {
            "structure": simple_cantilever_structure(),
            "freq_start": 2.0,
            "freq_end": 12.0,
            "num_points": 5,
            "damping_ratio": 0.02,
        }
    )

    assert "error" not in result
    frequencies = np.array(result["frequencies_sweep"])
    omega = 2.0 * np.pi * frequencies
    node_series = result["node_response_series"][2]
    displacement = np.array(node_series["displacement_m"])
    velocity = np.array(node_series["velocity_m_s"])
    acceleration = np.array(node_series["acceleration_m_s2"])

    np.testing.assert_allclose(velocity, omega * displacement, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(acceleration, omega**2 * displacement, rtol=1e-12, atol=1e-12)

    peak_summary = result["node_peak_summary"][2]
    peak_index = int(np.argmax(displacement))
    assert peak_summary["frequency_hz"] == frequencies[peak_index]
    assert peak_summary["displacement_m"] == displacement[peak_index]
    assert peak_summary["velocity_m_s"] == velocity[peak_index]
    assert peak_summary["acceleration_m_s2"] == acceleration[peak_index]


def test_direct_frequency_response_unbalanced_force_scales_with_omega_squared():
    from analysis_core.analisis_modal_3d.analysis.frequency_response import direct_frequency_response

    mass = 2.0
    stiffness = 900.0
    damping = 5.0
    mass_eccentricity = 0.03
    frequencies = np.array([1.0, 3.0, 7.0])

    response = direct_frequency_response(
        np.array([[stiffness]]),
        np.array([[mass]]),
        np.array([[damping]]),
        np.array([1.0]),
        frequencies,
        is_unbalanced_force=True,
        unbalanced_mass_product=mass_eccentricity,
    )[:, 0]

    omega = 2.0 * np.pi * frequencies
    expected_force = mass_eccentricity * omega**2
    expected = expected_force / ((stiffness - omega**2 * mass) + 1j * omega * damping)

    np.testing.assert_allclose(response, expected, rtol=1e-12, atol=1e-12)


def test_rayleigh_damping_matrix_is_alpha_m_plus_beta_k():
    from analysis_core.analisis_modal_3d.analysis.damping import rayleigh_damping_matrix

    m = np.array([[2.0, 0.1], [0.1, 3.0]])
    k = np.array([[100.0, -25.0], [-25.0, 80.0]])
    alpha = 0.12
    beta = 0.004

    damping = rayleigh_damping_matrix(m, k, alpha, beta)

    np.testing.assert_allclose(damping, alpha * m + beta * k, rtol=1e-12, atol=1e-12)


def test_run_harmonic_cantilever_matches_independent_free_dof_matrix_solution():
    from analysis_core.api_adapters import _build_structure
    from analysis_core.analisis_modal_3d.analysis.assembler import assemble_global_matrices
    from analysis_core.analisis_modal_3d.analysis.damping import rayleigh_damping_matrix
    from analysis_core.analisis_modal_3d.analysis.frequency_response import direct_frequency_response
    from analysis_core.analisis_modal_3d.analysis.modal import modal_analysis

    structure_data = simple_cantilever_structure()
    structure_data["settings"] = {"mass_type": "consistent"}
    frequencies = np.linspace(2.0, 14.0, 6)
    damping_ratio = 0.025

    adapter_result = run_harmonic_analysis(
        {
            "structure": structure_data,
            "freq_start": float(frequencies[0]),
            "freq_end": float(frequencies[-1]),
            "num_points": len(frequencies),
            "damping_ratio": damping_ratio,
        }
    )
    assert "error" not in adapter_result

    structure, nodes_map_core, _, id_map, _ = _build_structure(structure_data, mass_type="consistent")
    k_global, m_global = assemble_global_matrices(structure, include_mass=True)
    natural_freqs, _, _ = modal_analysis(k_global, m_global, structure, num_modes=1, debug=False)
    beta = 2.0 * damping_ratio / (2.0 * np.pi * natural_freqs[0])
    c_global = rayleigh_damping_matrix(m_global, k_global, 0.0, beta)

    force = np.zeros(structure.num_dofs)
    node2 = nodes_map_core[id_map[2]]
    force[node2.dofs[1]] = 1000.0

    constrained = structure.get_constrained_dofs()
    free = np.setdiff1d(np.arange(structure.num_dofs), constrained)
    expected_free = direct_frequency_response(
        k_global[free, :][:, free],
        m_global[free, :][:, free],
        c_global[free, :][:, free],
        force[free],
        frequencies,
    )

    full = np.zeros((len(frequencies), structure.num_dofs), dtype=np.complex128)
    full[:, free] = expected_free
    expected_uy = full[:, node2.dofs[1]]
    components = adapter_result["node_displacement_components"][2]

    np.testing.assert_allclose(components["uy_real_m"], expected_uy.real, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(components["uy_imag_m"], expected_uy.imag, rtol=1e-8, atol=1e-12)

    expected_magnitude = np.sqrt(
        np.abs(full[:, node2.dofs[0]]) ** 2
        + np.abs(full[:, node2.dofs[1]]) ** 2
        + np.abs(full[:, node2.dofs[2]]) ** 2
    )
    np.testing.assert_allclose(adapter_result["response_amplitudes"][2], expected_magnitude, rtol=1e-8, atol=1e-12)
