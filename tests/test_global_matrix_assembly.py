from copy import deepcopy

import numpy as np
import pytest

from analysis_core.api_adapters import (
    build_global_matrix_bundle,
    get_global_matrix_view,
)


def _two_element_structure():
    return {
        "nodes": [
            {"id": 1, "coords": [0.0, 0.0, 0.0]},
            {"id": 2, "coords": [2.0, 0.0, 0.0], "mass": 5.0},
            {"id": 3, "coords": [4.0, 0.0, 0.0]},
        ],
        "elements": [
            {"id": 1, "node_ids": [1, 2], "material_id": 1, "section_id": 1},
            {"id": 2, "node_ids": [2, 3], "material_id": 1, "section_id": 1},
        ],
        "materials": [
            {"id": 1, "name": "Acero", "E": 210e9, "nu": 0.3, "rho": 7850},
        ],
        "sections": [
            {
                "id": 1,
                "name": "Rectangular",
                "area": 0.003,
                "Iy": 8e-6,
                "Iz": 1.2e-5,
                "J": 2e-6,
            },
        ],
        "restraints": {"1": ["ux", "uy", "uz", "rx", "ry", "rz"]},
        "loads": [],
    }


def test_global_bundle_assembles_shared_node_and_reduces_constrained_dofs():
    data = _two_element_structure()
    bundle = build_global_matrix_bundle(deepcopy(data))
    K = bundle["matrices"][("stiffness", "full")]
    M = bundle["matrices"][("mass", "full")]

    assert K.shape == (18, 18)
    assert M.shape == (18, 18)
    assert bundle["matrices"][("stiffness", "free")].shape == (12, 12)
    assert bundle["matrices"][("mass", "free")].shape == (12, 12)
    assert bundle["metadata"]["total_dofs"] == 18
    assert bundle["metadata"]["free_dofs"] == 12
    assert bundle["metadata"]["constrained_dofs"] == 6

    element_axial = 210e9 * 0.003 / 2.0
    assert K[0, 0] == pytest.approx(element_axial)
    assert K[0, 6] == pytest.approx(-element_axial)
    # El GDL ux del nodo compartido recibe la suma de ambos elementos.
    assert K[6, 6] == pytest.approx(2.0 * element_axial)
    assert K[6, 12] == pytest.approx(-element_axial)

    element_mass = 7850 * 0.003 * 2.0
    # Mitad de cada barra más la masa puntual del nodo 2.
    assert M[6, 6] == pytest.approx(element_mass + 5.0)


def test_global_matrix_view_returns_complete_heatmap_and_numeric_window():
    bundle = build_global_matrix_bundle(_two_element_structure())
    result = get_global_matrix_view(
        bundle,
        matrix_kind="stiffness",
        matrix_scope="full",
        row_start=5,
        col_start=6,
        window_size=6,
        heatmap_bins=16,
    )

    assert result["matrix"]["dimension"] == 18
    assert result["matrix"]["nnz"] > 0
    assert result["matrix"]["density"] < 1
    assert result["matrix"]["symmetry_error"] == pytest.approx(0.0)
    assert result["heatmap"]["bins"] == 16
    assert len(result["heatmap"]["values"]) == 16
    assert len(result["heatmap"]["values"][0]) == 16
    assert sum(sum(row) for row in result["heatmap"]["counts"]) == result["matrix"]["nnz"]

    window = result["window"]
    assert window["row_start"] == 5
    assert window["col_start"] == 6
    assert len(window["values"]) == 6
    assert all(len(row) == 6 for row in window["values"])
    assert window["row_labels"][1] == "N2·ux"
    assert window["col_labels"][0] == "N2·ux"
    assert window["values"][1][0] == pytest.approx(
        bundle["matrices"][("stiffness", "full")][6, 6]
    )


def test_free_mass_view_matches_modal_dofs_and_visual_metadata_is_invariant():
    baseline_data = _two_element_structure()
    baseline = build_global_matrix_bundle(deepcopy(baseline_data))

    visual_data = deepcopy(baseline_data)
    visual_data["sections"][0].update({
        "visual_shape": "h",
        "visual_height": 0.5,
        "visual_width": 0.3,
        "visual_web_thickness": 0.02,
        "visual_flange_thickness": 0.03,
    })
    visual_data["elements"][0]["visual_rotation_deg"] = 73.0
    visual = build_global_matrix_bundle(visual_data)

    for key in baseline["matrices"]:
        np.testing.assert_allclose(
            baseline["matrices"][key].toarray(),
            visual["matrices"][key].toarray(),
        )

    result = get_global_matrix_view(
        baseline,
        matrix_kind="mass",
        matrix_scope="free",
        row_start=999,
        col_start=999,
        window_size=12,
        heatmap_bins=64,
    )
    assert result["matrix"]["dimension"] == 12
    assert result["window"]["row_start"] == 0
    assert result["window"]["col_start"] == 0
    assert result["window"]["row_labels"][0] == "N2·ux"
    assert all(not label.startswith("N1·") for label in result["window"]["row_labels"])
