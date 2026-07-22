import plotly.graph_objects as go
import pytest
from copy import deepcopy

from analysis_core.api_adapters import get_element_matrices, run_static_analysis
from schemas.structure_schemas import StructureInput
from visualization.plotly_engine import generate_results_figure, generate_structure_figure


def _structure(section):
    return {
        "nodes": [
            {"id": 1, "coords": [0.0, 0.0, 0.0]},
            {"id": 2, "coords": [5.0, 0.0, 0.0]},
        ],
        "elements": [
            {"id": 1, "node_ids": [1, 2], "material_id": 1, "section_id": 1},
        ],
        "materials": [
            {"id": 1, "name": "Acero A36", "E": 210e9, "nu": 0.3, "rho": 7850},
        ],
        "sections": [section],
        "restraints": {},
        "loads": [],
    }


def test_structure_figure_extrudes_i_section_using_real_dimensions():
    data = _structure({
        "id": 1,
        "name": "Perfil I personalizado",
        "A": 0.003,
        "Iz": 1e-4,
        "Iy": 2e-5,
        "J": 1e-6,
        "visual_shape": "i",
        "visual_height": 0.2,
        "visual_width": 0.1,
        "visual_web_thickness": 0.006,
        "visual_flange_thickness": 0.01,
    })

    # También prueba que el contrato Pydantic preserve los nuevos datos geométricos.
    validated = StructureInput.model_validate(data).model_dump()
    figure = generate_structure_figure(validated)
    meshes = [trace for trace in figure.data if isinstance(trace, go.Mesh3d)]

    assert len(meshes) == 1
    mesh = meshes[0]
    assert len(mesh.x) == 24  # dos alas + alma, ocho vértices por prisma
    assert min(mesh.y) == pytest.approx(-50.0)
    assert max(mesh.y) == pytest.approx(50.0)
    assert min(mesh.z) == pytest.approx(-100.0)
    assert max(mesh.z) == pytest.approx(100.0)
    assert "Forma: Perfil I" in mesh.text[0]


def test_legacy_ipe_section_is_inferred_and_not_clipped():
    data = _structure({
        "id": 1,
        "name": "Perfil IPE 200",
        "area": 0.01,
        "Iz": 1e-4,
        "Iy": 1e-4,
        "J": 2e-4,
    })

    figure = generate_structure_figure(data)
    mesh = next(trace for trace in figure.data if isinstance(trace, go.Mesh3d))

    assert min(mesh.z) == pytest.approx(-100.0)
    assert max(mesh.z) == pytest.approx(100.0)
    assert figure.layout.scene.yaxis.range[0] < min(mesh.y)
    assert figure.layout.scene.yaxis.range[1] > max(mesh.y)
    assert figure.layout.scene.zaxis.range[0] < min(mesh.z)
    assert figure.layout.scene.zaxis.range[1] > max(mesh.z)


def test_circular_section_generates_cylindrical_mesh():
    data = _structure({
        "id": 1,
        "name": "Tubo circular",
        "area": 0.00785,
        "Iz": 4.9e-6,
        "Iy": 4.9e-6,
        "J": 9.8e-6,
        "visual_shape": "circular",
        "visual_width": 0.1,
        "visual_height": 0.1,
    })

    figure = generate_structure_figure(data)
    mesh = next(trace for trace in figure.data if isinstance(trace, go.Mesh3d))

    assert len(mesh.x) == 32
    assert min(mesh.y) == pytest.approx(-50.0)
    assert max(mesh.y) == pytest.approx(50.0)
    assert "Forma: Circular" in mesh.text[0]


def test_results_view_uses_solid_as_transparent_reference():
    data = _structure({
        "id": 1,
        "name": "Perfil IPE 200",
        "area": 0.01,
        "Iz": 1e-4,
        "Iy": 1e-4,
        "J": 2e-4,
    })
    displacements = {1: [0.0] * 6, 2: [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]}

    figure = generate_results_figure(data, displacements, scale=10.0)
    mesh = next(trace for trace in figure.data if isinstance(trace, go.Mesh3d))

    assert mesh.opacity == pytest.approx(0.15)
    assert mesh.showlegend is False


@pytest.mark.parametrize(
    ("section_name", "area", "expected_height_mm", "expected_width_mm"),
    [
        ("80x40", 0.008 * 0.004, 8.0, 4.0),
        ("100x80", 0.010 * 0.008, 10.0, 8.0),
        ("80x80", 0.008 * 0.008, 8.0, 8.0),
    ],
)
def test_bailey_rectangles_preserve_scaled_area_and_name_proportions(
    section_name,
    area,
    expected_height_mm,
    expected_width_mm,
):
    data = _structure({
        "id": 1,
        "name": section_name,
        "area": area,
        "Iz": 1e-10,
        "Iy": 1e-10,
        "J": 1e-10,
    })

    figure = generate_structure_figure(data)
    mesh = next(trace for trace in figure.data if isinstance(trace, go.Mesh3d))

    assert max(mesh.z) - min(mesh.z) == pytest.approx(expected_height_mm)
    assert max(mesh.y) - min(mesh.y) == pytest.approx(expected_width_mm)


def test_bailey_h_section_is_recovered_from_area_inertia_and_designation():
    data = _structure({
        "id": 1,
        "name": "H420x180",
        "area": 0.000156,
        "Iz": 1.969333333e-9,
        "Iy": 3.7969333333e-8,
        "J": 3.9938666666e-8,
    })
    data["nodes"][1]["coords"] = [0.0, 0.5, 0.0]

    figure = generate_structure_figure(data)
    mesh = next(trace for trace in figure.data if isinstance(trace, go.Mesh3d))

    assert max(mesh.z) - min(mesh.z) == pytest.approx(42.0)
    assert max(mesh.x) - min(mesh.x) == pytest.approx(18.0)
    assert "Forma: Perfil H" in mesh.text[0]


def test_vertical_rectangle_keeps_height_in_bridge_plane_and_width_out_of_plane():
    data = _structure({
        "id": 1,
        "name": "80x40",
        "area": 0.008 * 0.004,
        "Iz": 1.70666667e-10,
        "Iy": 4.2666667e-11,
        "J": 2.13333334e-10,
    })
    data["nodes"][1]["coords"] = [0.0, 0.0, 0.11]

    figure = generate_structure_figure(data)
    mesh = next(trace for trace in figure.data if isinstance(trace, go.Mesh3d))

    # Montante en Z: los 8 mm quedan en el plano lateral X-Z y 4 mm en Y.
    assert max(mesh.x) - min(mesh.x) == pytest.approx(8.0)
    assert max(mesh.y) - min(mesh.y) == pytest.approx(4.0)


def test_visual_rotation_can_correct_section_without_changing_connectivity():
    data = _structure({
        "id": 1,
        "name": "80x40",
        "area": 0.008 * 0.004,
        "Iz": 1.70666667e-10,
        "Iy": 4.2666667e-11,
        "J": 2.13333334e-10,
    })
    data["elements"][0]["visual_rotation_deg"] = 90.0

    figure = generate_structure_figure(data)
    mesh = next(trace for trace in figure.data if isinstance(trace, go.Mesh3d))

    assert max(mesh.y) - min(mesh.y) == pytest.approx(8.0)
    assert max(mesh.z) - min(mesh.z) == pytest.approx(4.0)


def test_visual_metadata_does_not_change_fem_results():
    data = _structure({
        "id": 1,
        "name": "80x40",
        "area": 0.008 * 0.004,
        "Iz": 1.70666667e-10,
        "Iy": 4.2666667e-11,
        "J": 2.13333334e-10,
    })
    data["nodes"][1]["coords"] = [1.0, 0.0, 0.0]
    data["restraints"] = {"1": ["ux", "uy", "uz", "rx", "ry", "rz"]}
    data["loads"] = [{"id": 1, "node_id": 2, "fx": 0.0, "fy": -10.0, "fz": 0.0}]
    baseline = run_static_analysis(deepcopy(data))

    visual = deepcopy(data)
    visual["sections"][0].update({
        "visual_shape": "h",
        "visual_height": 0.5,
        "visual_width": 0.4,
        "visual_web_thickness": 0.03,
        "visual_flange_thickness": 0.04,
    })
    visual["elements"][0]["visual_rotation_deg"] = 37.0
    changed = run_static_analysis(visual)

    assert changed["displacements"] == pytest.approx(baseline["displacements"])
    assert changed["element_forces"].keys() == baseline["element_forces"].keys()
    for element_id, forces in baseline["element_forces"].items():
        assert changed["element_forces"][element_id] == pytest.approx(forces)
    assert changed["reactions"] == pytest.approx(baseline["reactions"])
    assert changed["stresses"] == pytest.approx(baseline["stresses"])


def test_element_matrices_are_returned_lazily_with_didactic_context():
    data = _structure({
        "id": 1,
        "name": "Sección importada",
        "area": 0.003,
        "Iz": 1.2e-5,
        "Iy": 8e-6,
        "J": 2e-6,
        "visual_shape": "i",
        "visual_height": 0.4,
        "visual_width": 0.2,
    })
    data["nodes"][1]["coords"] = [3.0, 4.0, 1.0]

    result = get_element_matrices(data, 1)

    assert "error" not in result
    assert result["element_id"] == 1
    assert result["node_ids"] == [1, 2]
    assert result["length_m"] == pytest.approx((3**2 + 4**2 + 1**2) ** 0.5)
    assert len(result["dof_labels"]) == 12
    assert result["properties"]["section_name"] == "Sección importada"
    assert result["properties"]["element_mass_kg"] == pytest.approx(
        7850 * 0.003 * result["length_m"]
    )

    for matrix_kind in ("stiffness", "mass"):
        for frame in ("local", "global"):
            matrix = result["matrices"][matrix_kind][frame]
            assert len(matrix) == 12
            assert all(len(row) == 12 for row in matrix)
            for row_index in range(12):
                for column_index in range(12):
                    assert matrix[row_index][column_index] == pytest.approx(
                        matrix[column_index][row_index]
                    )

    # La barra inclinada exige una rotación: K local y global no son la misma vista.
    assert result["matrices"]["stiffness"]["local"] != result["matrices"]["stiffness"]["global"]


def test_element_matrix_lookup_reports_unknown_element():
    result = get_element_matrices(
        _structure({
            "id": 1,
            "name": "Rectangular",
            "area": 0.003,
            "Iz": 1e-5,
            "Iy": 8e-6,
            "J": 2e-6,
        }),
        999,
    )

    assert "error" in result
    assert "999" in result["error"]
