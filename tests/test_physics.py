import pytest
import numpy as np
import time
from analysis_core.api_adapters import run_modal_analysis

def create_large_beam(num_nodes=500):
    nodes = []
    for i in range(num_nodes):
        nodes.append({"id": i, "coords": [float(i), 0.0, 0.0]})
    
    elements = []
    for i in range(num_nodes - 1):
        elements.append({
            "id": i,
            "node_ids": [i, i+1],
            "material_id": 1,
            "section_id": 1
        })
    
    materials = [{"id": 1, "E": 200e9, "nu": 0.3, "rho": 7850}]
    sections = [{"id": 1, "area": 0.01, "Iz": 1e-4, "Iy": 1e-4, "J": 2e-4}]
    restraints = {"0": ["ux", "uy", "uz", "rx", "ry", "rz"]}
    
    return {
        "nodes": nodes, 
        "elements": elements, 
        "materials": materials, 
        "sections": sections,
        "restraints": restraints, 
        "loads": []
    }

@pytest.mark.performance
def test_modal_performance_large_model():
    # Usamos 500 para el test de CI, pero el objetivo es 1000
    structure_data = create_large_beam(500)
    
    start_time = time.time()
    result = run_modal_analysis(structure_data, num_modes=12)
    duration = time.time() - start_time
    
    assert "error" not in result
    assert "frequencies" in result
    assert len(result["frequencies"]) == 12
    assert duration < 30, f"Analysis took too long: {duration:.2f}s"
