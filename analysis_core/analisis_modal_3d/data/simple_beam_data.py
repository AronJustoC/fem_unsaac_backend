"""
Módulo para proporcionar los datos de entrada de una viga simple 3D.
"""

def get_structure_data():
    """
    Retorna un diccionario con todos los datos de entrada para la estructura.
    """
    data = {
        "materials": {
            "DefaultMaterial": {
                "E": 210e9,  # Módulo de Young (Pa)
                "G": 80e9,   # Módulo de Cortante (Pa)
                "rho": 7850  # Densidad (kg/m^3)
            }
        },
        "sections": {
            "DefaultSection": {
                "area": 0.015,
                "Ix": 8.5e-5,
                "Iy": 1.25e-4,
                "Iz": 5.8e-5
            }
        },
        "nodes": [
            # ID, X, Y, Z en metros
            (0, 0, 0, 0),
            (1, 5, 0, 0),
            (2, 10, 0, 0),
            (3, 0, 5, 0),
            (4, 5, 5, 0),
            (5, 10, 5, 0),
            (6, 5, 2.5, 5)
        ],
        "elements": [
            # (nodo_inicial_ID, nodo_final_ID, sección, material)
            (0, 1, "DefaultSection", "DefaultMaterial"),
            (1, 2, "DefaultSection", "DefaultMaterial"),
            (3, 4, "DefaultSection", "DefaultMaterial"),
            (4, 5, "DefaultSection", "DefaultMaterial"),
            (0, 3, "DefaultSection", "DefaultMaterial"),
            (1, 4, "DefaultSection", "DefaultMaterial"),
            (2, 5, "DefaultSection", "DefaultMaterial"),
            (0, 6, "DefaultSection", "DefaultMaterial"),
            (1, 6, "DefaultSection", "DefaultMaterial"),
            (2, 6, "DefaultSection", "DefaultMaterial"),
            (3, 6, "DefaultSection", "DefaultMaterial"),
            (4, 6, "DefaultSection", "DefaultMaterial"),
            (5, 6, "DefaultSection", "DefaultMaterial")
        ],
        "constraints": {
            0: ["ux", "uy", "uz", "rx", "ry", "rz"], # n1
            2: ["ux", "uy", "uz", "rx", "ry", "rz"], # n3
            3: ["ux", "uy", "uz", "rx", "ry", "rz"], # n4
            5: ["ux", "uy", "uz", "rx", "ry", "rz"]  # n6
        },
        "masses": {
            # No hay masas puntuales explícitas en el ejemplo original, pero se incluye la estructura
            "node_id": None,
            "mass": 0.0
        },
        "static_loads": [], # No hay cargas estáticas explícitas en el ejemplo original
        "analysis_settings": {
            "modal_analysis": {
                "num_modes": 5 # El ejemplo original plotea los primeros 5 modos
            },
            "damping": {
                "zeta_target": 0.02,
                "rayleigh_modes": [1, 3]
            },
            "frequency_response": {
                "freq_range_hz": [1, 5, 10, 15, 20], # Frecuencias de ejemplo
                "unbalanced_force": {
                    "mass": 0.0,              # kg
                    "eccentricity_mm": 0.0   # mm
                }
            },
            "post_processing": {
                "nodes_of_interest": [],
                "num_modes_to_plot": 3, # El ejemplo original plotea los primeros 3 modos
                "scale_factor": 50 # Factor de escala por defecto para visualización
            }
        }
    }
    return data
