"""
Módulo para proporcionar los datos de entrada de un ejemplo de estructura vacía (pórtico 2D simple).
"""

def get_structure_data():
    """
    Retorna un diccionario con todos los datos de entrada para la estructura.
    """
    data = {
        "materials": {
            "acero": {
                "E": 200e9,  # Módulo de elasticidad (Pa)
                "G": 77e9,   # Módulo de corte (Pa)
                "rho": 7850, # Densidad (kg/m³)
            }
        },
        "sections": {
            "columna_30x30": {
                "area": 0.3 * 0.3,
                "Ix": (1/12) * 0.3 * 0.3**3,
                "Iy": (1/12) * 0.3 * 0.3**3,
                "Iz": (1/12) * 0.3 * 0.3**3,
            },
            "viga_40x20": {
                "area": 0.4 * 0.2,
                "Ix": 0.0001,
                "Iy": (1/12) * 0.2 * 0.4**3,
                "Iz": (1/12) * 0.4 * 0.2**3,
            },
        },
        "nodes": [
            # ID, X, Y, Z en metros
            (0, 0, 0, 0), # Original ID 1
            (1, 5, 0, 0), # Original ID 2
            (2, 0, 0, 3), # Original ID 3
            (3, 5, 0, 3)  # Original ID 4
        ],
        "elements": [
            # Columnas
            (0, 2, "columna_30x30", "acero"),
            (1, 3, "columna_30x30", "acero"),
            # Vigas
            (2, 3, "viga_40x20", "acero"),
        ],
        "constraints": {
            0: ["ux", "uy", "uz", "rx", "ry", "rz"], # Empotramiento en la base (n1)
            1: ["ux", "uy", "uz", "rx", "ry", "rz"], # Empotramiento en la base (n2)
            2: ["uy", "rx", "rz"], # Restricción 2D para n3
            3: ["uy", "rx", "rz"]  # Restricción 2D para n4
        },
        "masses": {
            "node_id": None,
            "mass": 0.0
        },
        "static_loads": [],
        "analysis_settings": {
            "modal_analysis": {
                "num_modes": 5
            },
            "damping": {
                "zeta_target": 0.02,
                "rayleigh_modes": [1, 3]
            },
            "frequency_response": {
                "freq_range_hz": [1, 5, 10, 15, 20],
                "unbalanced_force": {
                    "mass": 0.0,
                    "eccentricity_mm": 0.0
                }
            },
            "post_processing": {
                "nodes_of_interest": [],
                "num_modes_to_plot": 5,
                "scale_factor": 0.5
            }
        }
    }
    return data
