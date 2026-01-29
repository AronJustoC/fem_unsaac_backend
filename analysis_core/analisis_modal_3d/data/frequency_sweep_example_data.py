"""
Módulo para proporcionar los datos de entrada de un ejemplo de barrido de frecuencia.
"""
import numpy as np

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
                "area": 0.05,
                "Ix": 1.67e-4,
                "Iy": 8.33e-5,
                "Iz": 8.33e-5
            }
        },
        "nodes": [
            # ID, X, Y, Z en metros
            (i, x, 0, 0) for i, x in enumerate(np.linspace(0, 10.0, 11))
        ],
        "elements": [
            # (nodo_inicial_ID, nodo_final_ID, sección, material)
            (i, i+1, "DefaultSection", "DefaultMaterial") for i in range(10)
        ],
        "constraints": {
            0: ["ux", "uy", "uz", "rx"], # Apoyo fijo
            10: ["uy", "uz"] # Apoyo móvil
        },
        "masses": {
            "node_id": None,
            "mass": 0.0
        },
        "static_loads": [],
        "analysis_settings": {
            "modal_analysis": {
                "num_modes": 10
            },
            "damping": {
                "zeta_target": 0.02,
                "rayleigh_modes": [1, 3] # Usar el primer y tercer modo (1-indexed)
            },
            "frequency_response": {
                "freq_range_hz": [15.0], # Frecuencia objetivo
                "unbalanced_force": {
                    "mass": 1.0,              # kg (asumiendo que unbalanced_mass_product es m*e)
                    "eccentricity_mm": 1.0   # mm (asumiendo que unbalanced_mass_product es m*e)
                }
            },
            "post_processing": {
                "nodes_of_interest": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "num_modes_to_plot": 5, # Por defecto, aunque este ejemplo no plotea modos
                "scale_factor": 1.0 # Factor de escala por defecto para visualización
            }
        }
    }
    return data
