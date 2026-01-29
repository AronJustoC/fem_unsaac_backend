
import requests
import json

def test_harmonic_analysis():
    url = "http://127.0.0.1:8000/api/analysis/harmonic"
    
    # Datos de una estructura simple (una viga de 2 nodos)
    payload = {
        "structure": {
            "nodes": [
                {"id": 1, "coords": [0.0, 0.0, 0.0]},
                {"id": 2, "coords": [1.0, 0.0, 0.0]}
            ],
            "materials": [
                {"id": 1, "E": 2.1e11, "nu": 0.3, "rho": 7850.0}
            ],
            "sections": [
                {"id": 1, "area": 0.01, "Iz": 1e-5, "Iy": 1e-5, "J": 2e-5}
            ],
            "elements": [
                {"id": 1, "node_ids": [1, 2], "material_id": 1, "section_id": 1}
            ],
            "loads": [
                {"node_id": 2, "fy": 1000.0}
            ],
            "restraints": {
                "1": ["ux", "uy", "uz", "rx", "ry", "rz"]
            }
        },
        "freq_start": 1.0,
        "freq_end": 100.0,
        "num_points": 10,
        "damping_ratio": 0.02
    }

    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            print("Analisis exitoso!")
            print(f"Frecuencias analizadas: {len(results['frequencies_sweep'])}")
            print(f"Nodos con respuesta: {list(results['response_amplitudes'].keys())}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error de conexion: {e}")

if __name__ == "__main__":
    test_harmonic_analysis()
