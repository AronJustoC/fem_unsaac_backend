import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve

def static_analysis(K, F):
    """
    Resuelve el problema estático [K]{u} = {F} para encontrar los desplazamientos {u}.
    """
    print("Iniciando análisis estático...")
    try:
        if issparse(K):
            u = spsolve(K, F)
        else:
            u = np.linalg.solve(K, F)
        
        u = u.astype(np.float64)
        print("Análisis estático completado exitosamente.")
        return u
    except Exception as e:
        print(f"Error en análisis estático: {str(e)}")
        return None
