"""
Módulo para el análisis de vibración forzada utilizando superposición modal.
"""
import numpy as np
from tqdm import tqdm

def harmonic_analysis_superposition(
    natural_frequencies_rad: np.ndarray,
    mode_shapes: np.ndarray,
    M: np.ndarray,
    K: np.ndarray,
    C: np.ndarray,
    force_vector: np.ndarray,
    force_frequency_rad: float,
    time_array: np.ndarray
) -> np.ndarray:
    """
    Calcula la respuesta en estado estacionario de una estructura bajo una carga armónica
    utilizando el método de superposición modal.

    Args:
        natural_frequencies_rad (np.ndarray): Vector de frecuencias naturales (rad/s).
        mode_shapes (np.ndarray): Matriz de formas modales (cada columna es un modo).
        M (np.ndarray): Matriz de masa global.
        K (np.ndarray): Matriz de rigidez global.
        C (np.ndarray): Matriz de amortiguamiento global.
        force_vector (np.ndarray): Vector de amplitud de la fuerza armónica (F0).
        force_frequency_rad (float): Frecuencia de la fuerza de excitación (rad/s).
        time_array (np.ndarray): Array de puntos en el tiempo para calcular la respuesta.

    Returns:
        np.ndarray: Matriz de desplazamientos de la estructura en el tiempo.
                    Cada fila corresponde a un grado de libertad, cada columna a un paso de tiempo.
    """
    num_modes = len(natural_frequencies_rad)
    num_dofs = M.shape[0]

    # 1. Desacoplar las ecuaciones (calcular propiedades modales)
    modal_mass = np.diag(mode_shapes.T @ M @ mode_shapes)
    modal_stiffness = np.diag(mode_shapes.T @ K @ mode_shapes)
    modal_damping = np.diag(mode_shapes.T @ C @ mode_shapes)
    
    # Verificar consistencia de rigidez modal y frecuencias naturales
    # k_modal_check = modal_mass * natural_frequencies_rad**2
    # print("Rigidez Modal Calculada:", modal_stiffness)
    # print("Rigidez Modal (M*w^2):", k_modal_check)

    modal_force = mode_shapes.T @ force_vector

    # 2. Resolver para la respuesta en estado estacionario en coordenadas modales
    # La solución para q(t) para una fuerza F0 * cos(Ωt) es q(t) = Q * cos(Ωt - θ)
    # donde Q es la amplitud de la respuesta modal.
    
    # Amplitud de la respuesta compleja en coordenadas modales
    Q_complex = np.zeros(num_modes, dtype=np.complex128)
    
    for i in range(num_modes):
        if modal_mass[i] > 1e-9: # Evitar división por cero si la masa modal es nula
            denominator = (modal_stiffness[i] - force_frequency_rad**2 * modal_mass[i]) + 1j * (force_frequency_rad * modal_damping[i])
            Q_complex[i] = modal_force[i] / denominator

    # 3. Calcular la respuesta en el tiempo en coordenadas modales
    q_t = np.zeros((num_modes, len(time_array)))
    for i in range(num_modes):
        # La respuesta en el tiempo es la parte real de Q_complex * e^(j*Ω*t)
        q_t[i, :] = np.real(Q_complex[i] * np.exp(1j * force_frequency_rad * time_array))

    # 4. Transformar de nuevo a coordenadas físicas
    u_t = mode_shapes @ q_t
    
    print("Análisis de vibración forzada completado.")
    return u_t
