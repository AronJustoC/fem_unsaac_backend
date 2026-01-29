"""
Módulo para el análisis de respuesta en frecuencia de estructuras.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from tqdm import tqdm

from typing import Any

def direct_frequency_response(
    K: Any,
    M: Any,
    C: Any,
    force_vector_amplitude: np.ndarray,
    frequency_range_hz: np.ndarray,
    is_unbalanced_force: bool = False,
    unbalanced_mass_product: float = 0.0
) -> np.ndarray:
    """
    Calcula la respuesta en estado estacionario (amplitudes y fases) de una estructura
    a lo largo de un rango de frecuencias utilizando el método directo.

    Resuelve la ecuación: (K - ω^2*M + i*ω*C) * U(ω) = F(ω)

    Args:
        K (np.ndarray): Matriz de rigidez global.
        M (np.ndarray): Matriz de masa global.
        C (np.ndarray): Matriz de amortiguamiento global.
        force_vector_amplitude (np.ndarray): Vector de amplitud de la fuerza (F0).
            Si is_unbalanced_force es True, este vector indica la DIRECCIÓN de la fuerza,
            y su magnitud se ignora.
        frequency_range_hz (np.ndarray): Vector de frecuencias a analizar (en Hz).
        is_unbalanced_force (bool): Si es True, la magnitud de la fuerza será proporcional
            al cuadrado de la frecuencia (F = (m*e) * ω^2).
        unbalanced_mass_product (float): El producto de la masa desbalanceada por la
            excentricidad (m*e). Requerido si is_unbalanced_force es True.

    Returns:
        np.ndarray: Matriz de respuestas complejas U(ω). Cada fila corresponde a una
                    frecuencia, cada columna a un grado de libertad.
    """
    num_dofs = K.shape[0]
    num_freqs = len(frequency_range_hz)
    
    # Optimización para sistemas grandes
    use_sparse_solver = (num_dofs > 1000)
    
    # Normalizar el vector de dirección de la fuerza si es necesario
    force_direction = np.zeros(num_dofs)
    if is_unbalanced_force:
        norm = np.linalg.norm(force_vector_amplitude)
        if norm > 1e-9:
            force_direction = force_vector_amplitude / norm
        else:
            # Si el vector de fuerza es cero, no hay nada que hacer
            return np.zeros((num_freqs, num_dofs), dtype=np.complex128)
    
    # Array para almacenar los resultados complejos de desplazamiento U(ω)
    complex_displacements = np.zeros((num_freqs, num_dofs), dtype=np.complex128)

    print("Iniciando barrido de frecuencias...")
    for i, freq_hz in enumerate(tqdm(frequency_range_hz, desc="Frequency Sweep")):
        omega = 2 * np.pi * freq_hz

        # Construir la matriz de impedancia dinámica Z(ω)
        Z = (K - omega**2 * M) + 1j * (omega * C)

        # Definir el vector de fuerza F(ω)
        if is_unbalanced_force:
            # Fuerza de desbalance: F = (m*e) * ω^2
            force_magnitude = unbalanced_mass_product * omega**2
            F = force_direction * force_magnitude
        else:
            # Fuerza de amplitud constante
            F = force_vector_amplitude

        # Resolver el sistema de ecuaciones lineales Z * U = F
        try:
            if use_sparse_solver:
                # Usar solver sparse para sistemas grandes
                if not sp.issparse(Z):
                    Z_sparse = sp.csc_matrix(Z)
                else:
                    Z_sparse = Z.tocsc()
                U = splu(Z_sparse).solve(F)
            else:
                # Solver denso para sistemas pequeños
                Z_dense = Z.toarray() if sp.issparse(Z) else Z
                U = np.linalg.solve(Z_dense, F)
            complex_displacements[i, :] = U
        except (np.linalg.LinAlgError, RuntimeError):
            print(f"Advertencia: Error al resolver para la frecuencia {freq_hz:.2f} Hz. La matriz puede ser singular.")
            complex_displacements[i, :] = np.nan

    print("Barrido de frecuencias completado.")
    return complex_displacements
