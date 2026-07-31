"""
Módulo para el análisis de respuesta en frecuencia de estructuras.
"""
import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from typing import Any


SPARSE_SOLVER_DOF_THRESHOLD = 180


def direct_frequency_response(
    K: Any,
    M: Any,
    C: Any,
    force_vector_amplitude: np.ndarray,
    frequency_range_hz: np.ndarray,
    is_unbalanced_force: bool = False,
    unbalanced_mass_product: float = 0.0,
    normalize_unbalanced_direction: bool = True,
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
        normalize_unbalanced_direction (bool): Si es True, normaliza el vector de
            dirección recibido. Si el llamador ya eliminó GDL restringidos, conviene
            pasar False para preservar la proyección física sobre los GDL libres.

    Returns:
        np.ndarray: Matriz de respuestas complejas U(ω). Cada fila corresponde a una
                    frecuencia, cada columna a un grado de libertad.
    """
    num_dofs = K.shape[0]
    num_freqs = len(frequency_range_hz)
    
    # Optimización para sistemas medianos/grandes:
    # el ensamblador FEM entrega matrices sparse. Convertirlas a denso y resolver
    # O(n³) en cada frecuencia se vuelve muy lento desde ~200 GDL libres.
    use_sparse_solver = sp.issparse(K) and num_dofs >= SPARSE_SOLVER_DOF_THRESHOLD
    
    # Normalizar el vector de dirección de la fuerza si es necesario
    force_direction = np.zeros(num_dofs, dtype=np.complex128)
    if is_unbalanced_force:
        if normalize_unbalanced_direction:
            norm = np.linalg.norm(force_vector_amplitude)
            if norm > 1e-9:
                force_direction = force_vector_amplitude / norm
            else:
                # Si el vector de fuerza es cero, no hay nada que hacer
                return np.zeros((num_freqs, num_dofs), dtype=np.complex128)
        else:
            force_direction = force_vector_amplitude.astype(np.complex128, copy=False)
            if np.linalg.norm(force_direction) <= 1e-12:
                return np.zeros((num_freqs, num_dofs), dtype=np.complex128)
    
    # Array para almacenar los resultados complejos de desplazamiento U(ω)
    complex_displacements = np.zeros((num_freqs, num_dofs), dtype=np.complex128)

    if use_sparse_solver:
        K_solver = K.tocsc() if sp.issparse(K) else sp.csc_matrix(K)
        M_solver = M.tocsc() if sp.issparse(M) else sp.csc_matrix(M)
        C_solver = C.tocsc() if sp.issparse(C) else sp.csc_matrix(C)
    else:
        # La conversión sparse -> dense se hace una sola vez, no en cada punto del barrido.
        K_solver = K.toarray() if sp.issparse(K) else np.asarray(K)
        M_solver = M.toarray() if sp.issparse(M) else np.asarray(M)
        C_solver = C.toarray() if sp.issparse(C) else np.asarray(C)

    print(
        f"Iniciando barrido de frecuencias: {num_freqs} puntos, {num_dofs} GDL, "
        f"solver={'sparse' if use_sparse_solver else 'dense'}."
    )
    for i, freq_hz in enumerate(frequency_range_hz):
        omega = 2 * np.pi * freq_hz

        # Construir la matriz de impedancia dinámica Z(ω)
        Z = (K_solver - omega**2 * M_solver) + 1j * (omega * C_solver)

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
                U = splu(Z.tocsc()).solve(F)
            else:
                # scipy.linalg.solve con check_finite=False reduce overhead dentro del sweep.
                U = scipy.linalg.solve(
                    Z,
                    F,
                    assume_a="gen",
                    check_finite=False,
                    overwrite_a=False,
                    overwrite_b=False,
                )
            complex_displacements[i, :] = U
        except (np.linalg.LinAlgError, RuntimeError):
            print(f"Advertencia: Error al resolver para la frecuencia {freq_hz:.2f} Hz. La matriz puede ser singular.")
            complex_displacements[i, :] = np.nan

    print("Barrido de frecuencias completado.")
    return complex_displacements
