import numpy as np
import scipy.linalg
from scipy.sparse.linalg import eigsh
from scipy import sparse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..structures.structure import Structure


def _check_positive_definiteness(M_dense: np.ndarray, name: str = "M_free") -> bool:
    try:
        np.linalg.cholesky(M_dense)
        print(f"[DIAG] {name} is positive definite: YES")
        return True
    except np.linalg.LinAlgError:
        print(f"[DIAG] {name} is positive definite: NO (Cholesky failed)")
        return False


def _print_matrix_diagnostics(M_sparse, name: str = "M_free"):
    M_dense = M_sparse.toarray() if sparse.issparse(M_sparse) else M_sparse
    try:
        cond_num = np.linalg.cond(M_dense)
        print(f"[DIAG] {name} condition number: {cond_num:.6e}")
        if cond_num > 1e12:
            print(f"[DIAG] WARNING: {name} is ill-conditioned (cond > 1e12)")
    except Exception as e:
        print(f"[DIAG] {name} condition number: FAILED ({e})")
    
    _check_positive_definiteness(M_dense, name)
    print(f"[DIAG] {name} shape: {M_sparse.shape}, nnz: {M_sparse.nnz if sparse.issparse(M_sparse) else 'dense'}")
    print(f"[DIAG] {name} diagonal range: [{M_dense.diagonal().min():.6e}, {M_dense.diagonal().max():.6e}]")


def _print_eigensolution_diagnostics(eigvals: np.ndarray, eigvecs: np.ndarray):
    print(f"[DIAG] eigvals count: {len(eigvals)}")
    print(f"[DIAG] eigvals range: [{eigvals.min():.6e}, {eigvals.max():.6e}]")
    print(f"[DIAG] eigvals (all): {eigvals}")
    print(f"[DIAG] eigvecs max absolute value: {np.abs(eigvecs).max():.6e}")
    print(f"[DIAG] eigvecs shape: {eigvecs.shape}")
    neg_count = np.sum(eigvals < 0)
    if neg_count > 0:
        print(f"[DIAG] WARNING: {neg_count} negative eigenvalues detected!")


def modal_analysis(
    K: Any,
    M: Any,
    structure: "Structure",
    num_modes=6,
    debug=False,
):
    if debug:
        print("\n" + "="*60)
        print("[DIAG] MODAL ANALYSIS DIAGNOSTICS")
        print("="*60)
    
    num_dofs = K.shape[0]
    constrained_dofs = structure.get_constrained_dofs()
    free_dofs = np.setdiff1d(np.arange(num_dofs), constrained_dofs)
    
    if debug:
        print(f"[DIAG] Total DOFs: {num_dofs}, Free DOFs: {len(free_dofs)}, Constrained DOFs: {len(constrained_dofs)}")

    K_free = K[free_dofs, :][:, free_dofs]
    M_free = M[free_dofs, :][:, free_dofs]
    
    if debug:
        print("\n[DIAG] --- Before regularization ---")
        _print_matrix_diagnostics(M_free, "M_free (original)")

    m_diag = M_free.diagonal()
    if np.any(m_diag < 1e-15):
        M_free += sparse.eye(M_free.shape[0]) * 1e-9
        if debug:
            print("\n[DIAG] --- After regularization (1e-9 on diagonal) ---")
            _print_matrix_diagnostics(M_free, "M_free (regularized)")
            _print_matrix_diagnostics(K_free, "K_free")

    actual_num_modes = min(num_modes, len(free_dofs) - 2)
    if debug:
        print(f"\n[DIAG] Requested modes: {num_modes}, Actual modes to compute: {actual_num_modes}")
    
    disparity = M_free.diagonal().max() / (M_free.diagonal().min() + 1e-12)
    
    if actual_num_modes <= 0 or disparity > 1e12:
        if debug:
            print(f"[DIAG] Using dense solver (scipy.linalg.eigh) - disparity {disparity:.1e}")
        actual_num_modes = min(num_modes, len(free_dofs))
        eigvals, eigvecs_red = scipy.linalg.eigh(K_free.toarray(), M_free.toarray())
        actual_num_modes = min(num_modes, len(eigvals))
        eigvals = eigvals[:actual_num_modes]
        eigvecs_red = eigvecs_red[:, :actual_num_modes]
    else:
        if debug:
            print(f"[DIAG] Using sparse solver (eigsh) with disparity {disparity:.1e}")
        try:
            # Intentar resolver usando el solver sparse de Arpack
            eigvals, eigvecs_red = eigsh(K_free, k=actual_num_modes, M=M_free, which='LM', sigma=0)
        except Exception as e:
            if debug:
                print(f"[DIAG] eigsh failed: {e}. Falling back to dense solver.")
            # Fallback a solver denso si eigsh falla (común en mecanismos)
            eigvals, eigvecs_red = scipy.linalg.eigh(K_free.toarray(), M_free.toarray())
            actual_num_modes = min(num_modes, len(eigvals))
            eigvals = eigvals[:actual_num_modes]
            eigvecs_red = eigvecs_red[:, :actual_num_modes]

    if debug:
        print("\n[DIAG] --- Eigensolution Results ---")
        _print_eigensolution_diagnostics(eigvals, eigvecs_red)

    mode_shapes = np.zeros((num_dofs, actual_num_modes))
    mode_shapes[free_dofs, :] = eigvecs_red

    if debug:
        print("\n[DIAG] --- Eigenvector Mass Normalization (phi^T M phi = 1) ---")
    
    for i in range(actual_num_modes):
        phi_i = mode_shapes[:, i]
        m_modal = phi_i.T @ M @ phi_i
        if debug:
            print(f"[DIAG] Mode {i+1}: m_modal before normalization = {m_modal:.6e}")
        if m_modal > 1e-12:
            mode_shapes[:, i] /= np.sqrt(m_modal)
            if debug:
                phi_normalized = mode_shapes[:, i]
                m_modal_check = phi_normalized.T @ M @ phi_normalized
                print(f"[DIAG] Mode {i+1}: m_modal after normalization = {m_modal_check:.6e}")

    frequencies = np.sqrt(np.maximum(eigvals, 0)) / (2 * np.pi)

    mass_participation = np.zeros((actual_num_modes, 6))
    R = np.zeros((num_dofs, 6))
    for j in range(6):
        R[j::6, j] = 1

    total_mass_vector = np.zeros(6)
    for j in range(6):
        Rj = R[:, j]
        total_mass_vector[j] = Rj.T @ M @ Rj
    
    if debug:
        print(f"[DIAG] total_mass_vector (R^T M R): {total_mass_vector}")

    L = mode_shapes.T @ M @ R
    effective_modal_mass = L**2

    for j in range(6):
        if total_mass_vector[j] > 1e-9:
            mass_participation[:, j] = (effective_modal_mass[:, j] / total_mass_vector[j]) * 100

    return frequencies, mode_shapes, mass_participation
