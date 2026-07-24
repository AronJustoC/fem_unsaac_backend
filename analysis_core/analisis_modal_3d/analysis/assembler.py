import numpy as np
from scipy import sparse

from ..structures.structure import Structure


def _assemble_element_matrices(structure: Structure, include_mass: bool):
    """Ensambla K (y M si include_mass) sumando matrices de elemento vectorizado via COO.

    scipy.sparse.coo_matrix suma automáticamente entradas duplicadas (i,j) al
    convertir a CSR, que es exactamente la semántica de acumulación que necesita
    el ensamblado por elementos (equivalente a K[i,j] += valor por cada elemento).
    """
    num_dofs = structure.num_dofs
    elements = structure.elements
    empty = sparse.csr_matrix((num_dofs, num_dofs), dtype=np.float64)

    if not elements:
        return empty, (empty if include_mass else empty)

    dof_indices = np.array(
        [[dof for node in element.nodes for dof in node.dofs] for element in elements],
        dtype=np.int64,
    )
    rows = np.repeat(dof_indices, 12, axis=1).reshape(-1)
    cols = np.tile(dof_indices, 12).reshape(-1)

    k_data = np.concatenate([element.k_global.reshape(-1) for element in elements])
    K = sparse.coo_matrix((k_data, (rows, cols)), shape=(num_dofs, num_dofs)).tocsr()

    if not include_mass:
        return K, empty

    m_data = np.concatenate([element.m_global.reshape(-1) for element in elements])
    M = sparse.coo_matrix((m_data, (rows, cols)), shape=(num_dofs, num_dofs)).tocsr()

    mass_rows = [node.dofs[i] for node in structure.nodes if node.mass > 0 for i in range(3)]
    if mass_rows:
        mass_vals = [node.mass for node in structure.nodes if node.mass > 0 for _ in range(3)]
        M = M + sparse.coo_matrix(
            (mass_vals, (mass_rows, mass_rows)), shape=(num_dofs, num_dofs)
        ).tocsr()

    return K, M


def apply_constraint_penalties(K, structure: Structure):
    """Devuelve K con penalización rígida (1e12x) añadida en la diagonal de los DOF restringidos."""
    num_dofs = structure.num_dofs

    if K.nnz > 0:
        penalty_value = 1e12 * np.max(np.abs(K.data))
    else:
        penalty_value = 1e12
    if penalty_value == 0:
        penalty_value = 1e12

    node_by_id = {node.id: node for node in structure.nodes}
    penalty_dofs = []
    for node_id, local_dofs_constrained in structure.constraints.items():
        node_obj = node_by_id.get(node_id)
        if node_obj is None:
            raise ValueError(f"Node with ID {node_id} not found in structure.")
        for dof_local_index in local_dofs_constrained:
            penalty_dofs.append(node_obj.dofs[dof_local_index])

    if not penalty_dofs:
        return K.tocsr()

    penalty_matrix = sparse.coo_matrix(
        (np.full(len(penalty_dofs), penalty_value), (penalty_dofs, penalty_dofs)),
        shape=(num_dofs, num_dofs),
    ).tocsr()
    return (K + penalty_matrix).tocsr()


def assemble_global_matrices(structure: Structure, include_mass=True, apply_constraints=True):
    K, M = _assemble_element_matrices(structure, include_mass)
    if apply_constraints:
        K = apply_constraint_penalties(K, structure)
    return K.tocsr(), M.tocsr()
