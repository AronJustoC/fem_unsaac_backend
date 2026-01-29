import numpy as np
from scipy import sparse

from ..structures.structure import Structure


def assemble_global_matrices(structure: Structure, include_mass=True, apply_constraints=True):
    num_dofs = structure.num_dofs
    K = sparse.lil_matrix((num_dofs, num_dofs), dtype=np.float64)
    M = sparse.lil_matrix((num_dofs, num_dofs), dtype=np.float64)

    for element in structure.elements:
        k_global = element.k_global
        m_global = element.m_global
        dof_indices = []
        for node in element.nodes:
            dof_indices.extend(node.dofs)

        for i, dof_i in enumerate(dof_indices):
            for j, dof_j in enumerate(dof_indices):
                K[dof_i, dof_j] += k_global[i, j]
                M[dof_i, dof_j] += m_global[i, j]

    for node in structure.nodes:
        if node.mass > 0:
            for i in range(3):
                dof_index = node.dofs[i]
                M[dof_index, dof_index] += node.mass

    if apply_constraints:
        if K.nnz > 0:
            K_csr_temp = K.tocsr()
            penalty_value = 1e12 * np.max(np.abs(K_csr_temp.data))
        else:
            penalty_value = 1e12
        if penalty_value == 0:
            penalty_value = 1e12

        for node_id, local_dofs_constrained in structure.constraints.items():
            node_obj = next((n for n in structure.nodes if n.id == node_id), None)
            if node_obj is None:
                raise ValueError(f"Node with ID {node_id} not found in structure.")
            for dof_local_index in local_dofs_constrained:
                dof_global_index = node_obj.dofs[dof_local_index]
                K[dof_global_index, dof_global_index] += penalty_value
                # M[dof_global_index, dof_global_index] += penalty_value  # No penalizar la masa, rompe la participación de masa modal

    return K.tocsr(), M.tocsr()

