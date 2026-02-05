import numpy as np
from scipy.sparse import issparse
from .analisis_modal_3d.structures.structure import Structure
from .analisis_modal_3d.structures.node import Node as CoreNode
from .analisis_modal_3d.structures.element import Element as CoreElement
from .analisis_modal_3d.analysis.assembler import assemble_global_matrices
from .analisis_modal_3d.analysis.static import static_analysis
from .analisis_modal_3d.analysis.modal import modal_analysis
from .analisis_modal_3d.analysis.frequency_response import direct_frequency_response
from .analisis_modal_3d.analysis.damping import rayleigh_damping_matrix


def _sanitize_for_json(data):
    """
    Recursively replaces NaN and Infinity values with None to ensure JSON compliance.
    """
    if isinstance(data, dict):
        return {k: _sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_sanitize_for_json(i) for i in data]
    elif isinstance(data, (float, np.floating)):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    elif isinstance(data, (int, np.integer)):
        return int(data)
    elif isinstance(data, np.ndarray):
        return _sanitize_for_json(data.tolist())
    return data

def _to_list(data):
    if issparse(data):
        return data.toarray().tolist()
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data

def _build_structure(structure_data: dict, mass_type: str = 'lumped') -> tuple[Structure, dict, dict, dict, list]:
    """Crea un objeto Structure fusionando nodos duplicados y filtrando aislados."""
    structure = Structure()
    
    consistent = (mass_type == 'consistent')

    nodes = structure_data.get('nodes', [])
    materials = structure_data.get('materials', [])
    sections = structure_data.get('sections', [])
    elements = structure_data.get('elements', [])
    restraints = structure_data.get('restraints', {})

    print(f"DEBUG: _build_structure | Recibidos {len(nodes)} nodos, {len(elements)} elementos.")
    
    invalid_data_warnings = []

    if not nodes or not elements:
        raise ValueError("La estructura debe tener al menos un nodo y un elemento.")

    # 1. Fusión de Nodos por Proximidad (1e-6)
    # original_id -> representative_id
    id_map = {}
    representative_nodes = [] # Lista de (coords_tuple, id_representativo)
    
    for n_data in nodes:
        coords = tuple(n_data['coords'])
        nid = n_data['id']
        found_id = None
        for r_coords, r_id in representative_nodes:
            if np.linalg.norm(np.array(coords) - np.array(r_coords)) < 1e-6:
                found_id = r_id
                break
        
        if found_id is not None:
            id_map[nid] = found_id
        else:
            # Si el ID ya existe con otras coordenadas, generamos uno nuevo para evitar conflictos
            if any(nid == r_id for _, r_id in representative_nodes):
                new_id = max([r_id for _, r_id in representative_nodes] + list(id_map.values()) + [nid]) + 1
                print(f"DEBUG: ID de nodo duplicado {nid} detectado. Reasignando a {new_id}")
                id_map[nid] = new_id
                representative_nodes.append((coords, new_id))
            else:
                id_map[nid] = nid
                representative_nodes.append((coords, nid))

    # 2. Identificar Nodos USADOS (basado en representativos)
    used_rep_node_ids = set()
    valid_elements_data = []
    
    for el in elements:
        # Solo elementos con nodos distintos tras el mapeo
        rid1 = id_map[el['node_ids'][0]]
        rid2 = id_map[el['node_ids'][1]]
        if rid1 != rid2:
            used_rep_node_ids.add(rid1)
            used_rep_node_ids.add(rid2)
            valid_elements_data.append(el)

    # Añadir nodos con apoyos (solo si el nodo existe en la lista de nodos recibida)
    for node_id_str in restraints.keys():
        try:
            nid = int(node_id_str)
            if nid in id_map:
                used_rep_node_ids.add(id_map[nid])
            else:
                warning = f"Restricción en nodo {nid} ignorada (nodo no existe)"
                print(f"WARNING: {warning}")
                invalid_data_warnings.append(warning)
        except ValueError:
            continue
        
    # Añadir nodos con cargas (solo si el nodo existe)
    for load in structure_data.get('loads', []):
        nid = load.get('node_id')
        if nid is not None:
            if nid in id_map:
                used_rep_node_ids.add(id_map[nid])
            else:
                warning = f"Carga en nodo {nid} ignorada (nodo no existe)"
                print(f"WARNING: {warning}")
                invalid_data_warnings.append(warning)

    # 3. Crear Mapa de Materiales y Secciones
    for m in materials:
        if m.get("G") is None:
            m["G"] = m["E"] / (2 * (1 + m["nu"]))
    
    # Normalizar secciones: aceptar tanto 'A' como 'area'
    for s in sections:
        if "A" in s and "area" not in s:
            s["area"] = s["A"]
        elif "area" not in s and "A" not in s:
            raise ValueError(f"Section {s.get('id')} must have either 'A' or 'area' property")
    
    materials_map = {m['id']: m for m in materials}
    sections_map = {s['id']: s for s in sections}

    # 4. Añadir Nodos al Core (solo los usados)
    nodes_map_core = {}
    
    # Mapa rápido para buscar masa original por ID
    original_node_masses = {n['id']: n.get('mass', 0.0) for n in nodes}
    
    for coords, rep_id in representative_nodes:
        if rep_id in used_rep_node_ids:
            # Obtener masa (si varios nodos se fusionan, sumamos sus masas?)
            # Por simplicidad, tomamos la masa del nodo representativo o la mayor.
            # Aquí buscaremos todos los IDs que mapean a este rep_id y sumaremos.
            total_mass = 0.0
            for orig_id, mapping_rep_id in id_map.items():
                if mapping_rep_id == rep_id:
                    total_mass += original_node_masses.get(orig_id, 0.0)
            
            node_core = CoreNode(rep_id, *coords)
            node_core.mass = total_mass
            structure.add_node(node_core)
            nodes_map_core[rep_id] = node_core
            print(f"DEBUG: Nodo representativo {rep_id} añadido con masa {total_mass}.")

    # 5. Añadir Elementos
    for el in valid_elements_data:
        node1 = nodes_map_core[id_map[el['node_ids'][0]]]
        node2 = nodes_map_core[id_map[el['node_ids'][1]]]
        
        if el['material_id'] not in materials_map:
            warning = f"Elemento {el['id']} hace referencia a un material inexistente {el['material_id']}"
            print(f"WARNING: {warning}")
            invalid_data_warnings.append(warning)
            continue

        if el['section_id'] not in sections_map:
            warning = f"Elemento {el['id']} hace referencia a una sección inexistente {el['section_id']}"
            print(f"WARNING: {warning}")
            invalid_data_warnings.append(warning)
            continue

        mat = materials_map[el['material_id']]
        sec = sections_map[el['section_id']]

        element = CoreElement(
            element_id=el['id'],
            node1=node1,
            node2=node2,
            material=mat,
            section=sec,
            consistent=consistent
        )
        structure.add_element(element)

    # 6. Consolidar y Añadir Restricciones
    consolidated_restraints = {}
    for node_id_str, dofs in restraints.items():
        node_id = int(node_id_str)
        if node_id not in id_map:
            warning = f"Restricción en nodo {node_id} ignorada (nodo no existe)"
            print(f"WARNING: {warning}")
            if warning not in invalid_data_warnings:
                invalid_data_warnings.append(warning)
            continue
        rep_id = id_map[node_id]
        if rep_id not in consolidated_restraints:
            consolidated_restraints[rep_id] = set()
        consolidated_restraints[rep_id].update(dofs)

    for rep_id, dofs_set in consolidated_restraints.items():
        if rep_id in nodes_map_core:
            structure.add_constraint(nodes_map_core[rep_id], list(dofs_set))

    return structure, nodes_map_core, materials_map, id_map, invalid_data_warnings


def run_static_analysis(structure_data: dict) -> dict:
    try:
        print("Iniciando el proceso de analisis estatico en el adaptador...")
        print(f"DEBUG: Estructura recibida - nodes: {len(structure_data.get('nodes', []))}, elements: {len(structure_data.get('elements', []))}, loads: {len(structure_data.get('loads', []))}, restraints: {len(structure_data.get('restraints', {}))}")
        
        mass_type = structure_data.get('settings', {}).get('mass_type', 'lumped')
        structure, nodes_map_core, _, id_map, warnings = _build_structure(structure_data, mass_type=mass_type)
        loads = structure_data.get('loads', [])
        
        if warnings:
            print(f"\n⚠️  ADVERTENCIAS DE DATOS INVALIDOS:")
            for warning in warnings:
                print(f"  - {warning}")
            print()

        print("DEBUG: Comenzando ensamblado de matrices...")
        K_pure, _ = assemble_global_matrices(structure, include_mass=False, apply_constraints=False)
        print("DEBUG: Matriz K_pure ensamblada")
        
        K_penalized, _ = assemble_global_matrices(structure, include_mass=False, apply_constraints=True)
        print("DEBUG: Matriz K_penalized ensamblada")
        
        F_global = np.zeros(structure.num_dofs, dtype=np.float64)
        dof_map = {"fx": 0, "fy": 1, "fz": 2, "mx": 3, "my": 4, "mz": 5}

        print(f"DEBUG: Procesando {len(loads)} cargas...")
        for load_data in loads:
            original_node_id = load_data.get('node_id')
            if original_node_id is None:
                continue

            # Mapear al representante
            rep_id = id_map.get(original_node_id)
            if rep_id is None or rep_id not in nodes_map_core:
                continue

            node_core = nodes_map_core[rep_id]
            print(f"DEBUG: Nodo rep {rep_id} (orig {original_node_id}) coords: {node_core.coords}")

            # Iterar sobre los tipos de fuerza y aplicarlos si existen en la carga
            for force_type, dof_local_index in dof_map.items():
                force_value = load_data.get(force_type, 0.0)
                if force_value != 0.0:
                    global_dof_index = node_core.dofs[dof_local_index]
                    F_global[global_dof_index] += force_value
                    print(f"DEBUG: Aplicada carga {force_type}={force_value} en nodo original {original_node_id} (rep: {rep_id}, DOF global: {global_dof_index})")

        # Log del vector de fuerzas solo para componentes distintas de cero
        nonzero_f = np.where(F_global != 0)[0]
        for idx in nonzero_f:
            print(f"DEBUG: F_global[{idx}] = {F_global[idx]}")

        print(f"DEBUG: K_global shape: {K_pure.shape}")
        print(f"DEBUG: K_penalized shape: {K_penalized.shape}")

        # --- Análisis Estático ---
        displacements = static_analysis(K_penalized, F_global)

        if displacements is None:
            return {"error": "El análisis estático falló. La estructura puede ser inestable."}

        # --- Post-procesar Resultados ---
        results = {
            "displacements": {},
            "element_forces": {},
            "reactions": {},
            "stresses": {}, # Esfuerzos de Von Mises por elemento
        }

        # Reacciones: R = K_pure * u - F_ext
        # Esta fórmula nos da la fuerza que el apoyo ejerce sobre la estructura.
        reactions_vector = K_pure @ displacements - F_global
        print(f"DEBUG: Max reacción detectada (K_pure @ u - F): {np.max(np.abs(reactions_vector))}")

        # Verificación de equilibrio global
        total_reactions = np.sum(reactions_vector.reshape(-1, 6), axis=0)
        total_loads = np.sum(F_global.reshape(-1, 6), axis=0)
        equilibrium_error = np.abs(total_reactions + total_loads)
        max_error = np.max(equilibrium_error)
        if max_error > 1e-3:
            print(f"⚠️  WARNING: Equilibrium error = {max_error:.2e} (expected ~0)")
        else:
            print(f"✓ Equilibrium verified: max error = {max_error:.2e}")

        for original_node_id in id_map.keys():
            rep_id = id_map[original_node_id]
            if rep_id in nodes_map_core:
                node_core = nodes_map_core[rep_id]
                node_displacements = _to_list(displacements[node_core.dofs])
                results["displacements"][original_node_id] = node_displacements

                if node_core.is_constrained:
                    node_reactions = _to_list(reactions_vector[node_core.dofs])
                    print(f"DEBUG: Nodo original {original_node_id} (rep {rep_id}) es restringido. Reacciones: {node_reactions}")
                    # Usamos un umbral para no mostrar ruido numérico
                    if np.any(np.abs(node_reactions) > 1e-6):
                        results["reactions"][original_node_id] = node_reactions
                    else:
                        print(f"DEBUG: Reacciones despreciables en nodo {original_node_id}")

        for element in structure.elements:
            # Obtener desplazamientos locales del elemento
            u_el = displacements[element.dof_indices]
            
            # Fuerzas internas
            internal_forces = element.get_internal_forces(displacements)
            results["element_forces"][element.element_id] = internal_forces
            
            # Esfuerzos de Von Mises
            results["stresses"][element.element_id] = element.get_stresses(u_el)

        results["stresses"] = results["stresses"] # Ensure it's explicitly included if needed, but it's already in the dict

        print("Proceso de analisis estatico en el adaptador finalizado.")
        return _sanitize_for_json(results)

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"ERROR CAPTURADO EN run_static_analysis:")
        print(f"Tipo: {type(e).__name__}")
        print(f"Mensaje: {str(e)}")
        print(f"Traceback completo:\n{error_traceback}")
        return {"error": f"Error fatal en el nucleo de analisis: {str(e)}"}

def run_modal_analysis(structure_data: dict, num_modes: int) -> dict:
    try:
        mass_type = structure_data.get('settings', {}).get('mass_type', 'lumped')
        structure, nodes_map_core, _, id_map, warnings = _build_structure(structure_data, mass_type=mass_type)
        
        if warnings:
            print(f"\n⚠️  ADVERTENCIAS DE DATOS INVALIDOS:")
            for warning in warnings:
                print(f"  - {warning}")
            print()

        # Para análisis modal, no aplicamos penalizaciones en el ensamblado
        # porque modal_analysis realiza la reducción de GDL internamente.
        # Esto asegura que el cálculo de participación de masa sea correcto.
        K_global, M_global = assemble_global_matrices(structure, include_mass=True, apply_constraints=False)

        if K_global is None or M_global is None:
            return {"error": "Error al ensamblar las matrices globales."}

        frequencies, modes, mass_participation = modal_analysis(K_global, M_global, structure, num_modes=num_modes, debug=False)

        results = {
            "frequencies": _to_list(frequencies),
            "mode_shapes": {},
            "mass_participation": _to_list(mass_participation)
        }
        for i in range(len(frequencies)):
            mode_shape = modes[:, i]
            for original_node_id, rep_id in id_map.items():
                if rep_id in nodes_map_core:
                    node_core = nodes_map_core[rep_id]
                    node_mode_shape = _to_list(mode_shape[node_core.dofs])
                    if original_node_id not in results["mode_shapes"]:
                        results["mode_shapes"][original_node_id] = []
                    results["mode_shapes"][original_node_id].append(node_mode_shape)
        return _sanitize_for_json(results)

    except Exception as e:

        return {"error": f"Error fatal en el nucleo de analisis modal: {str(e)}"}

def run_harmonic_analysis(request_data: dict) -> dict:
    try:
        # Extraer parámetros del request
        structure_data = request_data.get('structure', {})
        freq_start = float(request_data.get('freq_start', 0.1))
        freq_end = float(request_data.get('freq_end', 100.0))
        num_points = int(request_data.get('num_points', 100))
        damping_ratio = float(request_data.get('damping_ratio', 0.05))
        is_unbalanced = bool(request_data.get('is_unbalanced', False))
        unbalanced_me = float(request_data.get('unbalanced_me', 0.0))
        mass_type = structure_data.get('settings', {}).get('mass_type', 'lumped')

        structure, nodes_map_core, materials_map, id_map, warnings = _build_structure(structure_data, mass_type=mass_type)
        loads = structure_data.get('loads', [])
        
        if warnings:
            print(f"\n⚠️  ADVERTENCIAS DE DATOS INVALIDOS:")
            for warning in warnings:
                print(f"  - {warning}")
            print()

        K_global, M_global = assemble_global_matrices(structure, include_mass=True)
        
        if K_global is None or M_global is None:
            return {"error": "Error al ensamblar las matrices globales."}

        # Calcular amortiguamiento de Rayleigh simplificado
        # Obtenemos la primera frecuencia para calibrar beta (stiffness proportional)
        freqs, _, _ = modal_analysis(K_global, M_global, structure, num_modes=1, debug=False)
        if len(freqs) > 0 and freqs[0] > 0:
            omega1 = 2 * np.pi * freqs[0]
            beta = 2 * damping_ratio / omega1
            alpha = 0.0
        else:
            # Fallback si no hay frecuencias (estructura libre?)
            beta = 0.001
            alpha = 0.0
        
        C_global = rayleigh_damping_matrix(M_global, K_global, alpha, beta)

        # Preparar vector de fuerza
        F_amplitude = np.zeros(structure.num_dofs)
        dof_map = {"fx": 0, "fy": 1, "fz": 2, "mx": 3, "my": 4, "mz": 5}
        for load_data in loads:
            original_node_id = load_data.get('node_id')
            rep_id = id_map.get(original_node_id)
            if rep_id and rep_id in nodes_map_core:
                node_core = nodes_map_core[rep_id]
                for force_type, dof_idx in dof_map.items():
                    F_amplitude[node_core.dofs[dof_idx]] += load_data.get(force_type, 0.0)

        freq_range = np.linspace(freq_start, freq_end, num_points)

        complex_responses = direct_frequency_response(
            K_global, M_global, C_global, F_amplitude, freq_range,
            is_unbalanced_force=is_unbalanced,
            unbalanced_mass_product=unbalanced_me
        )

        # Amplitudes reales
        amplitudes = np.abs(complex_responses)
        
        results = {
            "frequencies_sweep": _to_list(freq_range),
            "response_amplitudes": {}
        }

        for original_node_id, rep_id in id_map.items():
            if rep_id in nodes_map_core:
                node_core = nodes_map_core[rep_id]
                # Magnitud de traslación (norma de ux, uy, uz)
                node_dofs = node_core.dofs[:3]
                node_amps = np.sqrt(np.sum(amplitudes[:, node_dofs]**2, axis=1))
                results["response_amplitudes"][original_node_id] = _to_list(node_amps)

        return _sanitize_for_json(results)

    except Exception as e:
        return {"error": f"Error fatal en el nucleo de analisis armonico: {str(e)}"}
