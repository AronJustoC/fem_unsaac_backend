import numpy as np
from scipy.sparse import issparse
from scipy import sparse
from .analisis_modal_3d.structures.structure import Structure
from .analisis_modal_3d.structures.node import Node as CoreNode
from .analisis_modal_3d.structures.element import Element as CoreElement
from .analisis_modal_3d.analysis.assembler import assemble_global_matrices, apply_constraint_penalties
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


def _is_massless_link(element) -> bool:
    """Detecta elementos de enlace rigido sin masa (p.ej. MOTOR_LINK: rho=0, E muy
    alto para inyectar una excitacion sin agregar flexibilidad/masa propia).

    Su fuerza interna es real (la usa el solver), pero su "esfuerzo" reportado es
    un artefacto de modelado sin significado fisico (seccion/rigidez ficticias) y
    puede ser ordenes de magnitud mayor que el resto de la estructura, arruinando
    la escala de color de todo el grafico. Se excluye de esfuerzos reportados; la
    carga que transporta ya queda reflejada en los elementos reales conectados al
    mismo nodo via el propio metodo de rigidez, sin necesidad de "transferirla" a mano.
    """
    return float(element.material.get('rho', 0.0) or 0.0) <= 0.0


def _compute_element_harmonic_stress_amplitudes(element, u_global_complex: np.ndarray) -> list[dict]:
    """Devuelve amplitudes alternantes por extremo del elemento: normal, cortante,
    los dos principales + cortante maximo (Mohr, mismo criterio que get_stresses)
    y Von Mises como resumen.

    La respuesta armónica U(ω) es compleja. Para reporte nodal usamos una envolvente
    conservadora compatible con get_stresses(): se recuperan fuerzas internas locales
    complejas, se toman amplitudes por componente y se combina axial + flexión +
    cortante/torsión para obtener una amplitud escalar de esfuerzo alternante.
    """
    u_local = element.T @ u_global_complex
    f_local = element.k_local @ u_local

    E = element.material['E']
    A = element.section.get('area', 0.001)
    Iz = element.section.get('Iz', 1e-8)
    Iy = element.section.get('Iy', 1e-8)
    J = element.section.get('J', 1e-8)
    h = element.section.get('height') or 0.1
    b = element.section.get('width') or 0.1

    if not A or A <= 0:
        A = 0.001
    if not Iz or Iz <= 0:
        Iz = 1e-8
    if not Iy or Iy <= 0:
        Iy = 1e-8
    if not J or J <= 0:
        J = 1e-8
    y_max = h / 2.0
    z_max = b / 2.0
    kappa = 5.0 / 6.0
    radius = max(y_max, z_max)

    stresses = []
    for i in [0, 6]:
        sgn = 1 if i == 0 else -1
        Fx = -f_local[i] * sgn
        Fy = f_local[i + 1] * sgn
        Fz = f_local[i + 2] * sgn
        Mx = -f_local[i + 3] * sgn
        My = f_local[i + 4] * sgn
        Mz = f_local[i + 5] * sgn

        sigma_axial = Fx / A
        # Máxima amplitud normal en las cuatro fibras extremas del marco local.
        sigma_candidates = [
            sigma_axial + sy * My * z_max / Iy + sz * Mz * y_max / Iz
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
        sigma_amp = max(float(np.abs(candidate)) for candidate in sigma_candidates)

        tau_torsion_amp = float(np.abs(Mx) * radius / J)
        tau_shear_amp = kappa * float(np.hypot(np.abs(Fy) / A, np.abs(Fz) / A))
        tau_amp = tau_torsion_amp + tau_shear_amp
        sigma_vm_amp = float(np.sqrt(sigma_amp**2 + 3.0 * tau_amp**2))

        # Mismo circulo de Mohr que get_stresses(), aplicado a la envolvente
        # (sigma_amp, tau_amp) en vez del valor instantaneo — no es exacto para un
        # complejo (Von Mises no es lineal), pero es la misma aproximación
        # conservadora que ya se usa para sigma_vm_amp.
        center = sigma_amp / 2.0
        mohr_radius = float(np.sqrt(center**2 + tau_amp**2))

        stresses.append({
            "sigma_normal": sigma_amp,
            "tau_shear": tau_amp,
            "sigma_1": center + mohr_radius,
            "sigma_2": center - mohr_radius,
            "tau_max": mohr_radius,
            "sigma_vm": sigma_vm_amp,
        })

    return stresses

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
        if m.get("E") is None or m.get("nu") is None or m.get("rho") is None:
            raise ValueError(f"Material {m.get('id')} tiene propiedades requeridas nulas: E={m.get('E')}, nu={m.get('nu')}, rho={m.get('rho')}")
        if m.get("G") is None:
            m["G"] = m["E"] / (2 * (1 + m["nu"]))
    
    _SECTION_REQUIRED_KEYS = ("area", "Iy", "Iz", "J")
    for s in sections:
        if "A" in s and "area" not in s:
            s["area"] = s["A"]
        elif "area" not in s and "A" not in s:
            raise ValueError(f"Section {s.get('id')} must have either 'A' or 'area' property")
        missing = [k for k in _SECTION_REQUIRED_KEYS if s.get(k) is None]
        if missing:
            raise ValueError(f"Section {s.get('id')} tiene propiedades requeridas nulas: {missing}. Datos: {s}")
        if s.get("height") is None:
            s["height"] = 0.1
        if s.get("width") is None:
            s["width"] = 0.1
    
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
        # Penalización reutiliza K_pure en vez de reensamblar todos los elementos de nuevo.
        K_penalized = apply_constraint_penalties(K_pure, structure)
        print(f"DEBUG: Matrices ensambladas - K_global shape: {K_pure.shape}")

        F_global = np.zeros(structure.num_dofs, dtype=np.float64)
        dof_map = {"fx": 0, "fy": 1, "fz": 2, "mx": 3, "my": 4, "mz": 5}

        for load_data in loads:
            original_node_id = load_data.get('node_id')
            if original_node_id is None:
                continue

            # Mapear al representante
            rep_id = id_map.get(original_node_id)
            if rep_id is None or rep_id not in nodes_map_core:
                continue

            node_core = nodes_map_core[rep_id]

            # Iterar sobre los tipos de fuerza y aplicarlos si existen en la carga
            for force_type, dof_local_index in dof_map.items():
                force_value = load_data.get(force_type, 0.0)
                if force_value != 0.0:
                    global_dof_index = node_core.dofs[dof_local_index]
                    F_global[global_dof_index] += force_value

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
            
            # Esfuerzos por extremo. Los enlaces rigidos sin masa (MOTOR_LINK) no
            # reportan esfuerzo propio: es un artefacto de modelado, no un valor real.
            # Lista aplanada (el schema es list[float]): orden fijo por extremo i,j:
            # [vm_i, vm_j, normal_i, normal_j, shear_i, shear_j, s1_i, s1_j, s2_i, s2_j, taumax_i, taumax_j]
            if _is_massless_link(element):
                results["stresses"][element.element_id] = [0.0] * 12
            else:
                end_i, end_j = element.get_stresses(u_el)
                results["stresses"][element.element_id] = [
                    end_i["sigma_vm"], end_j["sigma_vm"],
                    end_i["sigma_normal"], end_j["sigma_normal"],
                    end_i["tau_shear"], end_j["tau_shear"],
                    end_i["sigma_1"], end_j["sigma_1"],
                    end_i["sigma_2"], end_j["sigma_2"],
                    end_i["tau_max"], end_j["tau_max"],
                ]

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


def get_element_matrices(structure_data: dict, element_id: int) -> dict:
    """Devuelve matrices y contexto didáctico de un único elemento.

    Se calcula bajo demanda para evitar incluir cientos de matrices 12x12 en la
    respuesta modal inicial. Los metadatos `visual_*` no intervienen en K o M.
    """
    try:
        mass_type = structure_data.get('settings', {}).get('mass_type', 'lumped')
        structure, _, _, _, warnings = _build_structure(structure_data, mass_type=mass_type)
        element = next(
            (candidate for candidate in structure.elements if candidate.element_id == element_id),
            None,
        )
        if element is None:
            return {"error": f"El elemento {element_id} no existe o fue descartado por longitud cero."}

        input_element = next(
            (candidate for candidate in structure_data.get('elements', []) if candidate.get('id') == element_id),
            {},
        )
        dof_labels = [
            'ux₁', 'uy₁', 'uz₁', 'rx₁', 'ry₁', 'rz₁',
            'ux₂', 'uy₂', 'uz₂', 'rx₂', 'ry₂', 'rz₂',
        ]
        section = element.section
        material = element.material
        total_mass = material['rho'] * section['area'] * element.L

        result = {
            "element_id": element.element_id,
            "node_ids": list(input_element.get('node_ids') or [element.nodes[0].id, element.nodes[1].id]),
            "length_m": element.L,
            "mass_type": mass_type,
            "dof_labels": dof_labels,
            "properties": {
                "material_name": material.get('name') or f"Material {input_element.get('material_id', '')}",
                "section_name": section.get('name') or f"Seccion {input_element.get('section_id', '')}",
                "E_pa": material['E'],
                "G_pa": material['G'],
                "rho_kg_m3": material['rho'],
                "area_m2": section['area'],
                "Iy_m4": section['Iy'],
                "Iz_m4": section['Iz'],
                "J_m4": section['J'],
                "element_mass_kg": total_mass,
            },
            "local_axes": element.T[:3, :3],
            "matrices": {
                "stiffness": {
                    "local": element.k_local,
                    "global": element.k_global,
                },
                "mass": {
                    "local": element.m_local,
                    "global": element.m_global,
                },
            },
            "checks": {
                "stiffness_symmetry_error": float(np.max(np.abs(element.k_local - element.k_local.T))),
                "mass_symmetry_error": float(np.max(np.abs(element.m_local - element.m_local.T))),
            },
            "warnings": warnings,
        }
        return _sanitize_for_json(result)
    except Exception as exc:
        return {"error": f"No se pudieron obtener las matrices del elemento: {str(exc)}"}


_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(5)


def _bd_rows_for_element(element) -> dict:
    """Deformacion-desplazamiento [B] y constitutiva [D] explicitas para el
    elemento de viga, evaluadas contra k_local para probar que son la misma
    formulacion (k_local = integral de B^T D B, solo que ya resuelta a mano
    para viga en vez de integrada numericamente como en un elemento continuo).
    """
    L = element.L
    E = element.material['E']
    G = element.material['G']
    A = element.section['area']
    Iy = element.section['Iy']
    Iz = element.section['Iz']
    J = element.section['J']

    def b_axial():
        return np.array([-1 / L, 0, 0, 0, 0, 0, 1 / L, 0, 0, 0, 0, 0])

    def b_torsion():
        return np.array([0, 0, 0, -1 / L, 0, 0, 0, 0, 0, 1 / L, 0, 0])

    def hermite_curv(x):
        return (
            -6 / L ** 2 + 12 * x / L ** 3,
            -4 / L + 6 * x / L ** 2,
            6 / L ** 2 - 12 * x / L ** 3,
            -2 / L + 6 * x / L ** 2,
        )

    def b_bend_z(x):
        h1, h2, h3, h4 = hermite_curv(x)
        row = np.zeros(12)
        row[1], row[5], row[7], row[11] = h1, h2, h3, h4
        return row

    def b_bend_y(x):
        h1, h2, h3, h4 = hermite_curv(x)
        row = np.zeros(12)
        row[2], row[4], row[8], row[10] = h1, -h2, h3, -h4
        return row

    xs = 0.5 * L * (_GAUSS_X + 1)
    ws = 0.5 * L * _GAUSS_W

    k_check = (
        E * A * np.outer(b_axial(), b_axial()) * L
        + G * J * np.outer(b_torsion(), b_torsion()) * L
        + sum(E * Iz * np.outer(b_bend_z(x), b_bend_z(x)) * w for x, w in zip(xs, ws))
        + sum(E * Iy * np.outer(b_bend_y(x), b_bend_y(x)) * w for x, w in zip(xs, ws))
    )

    return {
        "D": {"E_axial_bending": E, "G_shear_torsion": G},
        "B": {
            "axial": b_axial().tolist(),
            "torsion": b_torsion().tolist(),
            "bend_z_end_i": b_bend_z(0.0).tolist(),
            "bend_z_end_j": b_bend_z(L).tolist(),
            "bend_y_end_i": b_bend_y(0.0).tolist(),
            "bend_y_end_j": b_bend_y(L).tolist(),
        },
        "verification": {
            "k_check": k_check.tolist(),
            "k_local": element.k_local.tolist(),
            "max_abs_diff": float(np.max(np.abs(k_check - element.k_local))),
        },
    }


def get_element_bd_matrices(structure_data: dict, element_id: int) -> dict:
    """[B] y [D] explicitas para un elemento, mas la verificacion de que
    integradas reproducen k_local (misma logica de rigidez, sin instanciarlas
    como arrays por separado en el resto del codigo).
    """
    try:
        mass_type = structure_data.get('settings', {}).get('mass_type', 'lumped')
        structure, _, _, _, _ = _build_structure(structure_data, mass_type=mass_type)
        element = next(
            (candidate for candidate in structure.elements if candidate.element_id == element_id),
            None,
        )
        if element is None:
            return {"error": f"El elemento {element_id} no existe o fue descartado por longitud cero."}

        dof_labels = [
            'ux₁', 'uy₁', 'uz₁', 'rx₁', 'ry₁', 'rz₁',
            'ux₂', 'uy₂', 'uz₂', 'rx₂', 'ry₂', 'rz₂',
        ]
        result = {
            "element_id": element.element_id,
            "node_ids": [element.nodes[0].id, element.nodes[1].id],
            "length_m": element.L,
            "dof_labels": dof_labels,
            **_bd_rows_for_element(element),
        }
        return _sanitize_for_json(result)
    except Exception as exc:
        return {"error": f"No se pudieron obtener las matrices B/D del elemento: {str(exc)}"}


GLOBAL_DOF_NAMES = ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')


def build_global_matrix_bundle(structure_data: dict) -> dict:
    """Ensambla K/M una sola vez y conserva sus vistas completa y modal libre.

    El bundle mantiene matrices CSR dentro de la caché del proceso; nunca se
    serializa directamente. Esto permite consultar ventanas numéricas sin
    transferir una matriz densa de hasta millones de entradas al navegador.
    """
    mass_type = structure_data.get('settings', {}).get('mass_type', 'lumped')
    structure, _, _, _, warnings = _build_structure(structure_data, mass_type=mass_type)
    K_global, M_global = assemble_global_matrices(
        structure,
        include_mass=True,
        apply_constraints=False,
    )
    K_global = K_global.tocsr()
    M_global = M_global.tocsr()

    constrained_dofs = np.asarray(structure.get_constrained_dofs(), dtype=int)
    all_dofs = np.arange(structure.num_dofs, dtype=int)
    free_dofs = np.setdiff1d(all_dofs, constrained_dofs)
    K_free = K_global[free_dofs, :][:, free_dofs].tocsr()
    M_free = M_global[free_dofs, :][:, free_dofs].tocsr()

    # modal_analysis aplica exactamente esta regularización si algún GDL libre
    # tiene masa prácticamente nula. La vista Mff debe coincidir con la ecuación
    # que realmente resuelve el eigensolver.
    mass_regularized = bool(M_free.shape[0] and np.any(M_free.diagonal() < 1e-15))
    regularization_value = 1e-9 if mass_regularized else 0.0
    if mass_regularized:
        M_free = (M_free + sparse.eye(M_free.shape[0], format='csr') * regularization_value).tocsr()

    labels = [''] * structure.num_dofs
    for node in structure.nodes:
        for local_index, global_index in enumerate(node.dofs):
            labels[global_index] = f"N{node.id}·{GLOBAL_DOF_NAMES[local_index]}"

    return {
        'structure': structure,
        'matrices': {
            ('stiffness', 'full'): K_global,
            ('mass', 'full'): M_global,
            ('stiffness', 'free'): K_free,
            ('mass', 'free'): M_free,
        },
        'labels': labels,
        'full_dof_indices': all_dofs,
        'free_dof_indices': free_dofs,
        'constrained_dof_indices': constrained_dofs,
        'metadata': {
            'node_count': len(structure.nodes),
            'element_count': len(structure.elements),
            'total_dofs': structure.num_dofs,
            'free_dofs': int(free_dofs.size),
            'constrained_dofs': int(constrained_dofs.size),
            'mass_type': mass_type,
            'mass_regularized': mass_regularized,
            'regularization_value': regularization_value,
            'warnings': warnings,
        },
    }


def _sparse_matrix_stats(matrix: sparse.csr_matrix) -> dict:
    absolute_data = np.abs(matrix.data)
    difference = matrix - matrix.T
    symmetry_error = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
    diagonal = matrix.diagonal()
    return {
        'dimension': int(matrix.shape[0]),
        'nnz': int(matrix.nnz),
        'density': float(matrix.nnz / max(matrix.shape[0] * matrix.shape[1], 1)),
        'max_abs': float(np.max(absolute_data)) if absolute_data.size else 0.0,
        'min_nonzero_abs': float(np.min(absolute_data[absolute_data > 0])) if np.any(absolute_data > 0) else 0.0,
        'diagonal_min': float(np.min(diagonal)) if diagonal.size else 0.0,
        'diagonal_max': float(np.max(diagonal)) if diagonal.size else 0.0,
        'symmetry_error': symmetry_error,
    }


def _build_sparse_heatmap(matrix: sparse.csr_matrix, requested_bins: int) -> dict:
    dimension = matrix.shape[0]
    bins = max(1, min(int(requested_bins), 120, max(dimension, 1)))
    magnitudes = np.zeros((bins, bins), dtype=float)
    counts = np.zeros((bins, bins), dtype=np.int64)

    if matrix.nnz:
        coo = matrix.tocoo()
        row_bins = np.minimum((coo.row * bins) // max(dimension, 1), bins - 1)
        col_bins = np.minimum((coo.col * bins) // max(dimension, 1), bins - 1)
        np.maximum.at(magnitudes, (row_bins, col_bins), np.abs(coo.data))
        np.add.at(counts, (row_bins, col_bins), 1)

    return {
        'bins': bins,
        'values': magnitudes,
        'counts': counts,
    }


def _rayleigh_beta_from_first_mode(km_bundle: dict, damping_ratio: float) -> tuple[float, float, float | None]:
    """alpha, beta, f1_hz de Rayleigh calibrados con la 1ra frecuencia natural —
    mismo criterio que run_harmonic_analysis, asi la C que se inspecciona es la
    misma que realmente entra en Z(w) del barrido armonico.
    """
    K_full = km_bundle['matrices'][('stiffness', 'full')]
    M_full = km_bundle['matrices'][('mass', 'full')]
    structure = km_bundle['structure']
    freqs, _, _ = modal_analysis(K_full, M_full, structure, num_modes=1, debug=False)
    f1 = float(freqs[0]) if len(freqs) > 0 else None
    if f1:
        omega1 = 2 * np.pi * f1
        beta = 2 * damping_ratio / omega1
    else:
        beta = 0.001
    return 0.0, beta, f1


def get_global_matrix_view(
    bundle: dict,
    matrix_kind: str = 'stiffness',
    matrix_scope: str = 'full',
    damping_ratio: float = 0.02,
    row_start: int = 0,
    col_start: int = 0,
    window_size: int = 12,
    heatmap_bins: int = 64,
) -> dict:
    """Serializa el mapa completo y una ventana legible de K/M/C ensamblada."""
    if matrix_kind not in {'stiffness', 'mass', 'damping'}:
        return {'error': "matrix_kind debe ser 'stiffness', 'mass' o 'damping'."}
    if matrix_scope not in {'full', 'free'}:
        return {'error': "matrix_scope debe ser 'full' o 'free'."}

    rayleigh_info = None
    if matrix_kind == 'damping':
        K = bundle['matrices'][('stiffness', matrix_scope)]
        M = bundle['matrices'][('mass', matrix_scope)]
        alpha, beta, f1 = _rayleigh_beta_from_first_mode(bundle, damping_ratio)
        matrix = rayleigh_damping_matrix(M, K, alpha, beta).tocsr()
        rayleigh_info = {
            'damping_ratio': damping_ratio,
            'rayleigh_alpha': alpha,
            'rayleigh_beta': beta,
            'first_natural_frequency_hz': f1,
        }
    else:
        matrix = bundle['matrices'][(matrix_kind, matrix_scope)]
    dimension = matrix.shape[0]
    size = max(6, min(int(window_size), 36, max(dimension, 6)))
    size = min(size, dimension) if dimension else 0
    max_start = max(dimension - size, 0)
    row_start = max(0, min(int(row_start), max_start))
    col_start = max(0, min(int(col_start), max_start))
    row_end = row_start + size
    col_end = col_start + size

    dof_indices = (
        bundle['full_dof_indices']
        if matrix_scope == 'full'
        else bundle['free_dof_indices']
    )
    labels = bundle['labels']
    row_global_indices = dof_indices[row_start:row_end]
    col_global_indices = dof_indices[col_start:col_end]

    result = {
        'matrix': {
            'kind': matrix_kind,
            'scope': matrix_scope,
            **({'rayleigh': rayleigh_info} if rayleigh_info else {}),
            **_sparse_matrix_stats(matrix),
        },
        'metadata': bundle['metadata'],
        'heatmap': _build_sparse_heatmap(matrix, heatmap_bins),
        'window': {
            'row_start': row_start,
            'col_start': col_start,
            'size': size,
            'values': matrix[row_start:row_end, col_start:col_end].toarray(),
            'row_labels': [labels[int(index)] for index in row_global_indices],
            'col_labels': [labels[int(index)] for index in col_global_indices],
            'row_global_indices': row_global_indices,
            'col_global_indices': col_global_indices,
        },
    }
    return _sanitize_for_json(result)


def _complex_sparse_matrix_stats(matrix: sparse.csr_matrix) -> dict:
    """Como _sparse_matrix_stats, pero magnitud |z| en vez de comparación real/compleja.

    Z(ω) = K - ω²M + iωC es compleja SIMÉTRICA (Z^T = Z), no hermítica, porque K/M/C
    son reales simétricas por reciprocidad FEM. Por eso el chequeo usa transpuesta
    simple, no transpuesta conjugada.
    """
    absolute_data = np.abs(matrix.data)
    difference = matrix - matrix.T
    symmetry_error = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
    diagonal_abs = np.abs(matrix.diagonal())
    return {
        'dimension': int(matrix.shape[0]),
        'nnz': int(matrix.nnz),
        'density': float(matrix.nnz / max(matrix.shape[0] * matrix.shape[1], 1)),
        'max_abs': float(np.max(absolute_data)) if absolute_data.size else 0.0,
        'min_nonzero_abs': float(np.min(absolute_data[absolute_data > 0])) if np.any(absolute_data > 0) else 0.0,
        'diagonal_min_abs': float(np.min(diagonal_abs)) if diagonal_abs.size else 0.0,
        'diagonal_max_abs': float(np.max(diagonal_abs)) if diagonal_abs.size else 0.0,
        'symmetry_error': symmetry_error,
    }


def build_impedance_matrix_bundle(km_bundle: dict, damping_ratio: float = 0.05) -> dict:
    """Deriva C (Rayleigh) del bundle K/M ya ensamblado, para construir Z(ω) = K - ω²M + iωC bajo demanda.

    Reutiliza exactamente el mismo criterio de amortiguamiento que run_harmonic_analysis
    (beta calibrado con la primera frecuencia natural, alpha=0), así la matriz que se
    inspecciona aquí coincide con la que realmente resuelve el barrido armónico.
    """
    K_full = km_bundle['matrices'][('stiffness', 'full')]
    M_full = km_bundle['matrices'][('mass', 'full')]
    K_free = km_bundle['matrices'][('stiffness', 'free')]
    M_free = km_bundle['matrices'][('mass', 'free')]

    alpha, beta, f1 = _rayleigh_beta_from_first_mode(km_bundle, damping_ratio)

    C_full = rayleigh_damping_matrix(M_full, K_full, alpha, beta).tocsr()
    C_free = rayleigh_damping_matrix(M_free, K_free, alpha, beta).tocsr()

    return {
        'km_bundle': km_bundle,
        'C': {'full': C_full, 'free': C_free},
        'damping_ratio': damping_ratio,
        'rayleigh_alpha': alpha,
        'rayleigh_beta': beta,
        'first_natural_frequency_hz': f1,
    }


def get_impedance_matrix_view(
    impedance_bundle: dict,
    frequency_hz: float,
    matrix_scope: str = 'free',
    row_start: int = 0,
    col_start: int = 0,
    window_size: int = 12,
    heatmap_bins: int = 64,
) -> dict:
    """Serializa el mapa completo y una ventana legible de Z(ω) = K - ω²M + iωC."""
    if matrix_scope not in {'full', 'free'}:
        return {'error': "matrix_scope debe ser 'full' o 'free'."}
    if frequency_hz < 0:
        return {'error': 'frequency_hz no puede ser negativa.'}

    km_bundle = impedance_bundle['km_bundle']
    K = km_bundle['matrices'][('stiffness', matrix_scope)]
    M = km_bundle['matrices'][('mass', matrix_scope)]
    C = impedance_bundle['C'][matrix_scope]

    omega = 2 * np.pi * float(frequency_hz)
    Z = ((K - (omega ** 2) * M) + 1j * omega * C).tocsr()

    dimension = Z.shape[0]
    size = max(6, min(int(window_size), 36, max(dimension, 6)))
    size = min(size, dimension) if dimension else 0
    max_start = max(dimension - size, 0)
    row_start = max(0, min(int(row_start), max_start))
    col_start = max(0, min(int(col_start), max_start))
    row_end = row_start + size
    col_end = col_start + size

    dof_indices = (
        km_bundle['full_dof_indices']
        if matrix_scope == 'full'
        else km_bundle['free_dof_indices']
    )
    labels = km_bundle['labels']
    row_global_indices = dof_indices[row_start:row_end]
    col_global_indices = dof_indices[col_start:col_end]
    window_values = Z[row_start:row_end, col_start:col_end].toarray()

    result = {
        'matrix': {
            'scope': matrix_scope,
            'frequency_hz': float(frequency_hz),
            'omega_rad_s': float(omega),
            'damping_ratio': impedance_bundle['damping_ratio'],
            **_complex_sparse_matrix_stats(Z),
        },
        'metadata': {
            **km_bundle['metadata'],
            'first_natural_frequency_hz': impedance_bundle['first_natural_frequency_hz'],
            'rayleigh_alpha': impedance_bundle['rayleigh_alpha'],
            'rayleigh_beta': impedance_bundle['rayleigh_beta'],
        },
        'heatmap': _build_sparse_heatmap(Z, heatmap_bins),
        'window': {
            'row_start': row_start,
            'col_start': col_start,
            'size': size,
            'values_real': window_values.real,
            'values_imag': window_values.imag,
            'row_labels': [labels[int(index)] for index in row_global_indices],
            'col_labels': [labels[int(index)] for index in col_global_indices],
            'row_global_indices': row_global_indices,
            'col_global_indices': col_global_indices,
        },
    }
    return _sanitize_for_json(result)


def run_harmonic_analysis(request_data: dict) -> dict:
    try:
        # Extraer parámetros del request
        structure_data = request_data.get('structure', {})
        freq_start = float(request_data.get('freq_start', 0.1))
        freq_end = float(request_data.get('freq_end', 100.0))
        num_points = int(request_data.get('num_points', 100))
        damping_ratio = float(request_data.get('damping_ratio', 0.04))
        is_unbalanced = bool(request_data.get('is_unbalanced', False))
        unbalanced_me = float(request_data.get('unbalanced_me', 0.0))
        unbalanced_node_id = request_data.get('unbalanced_node_id')
        unbalanced_direction = request_data.get('unbalanced_direction') or [0.0, 1.0, 0.0]
        unbalanced_plane = str(request_data.get('unbalanced_plane') or 'direction').lower()
        unbalanced_mass = float(request_data.get('unbalanced_mass') or 0.0)
        unbalanced_eccentricity = float(request_data.get('unbalanced_eccentricity') or 0.0)
        mass_type = structure_data.get('settings', {}).get('mass_type', 'lumped')

        if freq_start < 0:
            return {"error": "La frecuencia inicial no puede ser negativa."}
        if freq_end <= freq_start:
            return {"error": "La frecuencia final debe ser mayor que la frecuencia inicial."}
        if num_points < 2 or num_points > 1000:
            return {"error": "El barrido armónico debe tener entre 2 y 1000 puntos."}
        if damping_ratio < 0:
            return {"error": "La relación de amortiguamiento no puede ser negativa."}
        has_explicit_unbalance = is_unbalanced and (
            unbalanced_node_id is not None or unbalanced_mass > 0 or unbalanced_eccentricity > 0
        )
        if is_unbalanced:
            if has_explicit_unbalance:
                if unbalanced_node_id is None:
                    return {"error": "Para fuerza de desbalance, selecciona el nodo donde se aplicará la excitación."}
                if unbalanced_mass <= 0:
                    return {"error": "Para fuerza de desbalance, la masa m debe ser mayor que cero."}
                if unbalanced_eccentricity <= 0:
                    return {"error": "Para fuerza de desbalance, la excentricidad e debe ser mayor que cero."}
                unbalanced_me = unbalanced_mass * unbalanced_eccentricity
            elif unbalanced_me <= 0:
                return {"error": "Para fuerza de desbalance, el producto m·e debe ser mayor que cero."}

        structure, nodes_map_core, _, id_map, warnings = _build_structure(structure_data, mass_type=mass_type)
        loads = structure_data.get('loads', [])
        
        if warnings:
            print(f"\n⚠️  ADVERTENCIAS DE DATOS INVALIDOS:")
            for warning in warnings:
                print(f"  - {warning}")
            print()

        K_global, M_global = assemble_global_matrices(structure, include_mass=True)
        
        if K_global is None or M_global is None:
            return {"error": "Error al ensamblar las matrices globales."}

        # Calcular amortiguamiento de Rayleigh simplificado con la primera frecuencia natural.
        freqs, _, _ = modal_analysis(K_global, M_global, structure, num_modes=1, debug=False)
        if len(freqs) > 0 and freqs[0] > 0:
            omega1 = 2 * np.pi * freqs[0]
            beta = 2 * damping_ratio / omega1
            alpha = 0.0
        else:
            beta = 0.001
            alpha = 0.0
        
        C_global = rayleigh_damping_matrix(M_global, K_global, alpha, beta)

        # Preparar vector de fuerza:
        # - Fuerza armónica normal: las cargas del editor son amplitudes F0.
        # - Fuerza por desbalance nueva: el usuario define nodo + dirección + m + e.
        # - Fuerza por desbalance legacy: las cargas del editor definen la dirección y
        #   la magnitud se calcula con unbalanced_me * ω².
        F_amplitude = np.zeros(structure.num_dofs, dtype=np.complex128)
        dof_map = {"fx": 0, "fy": 1, "fz": 2, "mx": 3, "my": 4, "mz": 5}

        if has_explicit_unbalance:
            try:
                selected_node_id = int(unbalanced_node_id)
            except (TypeError, ValueError):
                return {"error": "El nodo de desbalance debe ser un ID de nodo válido."}

            rep_id = id_map.get(selected_node_id)
            if rep_id is None or rep_id not in nodes_map_core:
                return {"error": f"El nodo de desbalance {selected_node_id} no existe o no pertenece a la estructura analizada."}

            node_core = nodes_map_core[rep_id]
            plane_axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
            if unbalanced_plane in plane_axes:
                axis_cos, axis_sin = plane_axes[unbalanced_plane]
                # Con la convención Re(F·e^(iωt)): 1 genera cos(ωt) y -i genera
                # sin(ωt). Cada componente conserva amplitud me·ω², por lo que la
                # resultante instantánea rota con magnitud constante me·ω².
                F_amplitude[node_core.dofs[axis_cos]] = 1.0 + 0.0j
                F_amplitude[node_core.dofs[axis_sin]] = 0.0 - 1.0j
            elif unbalanced_plane == "direction":
                try:
                    direction = np.array(unbalanced_direction, dtype=float).reshape(-1)
                except (TypeError, ValueError):
                    return {"error": "La dirección del desbalance debe tener componentes numéricas [x, y, z]."}

                if direction.size != 3 or np.linalg.norm(direction) <= 1e-12:
                    return {"error": "La dirección del desbalance debe ser un vector translacional no nulo [x, y, z]."}

                direction = direction / np.linalg.norm(direction)
                for dof_idx, value in enumerate(direction):
                    F_amplitude[node_core.dofs[dof_idx]] = value
            else:
                return {"error": "El plano de giro debe ser uno de: xy, xz, yz o direction."}
        else:
            for load_data in loads:
                original_node_id = load_data.get('node_id')
                rep_id = id_map.get(original_node_id)
                if rep_id is not None and rep_id in nodes_map_core:
                    node_core = nodes_map_core[rep_id]
                    for force_type, dof_idx in dof_map.items():
                        F_amplitude[node_core.dofs[dof_idx]] += float(load_data.get(force_type, 0.0) or 0.0)

        if np.linalg.norm(F_amplitude) <= 1e-12:
            if is_unbalanced:
                return {"error": "El análisis armónico por desbalance requiere un nodo y una dirección no nula para definir la excitación."}
            return {"error": "El análisis armónico requiere al menos una carga nodal no nula para definir la excitación."}

        freq_range = np.linspace(freq_start, freq_end, num_points)

        constrained_dofs = structure.get_constrained_dofs()
        free_dofs = np.setdiff1d(np.arange(structure.num_dofs), constrained_dofs)
        if free_dofs.size == 0:
            return {"error": "La estructura no tiene grados de libertad libres para análisis armónico."}

        solver_force = F_amplitude
        if is_unbalanced:
            force_norm = np.linalg.norm(F_amplitude)
            if force_norm <= 1e-12:
                return {"error": "El análisis armónico por desbalance requiere una dirección no nula para definir la excitación."}
            # Un fasor rotativo posee dos componentes unitarias en cuadratura.
            # Normalizar su norma sqrt(2) reduciría incorrectamente la fuerza física.
            solver_force = F_amplitude if unbalanced_plane in {"xy", "xz", "yz"} else F_amplitude / force_norm

        F_free = solver_force[free_dofs]
        if np.linalg.norm(F_free) <= 1e-12:
            return {"error": "La excitación armónica está aplicada solo en grados de libertad restringidos; no genera respuesta dinámica libre."}

        K_free = K_global[free_dofs, :][:, free_dofs]
        M_free = M_global[free_dofs, :][:, free_dofs]
        C_free = C_global[free_dofs, :][:, free_dofs]

        complex_responses = direct_frequency_response(
            K_free, M_free, C_free, F_free, freq_range,
            is_unbalanced_force=is_unbalanced,
            unbalanced_mass_product=unbalanced_me,
            normalize_unbalanced_direction=not is_unbalanced,
        )

        amplitudes = np.abs(complex_responses)
        omega = 2.0 * np.pi * freq_range
        free_index_lookup = np.full(structure.num_dofs, -1, dtype=int)
        free_index_lookup[free_dofs] = np.arange(free_dofs.size)

        # Expandir la solución reducida a un vector global complejo por frecuencia.
        # Los GDL restringidos permanecen en cero, lo cual permite recuperar esfuerzos
        # elementales sin perder la correspondencia de índices globales.
        full_complex_responses = np.zeros((len(freq_range), structure.num_dofs), dtype=np.complex128)
        full_complex_responses[:, free_dofs] = complex_responses

        # Cada esfuerzo del circulo de Mohr (ver _compute_element_harmonic_stress_amplitudes)
        # se reporta por separado, no solo Von Mises: normal, cortante, los dos
        # principales y el cortante maximo. "Peor caso" entre elementos conectados
        # = el mas grande para todos, EXCEPTO sigma_2 (principal de compresion, que
        # en este modelo de amplitud siempre es <= 0): ahi el peor caso es el mas
        # NEGATIVO. invert=True acumula -sigma_2 con el mismo max() y se des-invierte
        # al leerlo, para no tener que inicializar el array en -inf.
        STRESS_END_FIELDS = (
            ("sigma_vm", "stress_pa", False),
            ("sigma_normal", "stress_normal_pa", False),
            ("tau_shear", "stress_shear_pa", False),
            ("sigma_1", "stress_sigma1_pa", False),
            ("sigma_2", "stress_sigma2_pa", True),
            ("tau_max", "stress_taumax_pa", False),
        )

        node_stress_series: dict[str, dict[int, np.ndarray]] = {
            series_name: {
                int(original_node_id): np.zeros_like(freq_range, dtype=float)
                for original_node_id in id_map.keys()
            }
            for _, series_name, _invert in STRESS_END_FIELDS
        }
        rep_to_originals: dict[int, list[int]] = {}
        for original_node_id, rep_id in id_map.items():
            rep_to_originals.setdefault(int(rep_id), []).append(int(original_node_id))

        element_series: dict[int, dict[str, np.ndarray]] = {
            element.element_id: {
                key: np.zeros_like(freq_range, dtype=float)
                for key in (
                    "displacement_m_i", "displacement_m_j",
                    "velocity_m_s_i", "velocity_m_s_j",
                    "acceleration_m_s2_i", "acceleration_m_s2_j",
                    *(f"{series_name}_{end}" for _, series_name, _invert in STRESS_END_FIELDS for end in ("i", "j")),
                )
            }
            for element in structure.elements
        }

        # Esfuerzo alternante por nodo = envolvente máxima de los extremos de elementos conectados.
        # Además, se conserva el resultado propio de cada elemento (sin colapsar a envolvente nodal).
        for freq_idx in range(len(freq_range)):
            u_full = full_complex_responses[freq_idx, :]
            for element in structure.elements:
                element_u = u_full[element.dof_indices]
                is_massless = _is_massless_link(element)
                # Enlace rigido sin masa: no aporta "temperatura" de esfuerzo propia.
                # La carga que transporta ya se refleja en los elementos reales
                # conectados al mismo nodo (metodo de rigidez estandar) — no hay
                # nada que transferir a mano, solo evitar reportar su artefacto.
                zero_end = {key: 0.0 for key, _, _invert in STRESS_END_FIELDS}
                end_i, end_j = (zero_end, zero_end) if is_massless else _compute_element_harmonic_stress_amplitudes(element, element_u)

                disp_i = float(np.sqrt(np.sum(np.abs(element_u[0:3]) ** 2)))
                disp_j = float(np.sqrt(np.sum(np.abs(element_u[6:9]) ** 2)))
                es = element_series[element.element_id]
                es["displacement_m_i"][freq_idx] = disp_i
                es["displacement_m_j"][freq_idx] = disp_j
                es["velocity_m_s_i"][freq_idx] = omega[freq_idx] * disp_i
                es["velocity_m_s_j"][freq_idx] = omega[freq_idx] * disp_j
                es["acceleration_m_s2_i"][freq_idx] = (omega[freq_idx] ** 2) * disp_i
                es["acceleration_m_s2_j"][freq_idx] = (omega[freq_idx] ** 2) * disp_j
                for stress_key, series_name, _invert in STRESS_END_FIELDS:
                    es[f"{series_name}_i"][freq_idx] = end_i[stress_key]
                    es[f"{series_name}_j"][freq_idx] = end_j[stress_key]

                if is_massless:
                    continue

                for node_idx, rep_node_id in enumerate([element.nodes[0].id, element.nodes[1].id]):
                    end_values = (end_i, end_j)[node_idx]
                    for original_node_id in rep_to_originals.get(int(rep_node_id), []):
                        for stress_key, series_name, invert in STRESS_END_FIELDS:
                            series = node_stress_series[series_name][original_node_id]
                            candidate = -end_values[stress_key] if invert else end_values[stress_key]
                            series[freq_idx] = max(series[freq_idx], candidate)

        results = {
            "frequencies_sweep": _to_list(freq_range),
            "response_amplitudes": {},
            "node_response_series": {},
            "node_displacement_components": {},
            "node_peak_summary": {},
            "element_response_series": {
                element_id: {key: _to_list(arr) for key, arr in metrics.items()}
                for element_id, metrics in element_series.items()
            },
            "peak_node_id": None,
            "peak_frequency": None,
            "peak_amplitude": None,
        }

        peak_node_id = None
        peak_frequency = None
        peak_amplitude = -np.inf

        for original_node_id, rep_id in id_map.items():
            if rep_id in nodes_map_core:
                node_core = nodes_map_core[rep_id]
                node_dofs = node_core.dofs[:3]
                node_free_indices = free_index_lookup[node_dofs]
                valid_indices = node_free_indices[node_free_indices >= 0]
                node_complex = full_complex_responses[:, node_dofs]
                # Rotacionales tambien, solo para poder reusar generate_results_figure
                # (necesita rx,ry,rz reales para la curva de flexion Hermite) — no
                # entran en disp_mag/vel/acel, esos siguen siendo solo traslacionales.
                node_rot_complex = full_complex_responses[:, node_core.dofs[3:6]]
                if valid_indices.size > 0:
                    disp_mag = np.sqrt(np.sum(np.abs(node_complex) ** 2, axis=1))
                else:
                    disp_mag = np.zeros_like(freq_range, dtype=float)
                vel_mag = omega * disp_mag
                acc_mag = (omega**2) * disp_mag
                zero_freq_array = np.zeros_like(freq_range, dtype=float)
                stress_mag = node_stress_series["stress_pa"].get(int(original_node_id), zero_freq_array)

                results["response_amplitudes"][original_node_id] = _to_list(disp_mag)
                results["node_response_series"][original_node_id] = {
                    "displacement_m": _to_list(disp_mag),
                    "velocity_m_s": _to_list(vel_mag),
                    "acceleration_m_s2": _to_list(acc_mag),
                    "stress_pa": _to_list(stress_mag),
                    # Circulo de Mohr por nodo (envolvente entre extremos conectados,
                    # mismo criterio que stress_pa/Von Mises).
                    "stress_normal_pa": _to_list(node_stress_series["stress_normal_pa"].get(int(original_node_id), zero_freq_array)),
                    "stress_shear_pa": _to_list(node_stress_series["stress_shear_pa"].get(int(original_node_id), zero_freq_array)),
                    "stress_sigma1_pa": _to_list(node_stress_series["stress_sigma1_pa"].get(int(original_node_id), zero_freq_array)),
                    # sigma_2 se acumulo invertido (ver STRESS_END_FIELDS): des-invertir aca.
                    "stress_sigma2_pa": _to_list(-node_stress_series["stress_sigma2_pa"].get(int(original_node_id), zero_freq_array)),
                    "stress_taumax_pa": _to_list(node_stress_series["stress_taumax_pa"].get(int(original_node_id), zero_freq_array)),
                }
                results["node_displacement_components"][original_node_id] = {
                    "ux_real_m": _to_list(np.real(node_complex[:, 0])),
                    "ux_imag_m": _to_list(np.imag(node_complex[:, 0])),
                    "uy_real_m": _to_list(np.real(node_complex[:, 1])),
                    "uy_imag_m": _to_list(np.imag(node_complex[:, 1])),
                    "uz_real_m": _to_list(np.real(node_complex[:, 2])),
                    "uz_imag_m": _to_list(np.imag(node_complex[:, 2])),
                    "rx_real_rad": _to_list(np.real(node_rot_complex[:, 0])),
                    "rx_imag_rad": _to_list(np.imag(node_rot_complex[:, 0])),
                    "ry_real_rad": _to_list(np.real(node_rot_complex[:, 1])),
                    "ry_imag_rad": _to_list(np.imag(node_rot_complex[:, 1])),
                    "rz_real_rad": _to_list(np.real(node_rot_complex[:, 2])),
                    "rz_imag_rad": _to_list(np.imag(node_rot_complex[:, 2])),
                }

                finite_mask = np.isfinite(disp_mag)
                if np.any(finite_mask):
                    finite_indices = np.where(finite_mask)[0]
                    local_idx = finite_indices[int(np.argmax(disp_mag[finite_mask]))]
                    local_peak = float(disp_mag[local_idx])
                    stress_finite = np.isfinite(stress_mag)
                    stress_idx = int(np.argmax(np.where(stress_finite, stress_mag, -np.inf))) if np.any(stress_finite) else local_idx
                    results["node_peak_summary"][original_node_id] = {
                        "frequency_hz": float(freq_range[local_idx]),
                        "displacement_m": float(disp_mag[local_idx]),
                        "velocity_m_s": float(vel_mag[local_idx]),
                        "acceleration_m_s2": float(acc_mag[local_idx]),
                        "stress_pa": float(stress_mag[local_idx]),
                        "stress_peak_pa": float(stress_mag[stress_idx]) if np.any(stress_finite) else 0.0,
                        "stress_peak_frequency_hz": float(freq_range[stress_idx]) if np.any(stress_finite) else float(freq_range[local_idx]),
                    }
                    if local_peak > peak_amplitude:
                        peak_amplitude = local_peak
                        peak_frequency = float(freq_range[local_idx])
                        peak_node_id = int(original_node_id)

        if peak_node_id is not None:
            results["peak_node_id"] = peak_node_id
            results["peak_frequency"] = peak_frequency
            results["peak_amplitude"] = peak_amplitude

        return _sanitize_for_json(results)

    except Exception as e:
        return {"error": f"Error fatal en el nucleo de analisis armonico: {str(e)}"}
