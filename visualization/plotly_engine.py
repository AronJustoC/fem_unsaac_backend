import plotly.graph_objects as go
import numpy as np
import re


def get_theme_config(theme="dark"):
    is_dark = theme == "dark"
    
    if is_dark:
        config = {
            'bg_color': '#000000',
            'grid_color': 'rgba(255, 255, 255, 0.15)',
            'axis_color': '#888888',
            'text_color': '#ffffff',
            'label_color': '#cccccc',
            'node_fill': '#000000',
            'node_border': '#ffffff',
            'palette': ['#1e90ff', '#00ff00', '#ffff00', '#ff00ff',
                       '#00ffff', '#ffa500', '#ffc0cb', '#ffffff'],
            'grid_bg_color': 'rgba(255,255,255,0.02)',
            'is_dark': True
        }
    else:
        config = {
            'bg_color': '#ffffff',
            'grid_color': 'rgba(0, 0, 0, 0.1)',
            'axis_color': '#444444',
            'text_color': '#000000',
            'label_color': '#333333',
            'node_fill': '#ffffff',
            'node_border': '#000000',
            'palette': ['#1e90ff', '#008000', '#c2a100', '#db2777',
                       '#009999', '#ea580c', '#db2777', '#000000'],
            'grid_bg_color': 'rgba(0,0,0,0.02)',
            'is_dark': False
        }
    
    config['font_family'] = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    return config


def process_nodes(structure_data):
    nodes = []
    for n in structure_data.get('nodes', []):
        nodes.append({
            **n,
            'coords': [n['coords'][0] * 1000, n['coords'][1] * 1000, n['coords'][2] * 1000]
        })
    return nodes


def _positive_float(value, fallback):
    try:
        parsed = float(value)
        return parsed if np.isfinite(parsed) and parsed > 0 else fallback
    except (TypeError, ValueError):
        return fallback


def _infer_i_dimensions(area_m2, inertia_m4, height_to_width_ratio):
    """Recupera h, b y t de un perfil H/I de espesor uniforme.

    Usa A e I fuerte, por lo que también respeta modelos a escala cuyo nombre
    conserva la designación nominal (p. ej. H420x180 dibujado a escala 1:10).
    """
    if not inertia_m4 or inertia_m4 <= 0 or height_to_width_ratio <= 0:
        return None

    width_to_height = 1.0 / height_to_width_ratio

    def properties(height):
        width = width_to_height * height
        thickness = area_m2 / (height + 2.0 * width)
        if thickness <= 0 or 2.0 * thickness >= height or thickness >= width:
            return None
        web_height = height - 2.0 * thickness
        strong_inertia = (
            2.0 * (
                width * thickness**3 / 12.0
                + width * thickness * ((height - thickness) / 2.0) ** 2
            )
            + thickness * web_height**3 / 12.0
        )
        return strong_inertia, width, thickness

    characteristic = max(np.sqrt(area_m2), 1e-6)
    candidates = np.geomspace(characteristic * 0.25, characteristic * 40.0, 240)
    previous = None
    bracket = None
    for height in candidates:
        current = properties(float(height))
        if current is None:
            continue
        delta = current[0] - inertia_m4
        if previous is not None and previous[1] * delta <= 0:
            bracket = (previous[0], float(height))
            break
        previous = (float(height), delta)

    if bracket is None:
        return None

    low, high = bracket
    for _ in range(70):
        mid = (low + high) / 2.0
        current = properties(mid)
        if current is None or current[0] < inertia_m4:
            low = mid
        else:
            high = mid

    height = (low + high) / 2.0
    _, width, thickness = properties(height)
    return height, width, thickness


def _section_geometry(section):
    """Normaliza la geometría de la sección y devuelve dimensiones en mm.

    Los proyectos antiguos solo guardaban A/I/J. Para no romperlos se infiere
    el tipo por el nombre y un tamaño razonable por el área o la designación IPE.
    """
    name = str(section.get('name') or '').strip().lower()
    raw_shape = str(section.get('visual_shape') or section.get('shape') or '').strip().lower()

    if raw_shape in {'h', 'hea', 'heb', 'wide_flange'}:
        shape = 'h'
    elif raw_shape in {'i', 'ipe'}:
        shape = 'i'
    elif raw_shape in {'circle', 'circular', 'round', 'pipe', 'tube'}:
        shape = 'circular'
    elif raw_shape in {'rect', 'rectangle', 'rectangular', 'square'}:
        shape = 'rectangular'
    elif any(token in name for token in ('hea', 'heb', 'perfil h', 'viga h')) or re.search(r'(^|\s)h\s*\d+\s*x', name):
        shape = 'h'
    elif any(token in name for token in ('ipe', 'perfil i', 'viga i')):
        shape = 'i'
    elif any(token in name for token in ('circular', 'tubo', 'pipe')):
        shape = 'circular'
    else:
        shape = 'rectangular'

    area_m2 = _positive_float(section.get('area', section.get('A')), 0.01)
    characteristic_m = max(np.sqrt(area_m2), 0.02)
    dimension_match = re.search(r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)', name)

    inferred_height_m = None
    inferred_width_m = None
    inferred_thickness_m = None
    if shape in {'i', 'h'} and dimension_match:
        nominal_height = float(dimension_match.group(1))
        nominal_width = float(dimension_match.group(2))
        strong_inertia = max(
            _positive_float(section.get('Iy'), 0.0),
            _positive_float(section.get('Iz'), 0.0),
        )
        inferred = _infer_i_dimensions(
            area_m2,
            strong_inertia,
            nominal_height / nominal_width,
        )
        if inferred:
            inferred_height_m, inferred_width_m, inferred_thickness_m = inferred
    elif shape == 'i':
        designation = re.search(r'ipe\s*[- ]?\s*(\d{2,4})', name)
        if designation:
            inferred_height_m = float(designation.group(1)) / 1000.0
            inferred_width_m = inferred_height_m * 0.5
    elif shape == 'rectangular' and dimension_match:
        # El área determina la escala real; el nombre solo aporta la proporción.
        # Así 80x40 con A=32 mm² se dibuja 8x4 mm, no 80x40 mm.
        ratio = float(dimension_match.group(1)) / float(dimension_match.group(2))
        inferred_height_m = np.sqrt(area_m2 * ratio)
        inferred_width_m = np.sqrt(area_m2 / ratio)
    elif shape == 'rectangular':
        # Rectángulo equivalente dinámico a partir de A e inercia fuerte.
        # Permite visualizar secciones importadas aunque su nombre no codifique medidas.
        strong_inertia = max(
            _positive_float(section.get('Iy'), 0.0),
            _positive_float(section.get('Iz'), 0.0),
        )
        if strong_inertia > 0:
            candidate_height = np.sqrt(12.0 * strong_inertia / area_m2)
            candidate_width = area_m2 / candidate_height
            if np.isfinite(candidate_height) and np.isfinite(candidate_width):
                inferred_height_m = max(candidate_height, candidate_width)
                inferred_width_m = min(candidate_height, candidate_width)

    height_m = _positive_float(
        section.get('visual_height', section.get('height')),
        inferred_height_m or (characteristic_m * (2.0 if shape in {'i', 'h'} else 1.0)),
    )
    width_m = _positive_float(
        section.get('visual_width', section.get('width')),
        inferred_width_m or (height_m * 0.5 if shape in {'i', 'h'} else characteristic_m),
    )
    flange_m = _positive_float(
        section.get('visual_flange_thickness', section.get('flange_thickness')),
        inferred_thickness_m or height_m * 0.08,
    )
    web_m = _positive_float(
        section.get('visual_web_thickness', section.get('web_thickness')),
        inferred_thickness_m or width_m * 0.08,
    )

    # Mantener una sección físicamente dibujable incluso con datos importados erróneos.
    flange_m = min(flange_m, height_m * 0.45)
    web_m = min(web_m, width_m * 0.9)

    return {
        'shape': shape,
        'height': height_m * 1000.0,
        'width': width_m * 1000.0,
        'flange_thickness': flange_m * 1000.0,
        'web_thickness': web_m * 1000.0,
    }


def _local_axes(c1, c2, rotation_deg=0.0):
    axis = np.asarray(c2, dtype=float) - np.asarray(c1, dtype=float)
    length = np.linalg.norm(axis)
    if length <= 1e-9:
        return None

    local_x = axis / length
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(local_x, helper)) > 0.9:
        # Para montantes verticales del Bailey: ancho fuera del plano global Y
        # y altura dentro del plano lateral global X-Z.
        helper = np.array([1.0, 0.0, 0.0])
    local_y = np.cross(helper, local_x)
    local_y /= np.linalg.norm(local_y)
    local_z = np.cross(local_x, local_y)
    local_z /= np.linalg.norm(local_z)

    angle = np.deg2rad(_positive_float(abs(rotation_deg), 0.0))
    if float(rotation_deg or 0.0) < 0:
        angle *= -1.0
    if abs(angle) > 1e-12:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotated_y = cos_a * local_y + sin_a * local_z
        rotated_z = -sin_a * local_y + cos_a * local_z
        local_y, local_z = rotated_y, rotated_z
    return local_y, local_z


def _append_box(mesh, c1, c2, local_y, local_z, width, height, hover_text, element_id, offset_y=0.0, offset_z=0.0):
    base = len(mesh['x'])
    offsets = [
        (-width / 2, -height / 2),
        (width / 2, -height / 2),
        (width / 2, height / 2),
        (-width / 2, height / 2),
    ]
    for point in (np.asarray(c1, dtype=float), np.asarray(c2, dtype=float)):
        center = point + local_y * offset_y + local_z * offset_z
        for dy, dz in offsets:
            vertex = center + local_y * dy + local_z * dz
            mesh['x'].append(float(vertex[0]))
            mesh['y'].append(float(vertex[1]))
            mesh['z'].append(float(vertex[2]))
            mesh['text'].append(hover_text)
            mesh['customdata'].append(element_id)

    faces = [
        (0, 2, 1), (0, 3, 2),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7),
    ]
    for i, j, k in faces:
        mesh['i'].append(base + i)
        mesh['j'].append(base + j)
        mesh['k'].append(base + k)


def _append_cylinder(mesh, c1, c2, local_y, local_z, diameter, hover_text, element_id, segments=16):
    base = len(mesh['x'])
    radius = diameter / 2.0
    for point in (np.asarray(c1, dtype=float), np.asarray(c2, dtype=float)):
        for index in range(segments):
            angle = 2.0 * np.pi * index / segments
            vertex = point + local_y * (radius * np.cos(angle)) + local_z * (radius * np.sin(angle))
            mesh['x'].append(float(vertex[0]))
            mesh['y'].append(float(vertex[1]))
            mesh['z'].append(float(vertex[2]))
            mesh['text'].append(hover_text)
            mesh['customdata'].append(element_id)

    for index in range(segments):
        nxt = (index + 1) % segments
        a, b = base + index, base + nxt
        c, d = base + segments + index, base + segments + nxt
        mesh['i'].extend([a, a])
        mesh['j'].extend([b, d])
        mesh['k'].extend([d, c])

    # Tapas trianguladas como abanico; no requieren vértices centrales adicionales.
    for index in range(1, segments - 1):
        mesh['i'].extend([base, base + segments])
        mesh['j'].extend([base + index + 1, base + segments + index])
        mesh['k'].extend([base + index, base + segments + index + 1])


def _append_element_solid(mesh, c1, c2, geometry, hover_text, element_id, rotation_deg=0.0):
    axes = _local_axes(c1, c2, rotation_deg=rotation_deg)
    if axes is None:
        return False
    local_y, local_z = axes

    if geometry['shape'] == 'circular':
        _append_cylinder(mesh, c1, c2, local_y, local_z, geometry['width'], hover_text, element_id)
    elif geometry['shape'] in {'i', 'h'}:
        height = geometry['height']
        flange = geometry['flange_thickness']
        web_height = max(height - 2.0 * flange, height * 0.1)
        flange_offset = (height - flange) / 2.0
        _append_box(mesh, c1, c2, local_y, local_z, geometry['width'], flange, hover_text, element_id, offset_z=flange_offset)
        _append_box(mesh, c1, c2, local_y, local_z, geometry['width'], flange, hover_text, element_id, offset_z=-flange_offset)
        _append_box(mesh, c1, c2, local_y, local_z, geometry['web_thickness'], web_height, hover_text, element_id)
    else:
        _append_box(mesh, c1, c2, local_y, local_z, geometry['width'], geometry['height'], hover_text, element_id)
    return True


def add_elements_to_figure(fig, elements, nodes, materials, sections, palette):
    node_map = {n['id']: n for n in nodes}
    section_ids = sorted(list(set(el.get('section_id')
                         for el in elements if el.get('section_id') is not None)))
    color_map = {sid: palette[i % len(palette)]
                 for i, sid in enumerate(section_ids)}

    sections_elements = {}
    for el in elements:
        sid = el.get('section_id')
        sections_elements.setdefault(sid, []).append(el)

    for sid, els in sections_elements.items():
        ex, ey, ez = [], [], []
        ex_customdata = []
        mid_x, mid_y, mid_z, mid_text, mid_labels = [], [], [], [], []
        mesh = {'x': [], 'y': [], 'z': [], 'i': [], 'j': [], 'k': [], 'text': [], 'customdata': []}

        section_info = sections.get(sid, {})
        section_name = section_info.get('name', f"Sec {sid}")
        geometry = _section_geometry(section_info)

        for el in els:
            n1 = node_map.get(el['node_ids'][0])
            n2 = node_map.get(el['node_ids'][1])
            if n1 and n2:
                c1, c2 = n1['coords'], n2['coords']
                ex.extend([c1[0], c2[0], None])
                ey.extend([c1[1], c2[1], None])
                ez.extend([c1[2], c2[2], None])
                ex_customdata.extend([el['id'], el['id'], None])

                mid_x.append((c1[0] + c2[0]) / 2)
                mid_y.append((c1[1] + c2[1]) / 2)
                mid_z.append((c1[2] + c2[2]) / 2)

                mat_id = el.get('material_id')
                mat_name = materials.get(mat_id, {}).get('name', f"M{mat_id}")
                shape_label = {
                    'i': 'Perfil I',
                    'h': 'Perfil H',
                    'circular': 'Circular',
                    'rectangular': 'Rectangular',
                }[geometry['shape']]
                rotation_deg = el.get('visual_rotation_deg', section_info.get('visual_rotation_deg', 0.0)) or 0.0
                hover_text = (
                    f"<b>Elemento {el['id']}</b><br>Sección: {section_name}"
                    f"<br>Forma: {shape_label}<br>Dimensiones: "
                    f"{geometry['height']:.1f} × {geometry['width']:.1f} mm"
                    f"<br>Material: {mat_name}"
                )
                mid_text.append(hover_text)
                mid_labels.append(f"E{el['id']}")
                _append_element_solid(
                    mesh,
                    c1,
                    c2,
                    geometry,
                    hover_text,
                    el['id'],
                    rotation_deg=rotation_deg,
                )

        if ex:
            color = color_map.get(sid, palette[0])
            if mesh['x']:
                fig.add_trace(go.Mesh3d(
                    x=mesh['x'], y=mesh['y'], z=mesh['z'],
                    i=mesh['i'], j=mesh['j'], k=mesh['k'],
                    color=color,
                    opacity=0.92,
                    flatshading=True,
                    lighting=dict(ambient=0.55, diffuse=0.8, specular=0.35, roughness=0.55, fresnel=0.15),
                    lightposition=dict(x=2000, y=2500, z=3500),
                    name=section_name,
                    text=mesh['text'],
                    customdata=mesh['customdata'],
                    hovertemplate='%{text}<extra></extra>',
                    legendgroup=f'sec_{sid}',
                    showlegend=True,
                ))
            fig.add_trace(go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode='lines',
                line=dict(color=color, width=2),
                name=f'Eje · {section_name}',
                customdata=ex_customdata,
                hovertemplate='<b>Elemento %{customdata}</b><br>Haz clic para inspeccionar matrices<extra></extra>',
                legendgroup=f'sec_{sid}',
                showlegend=not bool(mesh['x']),
                opacity=0.55,
            ))
            fig.add_trace(go.Scatter3d(
                x=mid_x, y=mid_y, z=mid_z,
                mode='markers',
                marker=dict(size=2, opacity=0, color=color),
                text=mid_labels,
                hovertext=mid_text,
                hoverinfo='text',
                showlegend=False,
                legendgroup=f'sec_{sid}'
            ))


def add_nodes_to_figure(fig, nodes, node_fill, node_border):
    if nodes:
        fig.add_trace(go.Scatter3d(
            x=[n['coords'][0] for n in nodes],
            y=[n['coords'][1] for n in nodes],
            z=[n['coords'][2] for n in nodes],
            mode='markers',
            marker=dict(
                size=6,
                color=node_fill,
                line=dict(color=node_border, width=1.5),
                symbol='circle',
                opacity=1.0
            ),
            hovertext=[f"<b>Nodo {n['id']}</b><br>({n['coords'][0]:.2f}, {n['coords'][1]:.2f}, {n['coords'][2]:.2f})" for n in nodes],
            hoverinfo='text',
            name='Nodos'
        ))


def add_supports_to_figure(fig, structure_data, nodes, node_border):
    node_map = {n['id']: n for n in nodes}
    restraints = structure_data.get('restraints', {})

    support_styles = {
        'fixed': {'symbol': 'square', 'color': '#00ff00', 'name': 'Empotrado'},
        'pinned': {'symbol': 'triangle-up', 'color': '#00ff00', 'name': 'Articulado'},
        'other': {'symbol': 'circle', 'color': '#00ff00', 'name': 'Restricción'}
    }

    support_data = {k: {'x': [], 'y': [], 'z': [], 'text': []} for k in support_styles}

    for nid_str, dofs in restraints.items():
        node = node_map.get(int(nid_str))
        if not node: continue

        dofs_set = set(dofs)
        if len(dofs_set) >= 6:
            k = 'fixed'
        elif all(d in dofs_set for d in ['TX', 'TY', 'TZ']):
            k = 'pinned'
        else:
            k = 'other'

        support_data[k]['x'].append(node['coords'][0])
        support_data[k]['y'].append(node['coords'][1])
        support_data[k]['z'].append(node['coords'][2])
        unique_dofs = sorted(list(set(dofs)), key=lambda x: ['ux', 'uy', 'uz', 'rx', 'ry', 'rz'].index(
            x) if x in ['ux', 'uy', 'uz', 'rx', 'ry', 'rz'] else 99)
        support_data[k]['text'].append(f"Nodo {nid_str}<br>{', '.join(unique_dofs)}")

    for k, style in support_styles.items():
        data = support_data[k]
        if data['x']:
            fig.add_trace(go.Scatter3d(
                x=data['x'], y=data['y'], z=data['z'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=style['color'],
                    symbol=style['symbol'],
                    line=dict(color=node_border, width=1),
                    opacity=1.0
                ),
                text=data['text'],
                name=style['name'],
                hoverinfo='text'
            ))


def add_loads_to_figure(fig, structure_data, nodes):
    node_map = {n['id']: n for n in nodes}
    loads = structure_data.get('loads', [])
    if not loads: return

    forces = [np.linalg.norm([l.get('fx', 0), l.get('fy', 0), l.get('fz', 0)]) for l in loads]
    if not forces: return

    max_f = max(forces + [1.0])
    all_coords = np.array([n['coords'] for n in nodes])
    if len(all_coords) > 0:
        min_c = np.min(all_coords, axis=0)
        max_c = np.max(all_coords, axis=0)
        max_dim = np.max(max_c - min_c)
        if max_dim == 0: max_dim = 100.0
    else:
        max_dim = 100.0

    arrow_scale = max(max_dim * 0.15, 80.0)
    shaft_x, shaft_y, shaft_z = [], [], []
    head_x, head_y, head_z = [], [], []
    head_u, head_v, head_w = [], [], []

    for load in loads:
        node = node_map.get(load['node_id'])
        if not node: continue

        fx, fy, fz = load.get('fx', 0), load.get('fy', 0), load.get('fz', 0)
        mag = np.linalg.norm([fx, fy, fz])

        if mag > 0:
            dx, dy, dz = fx/mag, fy/mag, fz/mag
            rel_mag = mag / max_f
            visual_len = arrow_scale * (0.4 + 0.6 * rel_mag)

            tail_x, tail_y, tail_z = node['coords']
            tip_x = tail_x + dx * visual_len
            tip_y = tail_y + dy * visual_len
            tip_z = tail_z + dz * visual_len

            head_len = visual_len * 0.35
            shaft_end_x = tip_x - dx * head_len
            shaft_end_y = tip_y - dy * head_len
            shaft_end_z = tip_z - dz * head_len

            shaft_x.extend([tail_x, shaft_end_x, None])
            shaft_y.extend([tail_y, shaft_end_y, None])
            shaft_z.extend([tail_z, shaft_end_z, None])

            head_x.append(tip_x)
            head_y.append(tip_y)
            head_z.append(tip_z)
            head_u.append(dx * head_len)
            head_v.append(dy * head_len)
            head_w.append(dz * head_len)

    if shaft_x:
        fig.add_trace(go.Scatter3d(
            x=shaft_x, y=shaft_y, z=shaft_z,
            mode='lines',
            line=dict(color='#ff0000', width=5),
            name='Líneas de Carga',
            showlegend=False,
            hoverinfo='none'
        ))

    if head_x:
        fig.add_trace(go.Cone(
            x=head_x, y=head_y, z=head_z,
            u=head_u, v=head_v, w=head_w,
            colorscale=[[0, '#ff0000'], [1, '#ff0000']],
            showscale=False,
            sizemode='scaled',
            sizeref=0.5,
            anchor='tip',
            name='Cargas',
            opacity=1.0,
            hoverinfo='none'
        ))


def apply_layout_config(fig, nodes, cfg):
    is_dark = cfg['is_dark']
    bg_color = cfg['bg_color']
    text_color = cfg['text_color']
    grid_color = cfg['grid_color']
    axis_color = cfg['axis_color']
    font_family = cfg['font_family']

    if nodes:
        all_coords = np.array([n['coords'] for n in nodes])
        min_coords = np.min(all_coords, axis=0)
        max_coords = np.max(all_coords, axis=0)
        global_span = max(float(np.max(max_coords - min_coords)), 1000.0)
        
        def get_range(min_v, max_v):
            span = max_v - min_v
            # En ejes planos se necesita espacio para ver el volumen de la sección,
            # no un rango de solo ±40 mm que recorte perfiles de 100–300 mm.
            margin = span * 0.2 if span > 1e-6 else max(global_span * 0.12, 150.0)
            return [min_v - margin, max_v + margin]

        rx = get_range(min_coords[0], max_coords[0])
        ry = get_range(min_coords[1], max_coords[1])
        rz = get_range(min_coords[2], max_coords[2])
    else:
        rx = ry = rz = [-100, 100]

    axis_config = dict(
        gridcolor=grid_color,
        zerolinecolor=axis_color,
        zerolinewidth=2,
        title_font=dict(size=12, family=font_family, color=text_color),
        tickfont=dict(size=10, family=font_family),
        nticks=6,
        tickformat='.0f',
        showgrid=True,
        color=text_color,
        showbackground=True,
        backgroundcolor='rgba(0,0,0,0.01)' if is_dark else 'rgba(0,0,0,0.005)',
        gridwidth=1.0,
        showspikes=False
    )

    fig.update_layout(
        template='plotly_dark' if is_dark else 'plotly_white',
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(family=font_family, color=text_color),
        scene=dict(
            xaxis=dict(**axis_config, title=dict(text='X (mm)'), range=rx, autorange=False),
            yaxis=dict(**axis_config, title=dict(text='Y (mm)'), range=ry, autorange=False),
            zaxis=dict(**axis_config, title=dict(text='Z (mm)'), range=rz, autorange=False),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.4),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                projection=dict(type='perspective')
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(
            yanchor="top", y=0.98, xanchor="right", x=0.98,
            bgcolor=bg_color, bordercolor=grid_color, borderwidth=1,
            font=dict(size=11, family=font_family, color=text_color)
        ),
        hoverlabel=dict(
            bgcolor=bg_color, font_size=11, font_family=font_family, bordercolor=grid_color
        ),
        hovermode='closest',
        dragmode='orbit'
    )


def generate_structure_figure(structure_data, theme="dark"):
    nodes = process_nodes(structure_data)
    cfg = get_theme_config(theme)
    fig = go.Figure()

    materials = {m['id']: m for m in structure_data.get('materials', [])}
    sections = {s['id']: s for s in structure_data.get('sections', [])}
    
    add_elements_to_figure(fig, structure_data.get('elements', []), nodes, materials, sections, cfg['palette'])
    add_nodes_to_figure(fig, nodes, cfg['node_fill'], cfg['node_border'])
    add_supports_to_figure(fig, structure_data, nodes, cfg['node_border'])
    add_loads_to_figure(fig, structure_data, nodes)
    apply_layout_config(fig, nodes, cfg)

    return fig


def interpolate_beam(p1, p2, d1, d2, segments=4):
    v_orig = np.array(p2) - np.array(p1)
    L = np.linalg.norm(v_orig)
    if L < 1e-6:
        return [], [], [], [], [], []
    v_unit = v_orig / L

    disp1 = np.array(d1[:3]) * 1000
    disp2 = np.array(d2[:3]) * 1000
    theta1 = np.array(d1[3:])
    theta2 = np.array(d2[3:])

    t1_def_unit = np.cross(theta1, v_unit)
    t2_def_unit = np.cross(theta2, v_unit)
    T1_unit = t1_def_unit * L
    T2_unit = t2_def_unit * L
    
    P1_base = np.array(p1)
    P2_base = np.array(p2)
    T1_base = v_unit * L
    T2_base = v_unit * L

    base_x, base_y, base_z = [], [], []
    delta_x, delta_y, delta_z = [], [], []
    t_vals = np.linspace(0, 1, segments+1)
    
    for t in t_vals:
        t2, t3 = t*t, t*t*t
        h00, h10 = 2*t3 - 3*t2 + 1, t3 - 2*t2 + t
        h01, h11 = -2*t3 + 3*t2, t3 - t2
        
        P_base = h00 * P1_base + h10 * T1_base + h01 * P2_base + h11 * T2_base
        base_x.append(float(P_base[0]))
        base_y.append(float(P_base[1]))
        base_z.append(float(P_base[2]))
        
        P_delta = h00 * disp1 + h10 * T1_unit + h01 * disp2 + h11 * T2_unit
        delta_x.append(float(P_delta[0]))
        delta_y.append(float(P_delta[1]))
        delta_z.append(float(P_delta[2]))
        
    return base_x, base_y, base_z, delta_x, delta_y, delta_z


def get_deformed_traces(
    sections_elements,
    node_map,
    displacements,
    scale,
    color,
    interpolation_segments=4,
    include_customdata=True,
    include_hover=True,
):
    traces = []
    
    def clean(v):
        if v is None: return None
        return float(v) if np.isfinite(v) else 0.0

    for sid, els in sections_elements.items():
        bx, by, bz = [], [], []
        dx, dy, dz = [], [], []
        element_ids = []
        for el in els:
            n1, n2 = node_map.get(el['node_ids'][0]), node_map.get(el['node_ids'][1])
            if not (n1 and n2): continue

            nid1, nid2 = el['node_ids'][0], el['node_ids'][1]
            d1 = displacements.get(nid1) or displacements.get(str(nid1), [0]*6)
            d2 = displacements.get(nid2) or displacements.get(str(nid2), [0]*6)
            d1, d2 = (list(d1) + [0]*6)[:6], (list(d2) + [0]*6)[:6]

            try:
                px_b, py_b, pz_b, px_d, py_d, pz_d = interpolate_beam(
                    n1['coords'],
                    n2['coords'],
                    d1,
                    d2,
                    segments=interpolation_segments,
                )
                bx.extend(px_b + [None]); by.extend(py_b + [None]); bz.extend(pz_b + [None])
                dx.extend(px_d + [None]); dy.extend(py_d + [None]); dz.extend(pz_d + [None])
                element_ids.extend([el['id']] * len(px_b) + [None])
            except:
                bx.extend([float(n1['coords'][0]), float(n2['coords'][0]), None])
                by.extend([float(n1['coords'][1]), float(n2['coords'][1]), None])
                bz.extend([float(n1['coords'][2]), float(n2['coords'][2]), None])
                dx.extend([float(d1[0]*1000), float(d2[0]*1000), None])
                dy.extend([float(d1[1]*1000), float(d2[1]*1000), None])
                dz.extend([float(d1[2]*1000), float(d2[2]*1000), None])
                element_ids.extend([el['id'], el['id'], None])

        final_x = [clean(bx[i] + dx[i]*scale) if bx[i] is not None else None for i in range(len(bx))]
        final_y = [clean(by[i] + dy[i]*scale) if by[i] is not None else None for i in range(len(by))]
        final_z = [clean(bz[i] + dz[i]*scale) if bz[i] is not None else None for i in range(len(bz))]

        trace_kwargs = dict(
            x=final_x, y=final_y, z=final_z,
            mode='lines',
            line=dict(color=color, width=4),
            name=f"Deformada {sid}",
        )

        if include_customdata:
            cdata = []
            for i in range(len(bx)):
                if bx[i] is None:
                    cdata.append([None] * 7)
                else:
                    cdata.append([
                        clean(bx[i]), clean(by[i]), clean(bz[i]),
                        clean(dx[i]), clean(dy[i]), clean(dz[i]),
                        element_ids[i],
                    ])
            trace_kwargs["customdata"] = cdata

        if include_hover:
            trace_kwargs["hovertemplate"] = "Deformada<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>"
        else:
            trace_kwargs["hoverinfo"] = "skip"
            trace_kwargs["showlegend"] = False

        traces.append(go.Scatter3d(**trace_kwargs))
    return traces


def generate_results_figure(
    structure_data,
    displacements,
    scale=1.0,
    theme="dark",
    animate=False,
    frame_count=10,
    interpolation_segments=4,
):
    fig = generate_structure_figure(structure_data, theme=theme)
    cfg = get_theme_config(theme)
    is_dark = cfg['is_dark']

    ghost_opacity = 0.15
    deformed_color = '#ff00ff'

    for trace in fig.data:
        if isinstance(trace, go.Mesh3d):
            # En resultados, el sólido original funciona como referencia y no debe
            # ocultar la línea deformada/modal superpuesta.
            trace.update(opacity=ghost_opacity, hoverinfo='skip', showlegend=False)
        elif isinstance(trace, go.Scatter3d):
            mode = getattr(trace, 'mode', '') or ''
            if mode == 'lines':
                orig_color = '#888888'
                try:
                    if hasattr(trace, 'line') and trace.line:
                        c = getattr(trace.line, 'color', None)
                        if c: orig_color = c
                except: pass
                trace.update(line=dict(width=2, color=orig_color), opacity=ghost_opacity)
            elif 'markers' in mode:
                trace.update(opacity=ghost_opacity)
        elif isinstance(trace, go.Cone):
            trace.update(opacity=ghost_opacity)

    nodes = process_nodes(structure_data)
    node_map = {n['id']: n for n in nodes}
    elements = structure_data.get('elements', [])
    sections_elements = {}
    for el in elements:
        sections_elements.setdefault(el.get('section_id'), []).append(el)

    interpolation_segments = int(max(1, min(interpolation_segments, 6)))
    base_traces = get_deformed_traces(
        sections_elements,
        node_map,
        displacements,
        scale,
        deformed_color,
        interpolation_segments=interpolation_segments,
        include_customdata=True,
        include_hover=True,
    )
    if base_traces: base_traces[0].showlegend = True
    for t in base_traces: fig.add_trace(t)

    if animate:
        frames = []
        num_frames = int(max(4, min(frame_count, 30)))
        static_trace_count = len(fig.data) - len(base_traces)
        indices_to_update = list(range(static_trace_count, len(fig.data)))

        for i in range(num_frames):
            factor = np.cos(2 * np.pi * i / num_frames) * scale
            frame_traces = get_deformed_traces(
                sections_elements,
                node_map,
                displacements,
                factor,
                deformed_color,
                interpolation_segments=interpolation_segments,
                include_customdata=False,
                include_hover=False,
            )
            frames.append(go.Frame(data=frame_traces, name=f'fr{i}', traces=indices_to_update))

        fig.frames = frames
        
        play_pause_buttons = [
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True, "mode": "immediate", "transition": {"duration": 0}, "repeat": True}]},
            {"label": "❚❚ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]},
            {"label": "■ Stop", "method": "animate", "args": [['fr0'], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}]}
        ]

        fig.update_layout(
            updatemenus=[{
                "type": "buttons", "showactive": False, "x": 0.5, "y": 0.05, "xanchor": "center", "yanchor": "bottom",
                "direction": "left", "pad": {"t": 0, "r": 10}, "buttons": play_pause_buttons,
                "bgcolor": "rgba(255, 255, 255, 0.9)" if not is_dark else "rgba(0, 0, 0, 0.9)",
                "font": {"color": "#000" if not is_dark else "#fff"}, "bordercolor": cfg['grid_color'], "borderwidth": 1
            }],
            sliders=[{"active": 0, "yanchor": "top", "xanchor": "left", "transition": {"duration": 0}, "pad": {"b": 10, "t": 50}, "len": 0.9, "x": 0.1, "y": 0, "steps": []}]
        )

    min_coords = np.array([float('inf')] * 3)
    max_coords = np.array([float('-inf')] * 3)
    found_any = False

    for trace in fig.data:
        if hasattr(trace, 'x') and trace.x is not None:
            pts_x = [float(v) for v in trace.x if v is not None]
            pts_y = [float(v) for v in trace.y if v is not None]
            pts_z = [float(v) for v in trace.z if v is not None]

            if pts_x:
                min_coords = np.minimum(min_coords, [min(pts_x), min(pts_y), min(pts_z)])
                max_coords = np.maximum(max_coords, [max(pts_x), max(pts_y), max(pts_z)])
                found_any = True

    if not found_any:
        min_coords, max_coords = np.array([-100.0]*3), np.array([100.0]*3)

    def get_ext_range(min_v, max_v):
        span = max_v - min_v
        margin = max(span, 100.0) * 0.4
        return [min_v - margin, max_v + margin]

    axis_style = dict(
        gridcolor='rgba(255, 255, 255, 0.08)' if is_dark else 'rgba(0, 0, 0, 0.06)',
        zerolinecolor='#888888' if is_dark else '#444444',
        nticks=6, tickformat='.0f', showgrid=True, zerolinewidth=2, gridwidth=1.0, showbackground=True, backgroundcolor='rgba(0,0,0,0)', showspikes=False
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=get_ext_range(min_coords[0], max_coords[0]), autorange=False, **axis_style, title_text='X (mm)'),
            yaxis=dict(range=get_ext_range(min_coords[1], max_coords[1]), autorange=False, **axis_style, title_text='Y (mm)'),
            zaxis=dict(range=get_ext_range(min_coords[2], max_coords[2]), autorange=False, **axis_style, title_text='Z (mm)'),
            aspectmode='data', camera=dict(eye=dict(x=1.8, y=1.8, z=1.4), projection=dict(type='perspective'))
        ),
        title_text=f"Análisis {scale}x{(' (Animación)' if animate else '')}"
    )

    return fig
