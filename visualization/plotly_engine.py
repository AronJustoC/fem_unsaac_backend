import plotly.graph_objects as go
import numpy as np


def generate_structure_figure(structure_data, theme="dark"):
    """
    Genera una figura de Plotly para la estructura con un estilo profesional
    y adaptativo (Claro/Oscuro).
    """
    # Convertir coordenadas de metros a milímetros para visualización
    nodes = []
    for n in structure_data.get('nodes', []):
        nodes.append({
            **n,
            'coords': [n['coords'][0] * 1000, n['coords'][1] * 1000, n['coords'][2] * 1000]
        })

    elements = structure_data.get('elements', [])
    materials = {m['id']: m for m in structure_data.get('materials', [])}
    sections = {s['id']: s for s in structure_data.get('sections', [])}
    node_map = {n['id']: n for n in nodes}

    is_dark = theme == "dark"

    # --- CONFIGURACIÓN DE TEMA ---
    if is_dark:
        # Tema Oscuro (Professional Dark / SAP2000 Style)
        bg_color = '#000000'      # SAP2000 Black
        grid_color = 'rgba(255, 255, 255, 0.15)'
        axis_color = '#888888'
        text_color = '#ffffff'
        label_color = '#cccccc'
        node_fill = '#000000'
        node_border = '#ffffff'
        # Paleta SAP2000 Estricta
        palette = ['#1e90ff', '#00ff00', '#ffff00', '#ff00ff',
                   '#00ffff', '#ffa500', '#ffc0cb', '#ffffff']
        grid_bg_color = 'rgba(255,255,255,0.02)'
    else:
        # Tema Claro (Professional Light / SAP2000 White Style)
        bg_color = '#ffffff'
        grid_color = 'rgba(0, 0, 0, 0.1)'
        axis_color = '#444444'
        text_color = '#000000'
        label_color = '#333333'
        node_fill = '#ffffff'
        node_border = '#000000'
        # Paleta SAP2000 Estricta (con ajustes de contraste para fondo claro)
        palette = ['#1e90ff', '#008000', '#c2a100', '#db2777',
                   '#009999', '#ea580c', '#db2777', '#000000']
        grid_bg_color = 'rgba(0,0,0,0.02)'

    font_family = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

    fig = go.Figure()

    # --- GRID PERSONALIZADO ---
    # Dibujamos un "suelo" de referencia si hay nodos
    if nodes:
        all_coords = np.array([n['coords'] for n in nodes])
        if len(all_coords) > 0:
            min_x, max_x = np.min(all_coords[:, 0]), np.max(all_coords[:, 0])
            min_y, max_y = np.min(all_coords[:, 1]), np.max(all_coords[:, 1])

            # Margen del 20%
            margin_x = max(max_x - min_x, 5.0) * 0.2
            margin_y = max(max_y - min_y, 5.0) * 0.2
            pass

    # --- ELEMENTOS ---
    # Agrupar por sección para asignar colores consistentes
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
        mid_x, mid_y, mid_z, mid_text = [], [], [], []

        section_info = sections.get(sid, {})
        section_name = section_info.get('name', f"Sec {sid}")

        for el in els:
            n1 = node_map.get(el['node_ids'][0])
            n2 = node_map.get(el['node_ids'][1])
            if n1 and n2:
                c1, c2 = n1['coords'], n2['coords']
                # Construimos líneas con 'None' para separarlas
                ex.extend([c1[0], c2[0], None])
                ey.extend([c1[1], c2[1], None])
                ez.extend([c1[2], c2[2], None])

                # Puntos medios para interactividad
                mid_x.append((c1[0] + c2[0]) / 2)
                mid_y.append((c1[1] + c2[1]) / 2)
                mid_z.append((c1[2] + c2[2]) / 2)

                mat_id = el.get('material_id')
                mat_name = materials.get(mat_id, {}).get('name', f"M{mat_id}")
                mid_text.append(f"<b>Elemento {
                                el['id']}</b><br>Sección: {section_name}<br>Material: {mat_name}")

        if ex:
            color = color_map.get(sid, palette[0])
            # Trazo de líneas (Elementos)
            fig.add_trace(go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode='lines',
                # Líneas más definidas
                line=dict(color=color_map[sid], width=5),
                name=section_name,
                hoverinfo='text',
                text=mid_text,
                legendgroup=f'sec_{sid}',
                showlegend=True
            ))
            # Puntos invisibles para Hover en elementos
            fig.add_trace(go.Scatter3d(
                x=mid_x, y=mid_y, z=mid_z,
                mode='markers',
                marker=dict(size=2, opacity=0, color=color),  # Casi invisible
                text=[f"E{e['id']}" for e in els],
                hovertext=mid_text,
                hoverinfo='text',
                showlegend=False,
                legendgroup=f"group{sid}"
            ))

    # --- NODOS ---
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
            hovertext=[f"<b>Nodo {n['id']}</b><br>({n['coords'][0]:.2f}, {n['coords'][1]:.2f}, {
                n['coords'][2]:.2f})" for n in nodes],
            hoverinfo='text',
            name='Nodos'
        ))

    # --- RESTRICCIONES (APOYOS) ---
    restraints = structure_data.get('restraints', {})

    # Definición de estilos de apoyo
    support_styles = {
        'fixed': {
            'symbol': 'square',
            'color': '#00ff00',  # SAP2000 Green
            'name': 'Empotrado'
        },
        'pinned': {
            'symbol': 'triangle-up',
            'color': '#00ff00',  # SAP2000 Green
            'name': 'Articulado'
        },
        'other': {
            'symbol': 'circle',
            'color': '#00ff00',  # SAP2000 Green
            'name': 'Restricción'
        }
    }

    support_data = {k: {'x': [], 'y': [], 'z': [], 'text': []}
                    for k in support_styles}

    for nid_str, dofs in restraints.items():
        node = node_map.get(int(nid_str))
        if not node:
            continue

        dofs_set = set(dofs)
        # Lógica simple para clasificar apoyos
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
        support_data[k]['text'].append(
            f"Nodo {nid_str}<br>{', '.join(unique_dofs)}")

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

    # --- CARGAS ---
    loads = structure_data.get('loads', [])
    if loads:
        lx, ly, lz, lu, lv, lw = [], [], [], [], [], []
        forces = [np.linalg.norm([l.get('fx', 0), l.get(
            'fy', 0), l.get('fz', 0)]) for l in loads]
        if forces:
            max_f = max(forces + [1.0])
            sizeref = max_f / 0.5  # Ajuste de escala de conos

            # --- CÁLCULO DE ESCALA VISUAL ---
            # Determinamos el tamaño de la estructura para escalar las flechas
            all_coords = np.array([n['coords'] for n in nodes])
            if len(all_coords) > 0:
                min_c = np.min(all_coords, axis=0)
                max_c = np.max(all_coords, axis=0)
                max_dim = np.max(max_c - min_c)
                if max_dim == 0:
                    max_dim = 100.0
            else:
                max_dim = 100.0

            # Longitud máxima de la flecha
            arrow_scale = max(max_dim * 0.15, 80.0)

            shaft_x, shaft_y, shaft_z = [], [], []
            head_x, head_y, head_z = [], [], []
            head_u, head_v, head_w = [], [], []

            for load in loads:
                node = node_map.get(load['node_id'])
                if not node:
                    continue

                # Vector fuerza
                fx, fy, fz = load.get('fx', 0), load.get(
                    'fy', 0), load.get('fz', 0)
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

    # --- LAYOUT CONFIG ---
    rx, ry, rz = None, None, None
    if nodes:
        all_coords = np.array([n['coords'] for n in nodes])
        min_coords = np.min(all_coords, axis=0)
        max_coords = np.max(all_coords, axis=0)
        center = (min_coords + max_coords) / 2
        size = np.max(max_coords - min_coords)
        if size == 0:
            size = 1000.0
        camera_distance = max(size * 1.5, 2.0)

        # Smart Ranges para evitar colapso de ejes en 2D/1D y dar aire
        def get_ux_range(min_v, max_v, center_v):
            span = max_v - min_v
            if span < 1e-6:
                span = 100.0  # Rango base si no hay dimensión
                margin = span * 0.4  # 40% buffer para colapsados
            else:
                margin = span * 0.2  # 20% margen para otros
            return [min_v - margin, max_v + margin]

        rx = get_ux_range(min_coords[0], max_coords[0], center[0])
        ry = get_ux_range(min_coords[1], max_coords[1], center[1])
        rz = get_ux_range(min_coords[2], max_coords[2], center[2])
    else:
        camera_distance = 10.0
        rx = [-100, 100]
        ry = [-100, 100]
        rz = [-100, 100]

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
        showspikes=False,
        spikecolor=axis_color,
        spikethickness=1
    )

    fig.update_layout(
        template='plotly_dark' if is_dark else 'plotly_white',
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(family=font_family, color=text_color),
        scene=dict(
            xaxis=dict(**axis_config, title=dict(text='X (mm)'),
                       range=rx, autorange=False),
            yaxis=dict(**axis_config, title=dict(text='Y (mm)'),
                       range=ry, autorange=False),
            zaxis=dict(**axis_config, title=dict(text='Z (mm)'),
                       range=rz, autorange=False),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.4),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                projection=dict(type='perspective')
            ),
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(
            yanchor="top", y=0.98, xanchor="right", x=0.98,
            bgcolor=bg_color,
            bordercolor=grid_color,
            borderwidth=1,
            font=dict(size=11, family=font_family, color=text_color)
        ),
        hoverlabel=dict(
            bgcolor=bg_color,
            font_size=11,
            font_family=font_family,
            bordercolor=grid_color
        ),
        hovermode='closest',
        dragmode='orbit',
        scene_camera_projection_type='perspective'
    )

    return fig


def generate_results_figure(structure_data, displacements, scale=1.0, theme="dark", animate=False):
    """
    Genera la figura de resultados (deformada) basándose en la figura base.
    Si animate=True, genera frames para animación armónica.
    """
    fig = generate_structure_figure(structure_data, theme=theme)
    is_dark = theme == "dark"

    # Colores
    ghost_opacity = 0.15
    deformed_color = '#ff00ff'  # SAP2000 Magenta

    if is_dark:
        bg_color = '#000000'
        grid_color = 'rgba(255, 255, 255, 0.15)'
        axis_color = '#888888'
        text_color = '#ffffff'
    else:
        bg_color = '#ffffff'
        grid_color = 'rgba(0, 0, 0, 0.1)'
        axis_color = '#444444'
        text_color = '#000000'

    # Atenuar la estructura original
    for trace in fig.data:
        if isinstance(trace, go.Scatter3d):
            mode = getattr(trace, 'mode', '') or ''
            if mode == 'lines':
                # Safely get color
                orig_color = '#888888'
                try:
                    if hasattr(trace, 'line') and trace.line:
                        c = getattr(trace.line, 'color', None)
                        if c:
                            orig_color = c
                except:
                    pass

                trace.update(
                    line=dict(width=2, color=orig_color), opacity=ghost_opacity)
            elif 'markers' in mode:
                trace.update(opacity=ghost_opacity)
        elif isinstance(trace, go.Cone):
            trace.update(opacity=ghost_opacity)

    # Preparar datos comunes
    nodes = []
    for n in structure_data.get('nodes', []):
        nodes.append({
            **n,
            'coords': [n['coords'][0] * 1000, n['coords'][1] * 1000, n['coords'][2] * 1000]
        })
    elements = structure_data.get('elements', [])
    node_map = {n['id']: n for n in nodes}
    sections = {s['id']: s for s in structure_data.get('sections', [])}
    sections_elements = {}
    for el in elements:
        sections_elements.setdefault(el.get('section_id'), []).append(el)

    # Función auxiliar para interpolación
    def interpolate_beam(p1, p2, d1, d2, current_scale, segments=4):
        # p1, p2: Coordenadas originales (mm)
        # d1, d2: Desplazamientos [ux, uy, uz, rx, ry, rz] (m, rad)

        # 1. Puntos finales deformados
        disp1 = np.array(d1[:3]) * 1000 * current_scale
        disp2 = np.array(d2[:3]) * 1000 * current_scale
        P1_def = np.array(p1) + disp1
        P2_def = np.array(p2) + disp2

        # 2. Vectores tangentes
        v_orig = np.array(p2) - np.array(p1)
        L = np.linalg.norm(v_orig)
        if L < 1e-6:
            return [], [], []

        theta1 = np.array(d1[3:]) * current_scale
        theta2 = np.array(d2[3:]) * current_scale

        v_unit = v_orig / L
        t1_def = v_unit + np.cross(theta1, v_unit)
        t2_def = v_unit + np.cross(theta2, v_unit)
        T1 = t1_def * L
        T2 = t2_def * L

        # 3. Generar puntos
        points_x, points_y, points_z = [], [], []
        t_vals = np.linspace(0, 1, segments+1)
        for t in t_vals:
            t2 = t * t
            t3 = t2 * t
            h00 = 2*t3 - 3*t2 + 1
            h10 = t3 - 2*t2 + t
            h01 = -2*t3 + 3*t2
            h11 = t3 - t2
            P = h00 * P1_def + h10 * T1 + h01 * P2_def + h11 * T2
            points_x.append(float(P[0]))
            points_y.append(float(P[1]))
            points_z.append(float(P[2]))
        return points_x, points_y, points_z

    def get_deformed_traces(current_scale):
        traces = []
        for sid, els in sections_elements.items():
            dx, dy, dz = [], [], []
            for el in els:
                n1 = node_map.get(el['node_ids'][0])
                n2 = node_map.get(el['node_ids'][1])
                nid1, nid2 = el['node_ids'][0], el['node_ids'][1]

                d1 = displacements.get(nid1)
                if d1 is None:
                    d1 = displacements.get(str(nid1), [0]*6)
                d2 = displacements.get(nid2)
                if d2 is None:
                    d2 = displacements.get(str(nid2), [0]*6)
                d1 = (list(d1) + [0]*6)[:6]
                d2 = (list(d2) + [0]*6)[:6]

                if n1 and n2:
                    try:
                        px, py, pz = interpolate_beam(
                            n1['coords'], n2['coords'], d1, d2, current_scale)
                        dx.extend(px + [None])
                        dy.extend(py + [None])
                        dz.extend(pz + [None])
                    except:
                        # Fallback simple
                        p1_def = [n1['coords'][i] + d1[i] *
                                  1000*current_scale for i in range(3)]
                        p2_def = [n2['coords'][i] + d2[i] *
                                  1000*current_scale for i in range(3)]
                        dx.extend([float(p1_def[0]), float(p2_def[0]), None])
                        dy.extend([float(p1_def[1]), float(p2_def[1]), None])
                        dz.extend([float(p1_def[2]), float(p2_def[2]), None])

            # Sanitizar
            def clean_val(v):
                if v is None:
                    return None
                try:
                    if np.isnan(v) or np.isinf(v):
                        return 0.0
                    return float(v)
                except:
                    return 0.0

            dx = [clean_val(v) for v in dx]
            dy = [clean_val(v) for v in dy]
            dz = [clean_val(v) for v in dz]

            if dx:
                traces.append(go.Scatter3d(
                    x=dx, y=dy, z=dz, mode='lines',
                    line=dict(color=deformed_color, width=5),
                    name=f'Deformada {sections[sid].get(
                        "name", f"Seccion {sid}")}' if sid in sections else 'Deformada',
                    showlegend=True if sid == list(
                        sections_elements.keys())[0] else False
                ))
        return traces

    # Agregar trazas base (estáticas)
    base_traces = get_deformed_traces(scale)
    base_traces[0].showlegend = True
    for t in base_traces:
        fig.add_trace(t)

    # Si es animación, generar frames
    if animate:
        frames = []
        num_frames = 20
        # Indices de trazas a actualizar (las últimas que agregamos)
        data_list = list(fig.data)
        static_trace_count = len(data_list) - len(base_traces)
        indices_to_update = list(range(static_trace_count, len(data_list)))

        for i in range(num_frames):
            phase = i / num_frames
            factor = np.cos(2 * np.pi * phase) * scale
            frame_traces = get_deformed_traces(factor)

            frames.append(go.Frame(
                data=frame_traces,
                name=f'fr{i}',
                traces=indices_to_update
            ))

        fig.frames = frames

        # Botones de Play/Pause y configuración de animación
        fig.update_layout(
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "x": 0.5, "y": 0.05, "xanchor": "center", "yanchor": "bottom",  # Centrado abajo
                "direction": "left",
                "pad": {"t": 0, "r": 10},
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                                "transition": {"duration": 0},
                                "repeat": True
                            }
                        ]

                    },
                    {
                        "label": "❚❚ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ]
                    },
                    {
                        "label": "■ Stop",
                        "method": "animate",
                        "args": [
                            ['fr0'],  # Volver al frame 0
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ]
                    }
                ],
                "bgcolor": "rgba(255, 255, 255, 0.9)" if not is_dark else "rgba(0, 0, 0, 0.9)",
                "font": {"color": "#000" if not is_dark else "#fff"},
                "bordercolor": grid_color,
                "borderwidth": 1
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Fase:",
                    "visible": False,
                    "xanchor": "right"
                },
                "transition": {"duration": 0},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": []  # Slider visual opcional, lo ocultamos para limpieza
            }]
        )

    # --- LAYOUT CONFIG & FIXED RANGES ---
    # Encontramos la extensión real de TODAS las trazas (original + deformada)
    # Esto asegura que la deformada interpolada nunca se corte.
    min_coords = np.array([float('inf')] * 3)
    max_coords = np.array([float('-inf')] * 3)
    found_any = False

    for trace in fig.data:
        # Extraer puntos x, y, z de la traza (Scatter3d)
        if hasattr(trace, 'x') and trace.x is not None:
            # Filtrar valores None (separadores) y convertir a float
            pts_x = [float(v) for v in trace.x if v is not None]
            pts_y = [float(v) for v in trace.y if v is not None]
            pts_z = [float(v) for v in trace.z if v is not None]

            if pts_x:
                min_coords[0] = min(min_coords[0], min(pts_x))
                max_coords[0] = max(max_coords[0], max(pts_x))
                min_coords[1] = min(min_coords[1], min(pts_y))
                max_coords[1] = max(max_coords[1], max(pts_y))
                min_coords[2] = min(min_coords[2], min(pts_z))
                max_coords[2] = max(max_coords[2], max(pts_z))
                found_any = True

    if not found_any:
        min_coords = np.array([-100, -100, -100])
        max_coords = np.array([100, 100, 100])

    def get_extended_range(min_v, max_v):
        span = max_v - min_v
        if span < 1e-4:
            span = 100.0  # Rango base si no hay dimensión

        # Buffer del 40% para asegurar que todo entre con aire
        margin = span * 0.4
        return [min_v - margin, max_v + margin]

    rx = get_extended_range(min_coords[0], max_coords[0])
    ry = get_extended_range(min_v=min_coords[1], max_v=max_coords[1])
    rz = get_extended_range(min_coords[2], max_coords[2])

    # Colores técnicos para los dos modos (Día/Noche)
    grid_col = 'rgba(255, 255, 255, 0.08)' if is_dark else 'rgba(0, 0, 0, 0.06)'
    zero_col = '#444444' if not is_dark else '#888888'

    axis_style = dict(
        gridcolor=grid_col,
        zerolinecolor=zero_col,
        nticks=6,
        tickformat='.0f',
        showgrid=True,
        zerolinewidth=2,
        gridwidth=1.0,
        showbackground=True,
        backgroundcolor='rgba(0,0,0,0)',
        showspikes=False
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=rx, autorange=False, **
                       axis_style, title_text='X (mm)'),
            yaxis=dict(range=ry, autorange=False, **
                       axis_style, title_text='Y (mm)'),
            zaxis=dict(range=rz, autorange=False, **
                       axis_style, title_text='Z (mm)'),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.4),
                projection=dict(type='perspective')
            )
        )
    )

    title_suffix = " (Animación)" if animate else ""
    fig.update_layout(title_text=f"Análisis {scale}x{title_suffix}")

    # Detectar predominancia (Heurística: Rotación vs Traslación)
    u_vals = [v for v in displacements.values() if v is not None]
    if u_vals and animate:
        arr = np.array(u_vals)
        norm_t = np.mean(np.linalg.norm(arr[:, :3], axis=1))
        norm_r = np.mean(np.linalg.norm(arr[:, 3:6], axis=1))

        # Factor de escala empírico para comparar m con rad
        # (Una rotación de 1 rad es enorme, una traslación de 1m es enorme)
        # Usamos ratio 1:1 para simplificar, o 1:0.5
        is_rot = norm_r > norm_t * 0.5
        pred = "ROTACIONAL / TORSIÓN" if is_rot else "TRASLACIONAL"

        fig.add_annotation(
            text=f"MODO {pred}",
            xref="paper", yref="paper",
            x=0.05, y=0.95, showarrow=False,
            font=dict(size=12, color='#ffffff' if is_dark else '#000000'),
            bgcolor=deformed_color,
            borderpad=4, opacity=0.8
        )

    return fig
