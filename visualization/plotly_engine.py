import plotly.graph_objects as go
import numpy as np


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
        mid_x, mid_y, mid_z, mid_text = [], [], [], []

        section_info = sections.get(sid, {})
        section_name = section_info.get('name', f"Sec {sid}")

        for el in els:
            n1 = node_map.get(el['node_ids'][0])
            n2 = node_map.get(el['node_ids'][1])
            if n1 and n2:
                c1, c2 = n1['coords'], n2['coords']
                ex.extend([c1[0], c2[0], None])
                ey.extend([c1[1], c2[1], None])
                ez.extend([c1[2], c2[2], None])

                mid_x.append((c1[0] + c2[0]) / 2)
                mid_y.append((c1[1] + c2[1]) / 2)
                mid_z.append((c1[2] + c2[2]) / 2)

                mat_id = el.get('material_id')
                mat_name = materials.get(mat_id, {}).get('name', f"M{mat_id}")
                mid_text.append(f"<b>Elemento {el['id']}</b><br>Sección: {section_name}<br>Material: {mat_name}")

        if ex:
            color = color_map.get(sid, palette[0])
            fig.add_trace(go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode='lines',
                line=dict(color=color_map[sid], width=5),
                name=section_name,
                hoverinfo='text',
                text=mid_text,
                legendgroup=f'sec_{sid}',
                showlegend=True
            ))
            fig.add_trace(go.Scatter3d(
                x=mid_x, y=mid_y, z=mid_z,
                mode='markers',
                marker=dict(size=2, opacity=0, color=color),
                text=[f"E{e['id']}" for e in els],
                hovertext=mid_text,
                hoverinfo='text',
                showlegend=False,
                legendgroup=f"group{sid}"
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
            hovertext=[f"<b>Nodo {n['id']}</b><br>({n['coords'][0]:.2f}, {n['coords'][1]:.2f}, {
                n['coords'][2]:.2f})" for n in nodes],
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
        center = (min_coords + max_coords) / 2
        
        def get_range(min_v, max_v):
            span = max_v - min_v
            margin = span * 0.2 if span > 1e-6 else 100.0 * 0.4
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


def get_deformed_traces(sections_elements, node_map, displacements, scale, color):
    traces = []
    
    def clean(v):
        if v is None: return None
        return float(v) if np.isfinite(v) else 0.0

    for sid, els in sections_elements.items():
        bx, by, bz = [], [], []
        dx, dy, dz = [], [], []
        for el in els:
            n1, n2 = node_map.get(el['node_ids'][0]), node_map.get(el['node_ids'][1])
            if not (n1 and n2): continue

            nid1, nid2 = el['node_ids'][0], el['node_ids'][1]
            d1 = displacements.get(nid1) or displacements.get(str(nid1), [0]*6)
            d2 = displacements.get(nid2) or displacements.get(str(nid2), [0]*6)
            d1, d2 = (list(d1) + [0]*6)[:6], (list(d2) + [0]*6)[:6]

            try:
                px_b, py_b, pz_b, px_d, py_d, pz_d = interpolate_beam(n1['coords'], n2['coords'], d1, d2)
                bx.extend(px_b + [None]); by.extend(py_b + [None]); bz.extend(pz_b + [None])
                dx.extend(px_d + [None]); dy.extend(py_d + [None]); dz.extend(pz_d + [None])
            except:
                bx.extend([float(n1['coords'][0]), float(n2['coords'][0]), None])
                by.extend([float(n1['coords'][1]), float(n2['coords'][1]), None])
                bz.extend([float(n1['coords'][2]), float(n2['coords'][2]), None])
                dx.extend([float(d1[0]*1000), float(d2[0]*1000), None])
                dy.extend([float(d1[1]*1000), float(d2[1]*1000), None])
                dz.extend([float(d1[2]*1000), float(d2[2]*1000), None])

        final_x = [clean(bx[i] + dx[i]*scale) if bx[i] is not None else None for i in range(len(bx))]
        final_y = [clean(by[i] + dy[i]*scale) if by[i] is not None else None for i in range(len(by))]
        final_z = [clean(bz[i] + dz[i]*scale) if bz[i] is not None else None for i in range(len(bz))]

        cdata = []
        for i in range(len(bx)):
            if bx[i] is None:
                cdata.append([None] * 6)
            else:
                cdata.append([clean(bx[i]), clean(by[i]), clean(bz[i]), clean(dx[i]), clean(dy[i]), clean(dz[i])])

        traces.append(go.Scatter3d(
            x=final_x, y=final_y, z=final_z,
            mode='lines',
            line=dict(color=color, width=4),
            name=f"Deformada {sid}",
            customdata=cdata,
            hovertemplate="Deformada<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>"
        ))
    return traces


def generate_results_figure(structure_data, displacements, scale=1.0, theme="dark", animate=False):
    fig = generate_structure_figure(structure_data, theme=theme)
    cfg = get_theme_config(theme)
    is_dark = cfg['is_dark']

    ghost_opacity = 0.15
    deformed_color = '#ff00ff'

    for trace in fig.data:
        if isinstance(trace, go.Scatter3d):
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

    base_traces = get_deformed_traces(sections_elements, node_map, displacements, scale, deformed_color)
    if base_traces: base_traces[0].showlegend = True
    for t in base_traces: fig.add_trace(t)

    if animate:
        frames = []
        num_frames = 20
        static_trace_count = len(fig.data) - len(base_traces)
        indices_to_update = list(range(static_trace_count, len(fig.data)))

        for i in range(num_frames):
            factor = np.cos(2 * np.pi * i / num_frames) * scale
            frame_traces = get_deformed_traces(sections_elements, node_map, displacements, factor, deformed_color)
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


