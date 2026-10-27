"""Offline Plotly 3D room viewer and static evaluation figures."""
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import trimesh
from .polycam import write_json


def load_mesh(path):
    scene = trimesh.load_scene(path)
    # Trimesh concatenation drops COLOR_0 on material-backed, untextured
    # primitives. Convert those visuals before applying scene transforms.
    for geometry in scene.geometry.values():
        attrs = getattr(geometry.visual, 'vertex_attributes', {})
        color = attrs.get('color')
        material = getattr(geometry.visual, 'material', None)
        if color is not None and getattr(material, 'baseColorTexture', None) is None:
            geometry.visual = trimesh.visual.ColorVisuals(vertex_colors=color)
    # to_mesh applies scene-node transforms before combining geometries.
    return scene.to_mesh()


def mesh_trace(path):
    mesh = load_mesh(path)
    # Vertex clustering for the viewer only. Geometry used by PnP/audit is untouched.
    _, inverse = np.unique(np.floor(mesh.vertices/.035).astype(int), axis=0, return_inverse=True)
    count = np.bincount(inverse)
    verts = np.stack([np.bincount(inverse, weights=mesh.vertices[:, d])/count for d in range(3)], axis=1)
    faces = inverse[mesh.faces]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    # Deduplicate triangles without sorting their vertex order. Sorting vertices
    # reverses some normals and breaks consistent lighting.
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(first)]
    colors = (mesh.visual.to_color() if hasattr(mesh.visual, 'to_color') else mesh.visual).vertex_colors[:, :3]
    colors = np.stack([np.bincount(inverse, weights=colors[:, d])/count for d in range(3)], axis=1).astype(int)
    # Explicit CSS colors avoid nested typed-array RGB serialization differences
    # between Plotly Python and JavaScript, which can leave meshes uncolored.
    real_colors = [f'rgb({r},{g},{b})' for r, g, b in colors]
    height = (verts[:, 1]-verts[:, 1].min())/max(np.ptp(verts[:, 1]), 1e-6)
    stops = np.array([0., .10, .40, .72, 1.])
    palette = np.array([[169,116,67],[206,166,106],[74,156,128],[83,137,178],[148,143,196]])
    levels = np.stack([np.interp(height, stops, palette[:, channel]) for channel in range(3)], axis=1).astype(int)
    level_colors = [f'rgb({r},{g},{b})' for r, g, b in levels]
    return go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], vertexcolor=real_colors,
        meta=dict(color_modes=dict(original=real_colors, levels=level_colors)),
        opacity=.90, name='GLB room (viewer simplification)', hoverinfo='skip',
        lighting=dict(ambient=.55, diffuse=.80, specular=.05, roughness=.85),
        lightposition=dict(x=100, y=200, z=150), flatshading=False)


def camera_trace(T, name, color, visible=True, length=.35, size=5):
    T = np.asarray(T)
    center = T[:3, 3]
    forward = center + T[:3, 2]*length
    xaxis = T[:3, 0]*.13
    yaxis = T[:3, 1]*.10
    corners = [forward-xaxis-yaxis, forward+xaxis-yaxis, forward+xaxis+yaxis, forward-xaxis+yaxis]
    points = []
    for c in corners:
        points += [center, c, [None]*3]
    for a, b in zip(corners, corners[1:]+corners[:1]):
        points += [a, b, [None]*3]
    arr = np.asarray(points, dtype=object)
    return go.Scatter3d(x=arr[:, 0], y=arr[:, 1], z=arr[:, 2], mode='lines',
        line=dict(color=color, width=size), name=name, visible=visible, hoverinfo='name')


def scene_html(capture, path, results=None, reference_ids=None):
    traces = []
    if capture.glb.is_file():
        traces.append(mesh_trace(capture.glb))
    ids = reference_ids if reference_ids is not None else capture.ids
    centers = np.array([capture.frames[id].T_world_cv[:3, 3] for id in ids])
    if len(centers):
        traces.append(go.Scatter3d(x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
            text=ids, mode='markers', marker=dict(size=3, color='#64748b'), name='Reference cameras',
            hovertemplate='%{text}<br>x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f}<extra></extra>'))
    if not results:
        for id in ids[::max(1, len(ids)//25)]:
            traces.append(camera_trace(capture.frames[id].T_world_cv, id, '#64748b', length=.16, size=2))
    base = len(traces)
    selections = []
    for j, r in enumerate(results or []):
        first = len(traces)
        if 'ground_truth' in r:
            traces.append(camera_trace(r['ground_truth']['T_world_camera_cv'], 'Ground truth', '#16a34a', j == 0))
        if r.get('success'):
            traces.append(camera_trace(r['pose']['T_world_camera_cv'], 'Estimated', '#ef4444', j == 0))
        label = f"{r.get('query_id', 'Query')}: "
        label += (f"{r['position_error_m']:.3f} m / {r['rotation_error_deg']:.2f} deg" if 'position_error_m' in r
                  else ('localized' if r.get('success') else 'failed'))
        selections.append((label, first, len(traces)))
    fig = go.Figure(traces)
    buttons = []
    for label, start, end in selections:
        buttons.append(dict(label=label, method='update', args=[
            {'visible': [i < base or start <= i < end for i in range(len(traces))]}, {'title': label}]))
    colors_menu = []
    if capture.glb.is_file():
        for label, mode in [('Original colors', 'original'), ('Height colors', 'levels')]:
            colors_menu.append(dict(label=label, method='restyle', args=[
                {'vertexcolor': [traces[0].meta['color_modes'][mode]]}, [0]]))
    menus = ([dict(buttons=buttons, x=0, y=1.06, xanchor='left')] if buttons else [])
    if colors_menu:
        menus.append(dict(buttons=colors_menu, x=1, y=1.06, xanchor='right'))
    fig.update_layout(title=selections[0][0] if selections else 'Reference cameras in GLB coordinates',
        scene=dict(xaxis_title='X (m)', yaxis_title='Y up (m)', zaxis_title='Z (m)', aspectmode='data',
                   camera=dict(up=dict(x=0, y=1, z=0), eye=dict(x=1.7, y=1.1, z=1.7))),
        margin=dict(l=0, r=0, b=0, t=80), height=850,
        updatemenus=menus,
        annotations=[dict(text='Drag to rotate | scroll to zoom | green: ground truth, red: estimate. '
                         'Original colors come from the scan. Height colors are display-only.',
                         x=0, y=0, xref='paper', yref='paper', showarrow=False)])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs=True, full_html=True)


def audit(capture, output, mesh_check=True):
    output = Path(output)
    if getattr(capture, 'dataset_format', '') == '7-scenes':
        report=dict(frames=len(capture.ids),reference_frames=len(capture.reference_ids),test_frames=len(capture.query_ids),
            dataset_format='7-scenes',alignment_mode='dataset',alignment_transform=capture.A.tolist(),
            convention='T_GLB_CV = world_to_glb @ T_original_world_CV',
            depth_registration='Reference depth rendered from aligned GLB into RGB pixels; raw depth PNG is not indexed as RGB depth.',
            warnings=['RGB intrinsics are approximate; official RGB and depth were not calibrated.',
                      'Supplied mesh may use all sequences. No test RGB, test depth or test pose enters reference feature construction.',
                      'Mesh-rendered depth is on the supplied mesh by construction; this is not an independent alignment audit.'])
        if mesh_check:
            f=capture.frames[capture.reference_ids[0]]
            yy,xx=np.mgrid[24:f.height-24:48,24:f.width-24:48]
            _,valid=capture.world_points(f,np.column_stack([xx.ravel(),yy.ravel()]))
            report['reference_mesh_depth_valid_fraction']=float(valid.mean())
        write_json(output/'inspection.json',report)
        scene_html(capture,output/'cameras.html',reference_ids=capture.reference_ids)
        return report
    report = dict(frames=len(capture.ids), alignment_mode=capture.alignment,
        alignment_transform=capture.raw_alignment.tolist(), depth_scale=capture.depth_scale,
        convention='T_world_camera_cv = alignment_transform @ T_polycam_camera @ diag(1,-1,-1,1)',
        depth_registration='Scaled same-frame sensor depth on corrected RGB; approximate undistortion registration.',
        warnings=['Polycam optimized poses are internal reference poses, not independent surveyed ground truth.'])
    if mesh_check and capture.glb.is_file():
        mesh = load_mesh(capture.glb)
        points = []
        for id in capture.ids[::max(1, len(capture.ids)//10)]:
            f = capture.frames[id]
            y, x = np.mgrid[24:f.height-24:48, 24:f.width-24:48]
            xyz, valid = capture.world_points(f, np.column_stack([x.ravel(), y.ravel()]))
            points.append(xyz[valid])
        pw = np.concatenate(points)
        if not len(pw):
            raise ValueError('No valid depth points for mesh audit')
        inv = np.linalg.inv(capture.A)
        source = pw @ inv[:3, :3].T + inv[:3, 3]
        comparisons = {}
        for label, A in [('forward', capture.raw_alignment), ('identity', np.eye(4)),
                         ('inverse', np.linalg.inv(capture.raw_alignment))]:
            query = source @ A[:3, :3].T + A[:3, 3]
            # Batch to bound memory with high triangle-count meshes.
            distances = []
            for batch in np.array_split(query, max(1, int(np.ceil(len(query)/300)))):
                distances.append(trimesh.proximity.closest_point(mesh, batch)[1])
            d = np.concatenate(distances)
            comparisons[label] = dict(points=len(d), median_distance_m=float(np.median(d)),
                p90_distance_m=float(np.percentile(d, 90)), fraction_within_10cm=float(np.mean(d <= .1)))
        report['mesh_depth_consistency'] = comparisons
        best = min(comparisons, key=lambda k: comparisons[k]['median_distance_m'])
        report['lowest_residual_alignment'] = best
        report['alignment_matches_lowest_residual'] = best == capture.alignment
        if best != capture.alignment:
            report['warnings'].append('Selected alignment does not have the lowest mesh residual. Inspect the viewer before proceeding.')
    write_json(output / 'inspection.json', report)
    scene_html(capture, output / 'cameras.html')
    return report


def evaluation_plot(capture, results, path):
    # Pillow plot avoids a matplotlib dependency for the command-line program.
    from PIL import Image, ImageDraw, ImageFont
    width, height = 1200, 680
    image = Image.new('RGB', (width, height), '#f8fafc')
    draw = ImageDraw.Draw(image)
    draw.font = ImageFont.load_default(size=17)
    draw.text((35, 20), 'Photo localization | meters | green: ground truth | red: estimated', fill='#0f172a')
    all_pos = np.array([f.T_world_cv[[0, 2], 3] for f in capture.frames.values()])
    good = [r for r in results if r.get('success')]
    if good:
        all_pos = np.concatenate([all_pos, np.array([np.array(r['pose']['position_m'])[[0, 2]] for r in good])])
    lo, hi = all_pos.min(axis=0)-.3, all_pos.max(axis=0)+.3
    scale = min(650/(hi[0]-lo[0]), 500/(hi[1]-lo[1]))
    def screen(p):
        a = (np.array(p)[[0, 2]]-lo)*scale
        return (int(60+a[0]), int(600-a[1]))
    for x in np.arange(np.ceil(lo[0]*2)/2, hi[0]+.01, .5):
        sx = int(60+(x-lo[0])*scale)
        draw.line([(sx, 80), (sx, 600)], fill='#e2e8f0')
        draw.text((sx-12, 610), f'{x:.1f}', fill='#64748b')
    for z in np.arange(np.ceil(lo[1]*2)/2, hi[1]+.01, .5):
        sy = int(600-(z-lo[1])*scale)
        draw.line([(60, sy), (750, sy)], fill='#e2e8f0')
        draw.text((5, sy-8), f'{z:.1f}', fill='#64748b')
    draw.text((360, 645), 'X (m)', fill='#334155')
    draw.text((15, 55), 'Z (m)', fill='#334155')
    for f in capture.frames.values():
        x, y = screen(f.T_world_cv[:3, 3]);draw.ellipse((x-2, y-2, x+2, y+2), fill='#94a3b8')
    for r in results:
        a = screen(r['ground_truth']['position_m']);draw.ellipse((a[0]-4,a[1]-4,a[0]+4,a[1]+4), fill='#16a34a')
        if r.get('success'):
            b = screen(r['pose']['position_m']);draw.line([a, b], fill='#ef4444', width=2)
            draw.ellipse((b[0]-3,b[1]-3,b[0]+3,b[1]+3), fill='#ef4444')
        else:
            draw.text((a[0]+5,a[1]), 'FAIL', fill='#b45309')
    draw.text((800, 70), f'Queries: {len(results)}', fill='#0f172a')
    draw.text((800, 100), f'Pose returned: {len(good)}/{len(results)}', fill='#0f172a')
    if good:
        draw.text((800, 140), f"Median position: {np.median([r['position_error_m'] for r in good]):.3f} m", fill='#0f172a')
        draw.text((800, 170), f"Median rotation: {np.median([r['rotation_error_deg'] for r in good]):.2f} deg", fill='#0f172a')
    draw.text((800, 240), 'Sampled official test sequences.' if getattr(capture, 'dataset_format', '') == '7-scenes' else 'Internal scan consistency test.', fill='#475569')
    draw.text((800, 270), 'Independent phone photos not tested.', fill='#475569')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
