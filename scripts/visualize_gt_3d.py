#!/usr/bin/env python3
"""
Visualize ground-truth 3D scene graph sequences from /media/gao/gao_ssd/3D Scene.

GT JSON format (sg_frame_XXXXX.json):
  {
    "scene":     str,
    "frame":     str,
    "frame_idx": int,
    "nodes": [
      {
        "node_id":     int,
        "label":       str,
        "center":      [x, y, z],
        "orientation": [w, x, y, z],   # unit quaternion, Y-up world
        "size":        {"L": float, "W": float, "H": float}
      },
      ...
    ]
  }
  (no edges)

Size convention: L → object-local X, W → object-local Z, H → Y (height).

Usage (from repo root):
    conda run -n commonscenes python scripts/visualize_gt_3d.py \
        [--sg_dir "/media/gao/gao_ssd/3D Scene/1_1"] \
        [--obj_dir "/media/gao/gao_ssd/3D objects"]

Keys:
    Space / Right arrow : next frame
    Left arrow          : previous frame
    Q                   : quit
"""

import argparse
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import open3d as o3d
import trimesh

# ── colour palette ─────────────────────────────────────────────────────────────
LABEL_COLOR = {
    'cabinet':           [0.70, 0.50, 0.30],
    'beanbag':           [0.90, 0.40, 0.40],
    'chair':             [0.40, 0.65, 0.90],
    'sofa':              [0.55, 0.40, 0.80],
    'table':             [0.50, 0.75, 0.50],
    'tv_stand':          [0.80, 0.80, 0.30],
    'fridge':            [0.60, 0.85, 0.85],
    'refrigerator':      [0.60, 0.85, 0.85],
    'bike':              [0.85, 0.60, 0.20],
    'indoor_plant':      [0.30, 0.70, 0.30],
    'tv_screen':         [0.85, 0.90, 0.95],
    'kitchen_counter':   [0.75, 0.65, 0.55],
    'kitchen_cupboards': [0.75, 0.65, 0.55],
    'chest_of_drawers':  [0.70, 0.55, 0.35],
    'wall_cabinet':      [0.70, 0.55, 0.35],
    'clothes_hanger':    [0.80, 0.70, 0.60],
    'doors':             [0.65, 0.45, 0.25],
}

LABEL_ALIAS = {
    'chestOfDrawers':   'chest_of_drawers',
    'kitchen_cupboard': 'kitchen_cupboards',
    'kitchenCupboard':  'kitchen_cupboards',
    'tvstand':          'tv_stand',
}

DEFAULT_LABEL_COLOR = [0.75, 0.75, 0.75]

URDF_FILES = {
    'chest_of_drawers':  'chestOfDrawers_01.urdf',
    'kitchen_counter':   'kitchen_counter.urdf',
    'kitchen_cupboards': 'kitchenCupboard_01.urdf',
    'fridge':            'fridge.urdf',
}


# ── quaternion helpers ─────────────────────────────────────────────────────────

def quat_to_mat3(wxyz):
    """Full 3×3 rotation matrix from unit quaternion [w, x, y, z]."""
    w, x, y, z = wxyz
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


def quat_yaw(wxyz) -> float:
    """Extract yaw (rotation around Y axis) from Y-up quaternion [w, x, y, z]."""
    w, x, y, z = wxyz
    return float(np.arctan2(2.0 * (w * y + x * z),
                             1.0 - 2.0 * (x * x + y * y)))


def quat_zup_yaw(wxyz) -> float:
    """Extract yaw from a Z-up quaternion [w, x, y, z] and return as Y-up yaw angle.

    URDF articulated-object quaternions in the GT data use the Habitat simulator's
    Z-up convention (Z is vertical). The Z-up yaw formula uses (w*z + x*y) terms.
    We return this angle and then apply it as a yaw around the Y-up axis.
    """
    w, x, y, z = wxyz
    return float(np.arctan2(2.0 * (w * z + x * y),
                             1.0 - 2.0 * (y * y + z * z)))


def yaw_mat3(yaw: float) -> np.ndarray:
    """3×3 rotation matrix for yaw (radians) around Y axis."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


# ── mesh cache & loading ───────────────────────────────────────────────────────
_mesh_cache: dict = {}   # glb_path → (trimesh.Trimesh, colors_or_None)


def _extract_vertex_colors(parts: list):
    color_chunks = []
    for part in parts:
        try:
            raw = part.visual.to_color().vertex_colors
            if raw.ndim == 1:
                # uniform material color (single RGBA) — broadcast to all vertices
                rgb = raw[:3].astype(np.float64) / 255.0
                vc  = np.tile(rgb, (len(part.vertices), 1))
            else:
                vc = raw[:, :3].astype(np.float64) / 255.0
            color_chunks.append(vc)
        except Exception:
            return None
    return np.concatenate(color_chunks, axis=0) if color_chunks else None


def load_mesh(obj_dir: str, label: str, model_name=None):
    """Return (trimesh.Trimesh copy, vertex_colors_or_None).

    model_name: stem of the GLB file (without .glb) as recorded in the JSON.
    Falls back to the first alphabetical GLB if model_name is absent or not found.
    """
    cat_dir   = os.path.join(obj_dir, label)
    glb_files = sorted(glob.glob(os.path.join(cat_dir, '*.glb'))) if os.path.isdir(cat_dir) else []
    if not glb_files:
        return None, None

    glb_path = glb_files[0]
    if model_name:
        candidate = os.path.join(cat_dir, model_name + '.glb')
        if os.path.exists(candidate):
            glb_path = candidate

    if glb_path in _mesh_cache:
        mesh, colors = _mesh_cache[glb_path]
        return mesh.copy(), colors

    loaded = trimesh.load(glb_path)
    if isinstance(loaded, trimesh.Scene):
        parts = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            return None, None
        colors = _extract_vertex_colors(parts)
        mesh   = trimesh.util.concatenate(parts)
    elif isinstance(loaded, trimesh.Trimesh):
        colors = _extract_vertex_colors([loaded])
        mesh   = loaded
    else:
        return None, None

    _mesh_cache[glb_path] = (mesh, colors)
    return mesh.copy(), colors


def place_mesh_at_gt(mesh: trimesh.Trimesh, center, q_wxyz) -> trimesh.Trimesh:
    """Apply full quaternion rotation around local origin, translate to GT center.

    GT center is the object's pivot (local origin) position in world space,
    not the geometric center of bounds — so we must NOT subtract the geometric
    center before rotating.
    """
    verts = mesh.vertices @ quat_to_mat3(q_wxyz).T
    mesh.vertices = verts + np.array(center, dtype=np.float64)
    return mesh


# ── URDF articulated-object loading ───────────────────────────────────────────
_urdf_cache: dict = {}


def _rpy_to_mat3(rpy):
    r, p, y = rpy
    Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
    return Rz @ Ry @ Rx


def _origin_T(elem) -> np.ndarray:
    if elem is None:
        return np.eye(4)
    xyz = list(map(float, elem.get('xyz', '0 0 0').split()))
    rpy = list(map(float, elem.get('rpy', '0 0 0').split()))
    T = np.eye(4)
    T[:3, :3] = _rpy_to_mat3(rpy)
    T[:3, 3]  = xyz
    return T


def _load_urdf_parts(obj_dir: str, label: str, urdf_file: str):
    robot = ET.parse(os.path.join(obj_dir, label, urdf_file)).getroot()

    joint_map = {}
    for j in robot.findall('joint'):
        parent = j.find('parent').get('link')
        child  = j.find('child').get('link')
        joint_map[child] = (parent, _origin_T(j.find('origin')))

    _SKIP_ROOT = {'chest_of_drawers', 'kitchen_counter', 'kitchen_cupboards'}
    T_world: dict = {'root': np.eye(4)}
    if label in _SKIP_ROOT:
        T_world['body'] = np.eye(4)

    def world_T(name):
        if name in T_world:
            return T_world[name]
        parent, T_j = joint_map[name]
        T = world_T(parent) @ T_j
        T_world[name] = T
        return T

    for lnk in robot.findall('link'):
        world_T(lnk.get('name'))

    parts = []
    for lnk in robot.findall('link'):
        link_name = lnk.get('name')
        vis = lnk.find('visual')
        if vis is None:
            continue
        mesh_elem = vis.find('geometry/mesh')
        if mesh_elem is None:
            continue
        glb_path = os.path.join(obj_dir, label, mesh_elem.get('filename'))
        if not os.path.exists(glb_path):
            continue

        T = world_T(link_name) @ _origin_T(vis.find('origin'))

        loaded = trimesh.load(glb_path)
        if isinstance(loaded, trimesh.Scene):
            sub = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
            if not sub:
                continue
            colors = _extract_vertex_colors(sub)
            mesh   = trimesh.util.concatenate(sub)
        elif isinstance(loaded, trimesh.Trimesh):
            colors = _extract_vertex_colors([loaded])
            mesh   = loaded
        else:
            continue

        mesh.apply_transform(T)
        parts.append((mesh, colors))

    return parts


def load_urdf_parts(obj_dir: str, label: str, urdf_file: str):
    if label not in _urdf_cache:
        _urdf_cache[label] = _load_urdf_parts(obj_dir, label, urdf_file)
    return [(m.copy(), c) for m, c in _urdf_cache[label]]


def place_parts_at_gt(parts, center, lwh, yaw: float):
    """Scale URDF parts to GT world-space size (L×H×W), apply yaw, translate to center."""
    L, W, H = lwh

    all_verts  = np.concatenate([m.vertices for m, _ in parts], axis=0)
    cur_center = (all_verts.min(0) + all_verts.max(0)) / 2.0
    cur_size   = all_verts.max(0) - all_verts.min(0)

    target      = np.array([L, H, W], dtype=np.float64)
    shape_scale = np.where(cur_size > 1e-6, target / cur_size, 1.0)

    R = yaw_mat3(yaw)
    result = []
    for mesh, colors in parts:
        m     = mesh.copy()
        verts = (m.vertices - cur_center) * shape_scale
        verts = verts @ R.T
        m.vertices = verts + np.array(center, dtype=np.float64)
        result.append((m, colors))
    return result


# ── Open3D conversion ──────────────────────────────────────────────────────────

def trimesh_to_o3d(mesh: trimesh.Trimesh,
                   vertex_colors,
                   fallback_color) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices  = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces,    np.int32))
    if vertex_colors is not None and len(vertex_colors) == len(mesh.vertices):
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    else:
        o3d_mesh.paint_uniform_color(fallback_color)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


# ── scene geometry builders ────────────────────────────────────────────────────

def _obb_wireframe(center, lwh, yaw, color) -> o3d.geometry.LineSet:
    """Oriented bounding box wireframe from center, L/W/H, yaw."""
    L, W, H = lwh
    half    = np.array([L / 2, H / 2, W / 2])
    corners_local = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=np.float64) * half

    R       = yaw_mat3(yaw)
    corners = corners_local @ R.T + np.array(center)

    edges = [[0,1],[1,2],[2,3],[3,0],
             [4,5],[5,6],[6,7],[7,4],
             [0,4],[1,5],[2,6],[3,7]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines  = o3d.utility.Vector2iVector(np.array(edges, np.int32))
    ls.paint_uniform_color(color)
    return ls


def _ground_grid(y: float = 0.0, extent: float = 12.0, step: float = 1.0) -> o3d.geometry.LineSet:
    pts, lines = [], []
    vals = np.arange(0, extent + step, step)
    for v in vals:
        i = len(pts)
        pts += [[0, y, v], [extent, y, v]]
        lines.append([i, i + 1])
        i = len(pts)
        pts += [[v, y, 0], [v, y, extent]]
        lines.append([i, i + 1])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts, np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines, np.int32))
    ls.paint_uniform_color([0.45, 0.45, 0.45])
    return ls


def build_scene_gt(frame_data: dict, obj_dir: str) -> list:
    geoms = []
    nodes = frame_data.get('nodes', [])
    geoms.append(_ground_grid())

    for node in nodes:
        raw_label  = node['label']
        label      = LABEL_ALIAS.get(raw_label, raw_label)
        model_name = node.get('model')
        center     = node['center']
        q_wxyz     = node['orientation']
        sz         = node['size']
        lwh        = (sz['L'], sz['W'], sz['H'])
        yaw        = quat_yaw(q_wxyz)
        color      = LABEL_COLOR.get(label, DEFAULT_LABEL_COLOR)

        if label in URDF_FILES:
            parts = load_urdf_parts(obj_dir, label, URDF_FILES[label])
            urdf_yaw = quat_zup_yaw(q_wxyz)
            parts = place_parts_at_gt(parts, center, lwh, urdf_yaw)
            for m, vc in parts:
                geoms.append(trimesh_to_o3d(m, vc, color))
        else:
            mesh, vc = load_mesh(obj_dir, label, model_name=model_name)
            if mesh is not None:
                mesh = place_mesh_at_gt(mesh, center, q_wxyz)
                geoms.append(trimesh_to_o3d(mesh, vc, color))
            else:
                geoms.append(_obb_wireframe(center, lwh, yaw, color))

    return geoms


# ── main ───────────────────────────────────────────────────────────────────────

def _bool(s):
    if s in ('1', 'True', 'true'):
        return True
    if s in ('0', 'False', 'false'):
        return False
    raise argparse.ArgumentTypeError(f'invalid bool: {s!r}')


def _accumulate_up_to(frames, idx):
    """Merge frames[0..idx] into one node list: each node_id keeps only its
    LATEST observation (by node_id, since GT json is per-frame "currently
    visible", not cumulative — a reappearing node_id overwrites the earlier
    entry with its newer pose)."""
    acc = {}
    for i in range(idx + 1):
        for node in frames[i].get('nodes', []):
            acc[node['node_id']] = node
    return list(acc.values())


def main():
    parser = argparse.ArgumentParser(description='Visualize GT 3D scene graph sequence')
    parser.add_argument('--sg_dir',  default='/media/gao/gao_ssd/3D Scene/1_1',
                        help='Directory with sg_frame_*.json GT files')
    parser.add_argument('--obj_dir', default='/media/gao/gao_ssd/3D objects',
                        help='Directory with per-category .glb models')
    parser.add_argument('--accumulate', type=_bool, default=True,
                        help='show every object seen in any frame up to the current '
                             'one (default), instead of only the objects visible in '
                             'the current frame')
    args = parser.parse_args()

    json_files = sorted(glob.glob(os.path.join(args.sg_dir, 'sg_frame_*.json')))
    if not json_files:
        print(f'No sg_frame_*.json found in {args.sg_dir}')
        sys.exit(1)

    frames = []
    for p in json_files:
        with open(p) as f:
            d = json.load(f)
        if d.get('nodes'):
            frames.append(d)

    if not frames:
        print('All frames are empty.')
        sys.exit(1)

    print(f'Loaded {len(frames)} non-empty frames from {args.sg_dir}')
    print('Keys:  Space / → : next frame    ← : previous frame    Q : quit\n')

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='GT 3D Scene Sequence', width=1280, height=800)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.10, 0.10, 0.13])
    opt.point_size = 4.0

    state = {'idx': 0, 'geoms': [], 'first': True}

    def refresh(vis):
        for g in state['geoms']:
            vis.remove_geometry(g, reset_bounding_box=False)
        state['geoms'].clear()

        fd = frames[state['idx']]
        show_nodes = (_accumulate_up_to(frames, state['idx']) if args.accumulate
                      else fd['nodes'])
        geoms = build_scene_gt({'nodes': show_nodes}, args.obj_dir)
        reset = state['first']
        for g in geoms:
            vis.add_geometry(g, reset_bounding_box=reset)
        state['geoms'] = geoms
        state['first'] = False

        cur_labels = [LABEL_ALIAS.get(n['label'], n['label']) for n in fd['nodes']]
        tag = f'accumulated({len(show_nodes)})' if args.accumulate else f'objects({len(cur_labels)})'
        print(f'\rframe_idx={fd["frame_idx"]:05d}  '
              f'[{state["idx"]+1}/{len(frames)}]  '
              f'scene={fd.get("scene","?")}  '
              f'{tag}  this_frame={cur_labels}     ',
              end='', flush=True)
        return False

    def next_frame(vis):
        state['idx'] = min(state['idx'] + 1, len(frames) - 1)
        refresh(vis)
        return False

    def prev_frame(vis):
        state['idx'] = max(state['idx'] - 1, 0)
        refresh(vis)
        return False

    vis.register_key_callback(262, next_frame)   # Right arrow
    vis.register_key_callback(263, prev_frame)   # Left arrow
    vis.register_key_callback(32,  next_frame)   # Space

    refresh(vis)
    vis.run()
    vis.destroy_window()
    print()


if __name__ == '__main__':
    main()
