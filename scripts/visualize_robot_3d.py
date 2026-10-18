#!/usr/bin/env python3
"""
Visualize robot-side scene graph frames with 3D meshes placed inside observed AABBs.

For each detected object the script:
  1. Randomly picks one .glb model from the matching category directory.
  2. Finds the yaw (0° or 90°) whose mesh AABB best fits the observed robot AABB.
  3. Scales the mesh to exactly fill the observed AABB, places it at the centroid.

The observed AABB wireframe is also drawn in a dim colour for reference.

Usage (from repo root):
    conda run -n commonscenes python scripts/visualize_robot_3d.py \
        --sg_dir  "/media/gao/gao_ssd/3D scene graph/1_1" \
        --obj_dir "/media/gao/gao_ssd/3D objects"

Keys:
    Space / Right arrow : next frame
    Left arrow          : previous frame
    Q                   : quit
"""

import argparse
import glob
import json
import os
import random
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
    'kitchen_cupboard':  [0.75, 0.65, 0.55],
    'kitchen_cupboards': [0.75, 0.65, 0.55],
    'chest_of_drawers':  [0.70, 0.55, 0.35],
    'wall_cabinet':      [0.70, 0.55, 0.35],
    'clothes_hanger':    [0.80, 0.70, 0.60],
    'doors':             [0.65, 0.45, 0.25],
}
DEFAULT_COLOR = [0.75, 0.75, 0.75]

URDF_FILES = {
    'chest_of_drawers':  'chestOfDrawers_01.urdf',
    'kitchen_counter':   'kitchen_counter.urdf',
    'kitchen_cupboards': 'kitchenCupboard_01.urdf',
    'fridge':            'fridge.urdf',
}

LABEL_ALIAS = {
    'kitchen_cupboard': 'kitchen_cupboards',
}


# ── rotation helpers ───────────────────────────────────────────────────────────

def _yaw_mat3(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def _rpy_to_mat3(rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# ── yaw estimation from AABB fitting ──────────────────────────────────────────

def _best_yaw(mesh_verts: np.ndarray, obs_x: float, obs_z: float) -> float:
    """Return 0.0 or π/2 — whichever yaw makes the mesh AABB best fit obs_x × obs_z.

    At yaw=0°  the mesh X-extent maps to world X, Z-extent maps to world Z.
    At yaw=90° the mesh Z-extent maps to world X, X-extent maps to world Z.
    We pick whichever minimises squared AABB error.
    """
    mn, mx = mesh_verts.min(0), mesh_verts.max(0)
    mesh_x = float(mx[0] - mn[0])
    mesh_z = float(mx[2] - mn[2])

    err_0  = (obs_x - mesh_x) ** 2 + (obs_z - mesh_z) ** 2
    err_90 = (obs_x - mesh_z) ** 2 + (obs_z - mesh_x) ** 2

    return 0.0 if err_0 <= err_90 else (np.pi / 2)


def _place_in_aabb(mesh: trimesh.Trimesh,
                   center: np.ndarray,
                   obs_x: float, obs_y: float, obs_z: float,
                   yaw: float) -> trimesh.Trimesh:
    """Scale mesh to fill obs AABB, rotate by yaw, translate to centroid."""
    v  = mesh.vertices
    mn, mx = v.min(0), v.max(0)
    cur = mx - mn                       # current extents before rotation
    geo_center = (mn + mx) / 2.0

    # Scale so that post-rotation AABB matches observed extents
    if abs(yaw) < 0.1:                  # 0° – X→X, Z→Z
        sx = obs_x / max(cur[0], 1e-6)
        sz = obs_z / max(cur[2], 1e-6)
    else:                               # 90° – Z→X, X→Z
        sx = obs_x / max(cur[2], 1e-6)
        sz = obs_z / max(cur[0], 1e-6)
    sy = obs_y / max(cur[1], 1e-6)

    verts = (v - geo_center) * np.array([sx, sy, sz])
    verts = verts @ _yaw_mat3(yaw).T + center
    mesh.vertices = verts
    return mesh


# ── mesh loading / caching ────────────────────────────────────────────────────
_glb_cache:  dict = {}   # glb_path → (trimesh.Trimesh, colors_or_None)
_urdf_cache: dict = {}   # label    → [(trimesh.Trimesh, colors_or_None)]


def _extract_colors(parts):
    chunks = []
    for part in parts:
        try:
            vc = part.visual.to_color().vertex_colors[:, :3].astype(np.float64) / 255.
            chunks.append(vc)
        except Exception:
            return None
    return np.concatenate(chunks, 0) if chunks else None


def _load_glb(path: str):
    if path in _glb_cache:
        m, c = _glb_cache[path]
        return m.copy(), c
    loaded = trimesh.load(path)
    if isinstance(loaded, trimesh.Scene):
        parts = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            return None, None
        colors = _extract_colors(parts)
        mesh   = trimesh.util.concatenate(parts)
    elif isinstance(loaded, trimesh.Trimesh):
        colors = _extract_colors([loaded])
        mesh   = loaded
    else:
        return None, None
    _glb_cache[path] = (mesh, colors)
    return mesh.copy(), colors


def load_random_mesh(obj_dir: str, label: str):
    """Return (trimesh.Trimesh, vertex_colors_or_None) for a random model in category."""
    cat_dir = os.path.join(obj_dir, label)
    glbs    = sorted(glob.glob(os.path.join(cat_dir, '*.glb'))) if os.path.isdir(cat_dir) else []
    if not glbs:
        return None, None
    return _load_glb(random.choice(glbs))


def _origin_T(elem) -> np.ndarray:
    if elem is None:
        return np.eye(4)
    xyz = list(map(float, elem.get('xyz', '0 0 0').split()))
    rpy = list(map(float, elem.get('rpy', '0 0 0').split()))
    T = np.eye(4)
    T[:3, :3] = _rpy_to_mat3(rpy)
    T[:3, 3]  = xyz
    return T


def _load_urdf_raw(obj_dir: str, label: str, urdf_file: str):
    robot = ET.parse(os.path.join(obj_dir, label, urdf_file)).getroot()
    joint_map = {}
    for j in robot.findall('joint'):
        parent = j.find('parent').get('link')
        child  = j.find('child').get('link')
        joint_map[child] = (parent, _origin_T(j.find('origin')))

    _SKIP_ROOT = {'chest_of_drawers', 'kitchen_counter', 'kitchen_cupboards'}
    T_world = {'root': np.eye(4)}
    if label in _SKIP_ROOT:
        T_world['body'] = np.eye(4)

    def world_T(name):
        if name in T_world:
            return T_world[name]
        parent, Tj = joint_map[name]
        T = world_T(parent) @ Tj
        T_world[name] = T
        return T

    for lnk in robot.findall('link'):
        world_T(lnk.get('name'))

    parts = []
    for lnk in robot.findall('link'):
        vis = lnk.find('visual')
        if vis is None:
            continue
        me = vis.find('geometry/mesh')
        if me is None:
            continue
        glb_path = os.path.join(obj_dir, label, me.get('filename'))
        if not os.path.exists(glb_path):
            continue
        T = world_T(lnk.get('name')) @ _origin_T(vis.find('origin'))
        mesh, colors = _load_glb(glb_path)
        if mesh is None:
            continue
        mesh.apply_transform(T)
        parts.append((mesh, colors))
    return parts


def load_urdf_parts(obj_dir: str, label: str, urdf_file: str):
    if label not in _urdf_cache:
        _urdf_cache[label] = _load_urdf_raw(obj_dir, label, urdf_file)
    return [(m.copy(), c) for m, c in _urdf_cache[label]]


def _place_urdf_in_aabb(parts, center, obs_x, obs_y, obs_z, yaw):
    """Scale URDF assembly to fit observed AABB, apply yaw, translate."""
    all_v = np.concatenate([m.vertices for m, _ in parts], 0)
    mn, mx = all_v.min(0), all_v.max(0)
    cur = mx - mn
    geo_center = (mn + mx) / 2.0

    if abs(yaw) < 0.1:
        sx = obs_x / max(cur[0], 1e-6)
        sz = obs_z / max(cur[2], 1e-6)
    else:
        sx = obs_x / max(cur[2], 1e-6)
        sz = obs_z / max(cur[0], 1e-6)
    sy = obs_y / max(cur[1], 1e-6)

    R = _yaw_mat3(yaw)
    result = []
    for mesh, colors in parts:
        m = mesh.copy()
        v = (m.vertices - geo_center) * np.array([sx, sy, sz])
        m.vertices = v @ R.T + center
        result.append((m, colors))
    return result


# ── Open3D conversion ──────────────────────────────────────────────────────────

def _to_o3d(mesh: trimesh.Trimesh, vertex_colors, fallback_color):
    o3d_m = o3d.geometry.TriangleMesh()
    o3d_m.vertices  = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, np.float64))
    o3d_m.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, np.int32))
    if vertex_colors is not None and len(vertex_colors) == len(mesh.vertices):
        o3d_m.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    else:
        o3d_m.paint_uniform_color(fallback_color)
    o3d_m.compute_vertex_normals()
    return o3d_m


def _aabb_wireframe(bbox_min, bbox_max, color) -> o3d.geometry.LineSet:
    mn, mx = np.array(bbox_min, np.float64), np.array(bbox_max, np.float64)
    pts = [
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mn[1], mx[2]], [mn[0], mn[1], mx[2]],
        [mn[0], mx[1], mn[2]], [mx[0], mx[1], mn[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ]
    edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts, np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.array(edges, np.int32))
    ls.paint_uniform_color(color)
    return ls


def _ground_grid(y=0.0, extent=12.0, step=1.0) -> o3d.geometry.LineSet:
    pts, lines = [], []
    for v in np.arange(0, extent + step, step):
        i = len(pts); pts += [[0, y, v], [extent, y, v]]; lines.append([i, i+1])
        i = len(pts); pts += [[v, y, 0], [v, y, extent]]; lines.append([i, i+1])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts, np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines, np.int32))
    ls.paint_uniform_color([0.30, 0.30, 0.30])
    return ls


# ── scene builder ──────────────────────────────────────────────────────────────

def build_scene_robot(frame_data: dict, obj_dir: str) -> list:
    geoms = [_ground_grid()]

    for node in frame_data.get('nodes', []):
        raw_label = node['label']
        label     = LABEL_ALIAS.get(raw_label, raw_label)
        color     = LABEL_COLOR.get(label, DEFAULT_COLOR)

        bmin    = np.array(node['bbox_min'], np.float64)
        bmax    = np.array(node['bbox_max'], np.float64)
        center  = np.array(node['center'], np.float64)
        obs_x   = float(bmax[0] - bmin[0])
        obs_y   = float(bmax[1] - bmin[1])
        obs_z   = float(bmax[2] - bmin[2])

        # dim AABB wireframe for reference
        geoms.append(_aabb_wireframe(bmin, bmax, [c * 0.4 for c in color]))

        placed = False

        if label in URDF_FILES:
            parts = load_urdf_parts(obj_dir, label, URDF_FILES[label])
            if parts:
                all_v = np.concatenate([m.vertices for m, _ in parts], 0)
                yaw   = _best_yaw(all_v, obs_x, obs_z)
                parts = _place_urdf_in_aabb(parts, center, obs_x, obs_y, obs_z, yaw)
                for m, vc in parts:
                    geoms.append(_to_o3d(m, vc, color))
                placed = True
        else:
            mesh, vc = load_random_mesh(obj_dir, label)
            if mesh is not None:
                yaw  = _best_yaw(mesh.vertices, obs_x, obs_z)
                mesh = _place_in_aabb(mesh, center, obs_x, obs_y, obs_z, yaw)
                geoms.append(_to_o3d(mesh, vc, color))
                placed = True

        if not placed:
            # fallback: bright wireframe
            geoms.append(_aabb_wireframe(bmin, bmax, color))

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere.translate(center)
        sphere.paint_uniform_color([1.0, 0.95, 0.0])
        sphere.compute_vertex_normals()
        geoms.append(sphere)

    return geoms


# ── key handling / main loop ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Robot SG frames with 3D meshes fitted to AABBs')
    parser.add_argument('--sg_dir',  default='/media/gao/gao_ssd/3D scene graph/1_1')
    parser.add_argument('--obj_dir', default='/media/gao/gao_ssd/3D objects')
    parser.add_argument('--all_frames', action='store_true',
                        help='Show every frame (default: only frames where SG changes)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for mesh selection (use -1 for random each run)')
    args = parser.parse_args()

    if args.seed >= 0:
        random.seed(args.seed)

    json_files = sorted(glob.glob(os.path.join(args.sg_dir, 'frame_*.json')))
    if not json_files:
        print(f'No frame_*.json found in {args.sg_dir}'); sys.exit(1)

    frames = [json.load(open(p)) for p in json_files if json.load(open(p)).get('nodes')]

    if not args.all_frames:
        # keep only change frames
        keep = [frames[0]]
        for prev, curr in zip(frames, frames[1:]):
            prev_ids = {n['id'] for n in prev['nodes']}
            curr_ids = {n['id'] for n in curr['nodes']}
            if prev_ids != curr_ids:
                keep.append(curr)
                continue
            prev_c = {n['id']: np.array(n['center']) for n in prev['nodes']}
            curr_c = {n['id']: np.array(n['center']) for n in curr['nodes']}
            if any(np.linalg.norm(curr_c[i] - prev_c[i]) > 0.05 for i in curr_ids):
                keep.append(curr)
        frames = keep

    print(f'Loaded {len(frames)} frames from {args.sg_dir}')
    print('Keys:  Space / → : next frame    ← : previous frame    Q : quit\n')

    idx   = [0]
    geoms = [None]

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='Robot SG – 3D mesh view', width=1280, height=800)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.10, 0.10, 0.13])
    opt.point_size       = 4.0

    def _refresh(v):
        v.clear_geometries()
        frame = frames[idx[0]]
        gs    = build_scene_robot(frame, args.obj_dir)
        for g in gs:
            v.add_geometry(g, reset_bounding_box=(g is gs[0]))
        geoms[0] = gs
        nodes = frame.get('nodes', [])
        print(f'\n── frame {frame.get("frame", "?")}  [{idx[0]+1}/{len(frames)}]  {len(nodes)} nodes ──')
        for n in nodes:
            sz = np.array(n['bbox_max']) - np.array(n['bbox_min'])
            print(f'  {n["label"]:<22s}  conf={n["confidence"]:.2f}  '
                  f'cen=({n["center"][0]:+.2f},{n["center"][1]:+.2f},{n["center"][2]:+.2f})  '
                  f'sz=({sz[0]:.2f},{sz[1]:.2f},{sz[2]:.2f})')

    def _next(v):
        if idx[0] < len(frames) - 1:
            idx[0] += 1; _refresh(v)

    def _prev(v):
        if idx[0] > 0:
            idx[0] -= 1; _refresh(v)

    def _quit(v):
        v.close()

    vis.register_key_callback(ord('Q'), _quit)
    vis.register_key_callback(32,       _next)   # Space
    vis.register_key_callback(262,      _next)   # Right arrow
    vis.register_key_callback(263,      _prev)   # Left arrow

    _refresh(vis)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
