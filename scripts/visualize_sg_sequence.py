#!/usr/bin/env python3
"""
Interactive 3D visualization of raw robot scene graph frames.

Reads frame_XXXXXX.json files from the robot SG directory
(e.g. /media/xxx/gao_ssd/3D scene graph/1_1/).

Each node is shown as:
  - An axis-aligned bounding box wireframe  (bbox_min / bbox_max)
  - A centroid sphere coloured by label

Edges between nodes are drawn as thin grey lines.
Frame info (labels, confidences, relations) is printed to the console.

Usage:
    python scripts/visualize_sg_sequence.py \
        --sg_dir "/media/xxx/gao_ssd/3D scene graph/1_1"

    # show every raw frame instead of only change-frames
    python scripts/visualize_sg_sequence.py \
        --sg_dir "/media/xxx/gao_ssd/3D scene graph/1_1" --all_frames

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

import numpy as np
import open3d as o3d

# ── colour palette (same as GT viewer) ────────────────────────────────────────
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

# ── geometry helpers ───────────────────────────────────────────────────────────

def _aabb_wireframe(bbox_min, bbox_max, color) -> o3d.geometry.LineSet:
    mn, mx = np.array(bbox_min, np.float64), np.array(bbox_max, np.float64)
    pts = [
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ]
    edges = [[0,1],[1,2],[2,3],[3,0],
             [4,5],[5,6],[6,7],[7,4],
             [0,4],[1,5],[2,6],[3,7]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts, np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.array(edges, np.int32))
    ls.paint_uniform_color(color)
    return ls


def _ground_grid(y: float = 0.0, extent: float = 12.0,
                 step: float = 1.0) -> o3d.geometry.LineSet:
    pts, lines = [], []
    vals = np.arange(0, extent + step, step)
    for v in vals:
        i = len(pts)
        pts += [[0, y, v], [extent, y, v]]; lines.append([i, i+1])
        i = len(pts)
        pts += [[v, y, 0], [v, y, extent]]; lines.append([i, i+1])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts, np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines, np.int32))
    ls.paint_uniform_color([0.40, 0.40, 0.40])
    return ls


# ── scene builder ──────────────────────────────────────────────────────────────

def build_scene(frame_data: dict) -> list:
    geoms = [_ground_grid()]
    nodes = frame_data.get('nodes', [])
    edges = frame_data.get('edges', [])

    id_to_centroid = {}
    for node in nodes:
        label    = node['label']
        centroid = np.array(node['center'], np.float64)
        color    = LABEL_COLOR.get(label, DEFAULT_COLOR)
        id_to_centroid[node['id']] = centroid

        geoms.append(_aabb_wireframe(node['bbox_min'], node['bbox_max'], color))

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere.translate(centroid)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        geoms.append(sphere)

    for edge in edges:
        src, dst = edge['from'], edge['to']
        if src in id_to_centroid and dst in id_to_centroid:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(
                [id_to_centroid[src], id_to_centroid[dst]])
            ls.lines  = o3d.utility.Vector2iVector([[0, 1]])
            ls.paint_uniform_color([0.55, 0.55, 0.55])
            geoms.append(ls)

    return geoms


# ── change detection ───────────────────────────────────────────────────────────
_MOVE_THRESH = 0.05   # metres

def _is_changed(prev: dict, curr: dict) -> bool:
    prev_ids = {n['id'] for n in prev['nodes']}
    curr_ids = {n['id'] for n in curr['nodes']}
    if prev_ids != curr_ids:
        return True
    prev_cen = {n['id']: np.array(n['center']) for n in prev['nodes']}
    curr_cen = {n['id']: np.array(n['center']) for n in curr['nodes']}
    return any(np.linalg.norm(curr_cen[nid] - prev_cen[nid]) > _MOVE_THRESH
               for nid in curr_ids)


# ── console print ──────────────────────────────────────────────────────────────

def _print_frame(frame_data: dict, frame_idx: int, total: int):
    nodes = frame_data.get('nodes', [])
    edges = frame_data.get('edges', [])
    print(f'\n── frame {frame_data["frame"]:06d}  [{frame_idx+1}/{total}]  '
          f'{len(nodes)} nodes  {len(edges)} edges ──')
    for n in nodes:
        sz = np.array(n['bbox_max']) - np.array(n['bbox_min'])
        print(f'  [{n["id"]}] {n["label"]:<20s}  '
              f'conf={n["confidence"]:.3f}  '
              f'cen=({n["center"][0]:+.2f},{n["center"][1]:+.2f},{n["center"][2]:+.2f})  '
              f'sz=({sz[0]:.2f},{sz[1]:.2f},{sz[2]:.2f})')
    if edges:
        id_to_label = {n['id']: n['label'] for n in nodes}
        print('  edges:')
        for e in edges[:8]:  # show first 8 to avoid wall of text
            print(f'    {id_to_label.get(e["from"],"?")} --[{e["relation"]}]--> '
                  f'{id_to_label.get(e["to"],"?")}  conf={e["confidence"]:.3f}')
        if len(edges) > 8:
            print(f'    ... ({len(edges)-8} more)')


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Interactive 3D viewer for robot scene graph frames')
    parser.add_argument('--sg_dir', default='/media/gao/gao_ssd/3D scene graph/1_1',
                        help='Directory containing frame_XXXXXX.json robot SG files')
    parser.add_argument('--all_frames', action='store_true',
                        help='Show every frame; default shows only change-frames')
    args = parser.parse_args()

    json_files = sorted(glob.glob(os.path.join(args.sg_dir, 'frame_*.json')))
    if not json_files:
        print(f'No frame_*.json found in {args.sg_dir}')
        sys.exit(1)

    all_frames = []
    for p in json_files:
        with open(p) as f:
            all_frames.append(json.load(f))

    if args.all_frames:
        frames = all_frames
    else:
        frames = [all_frames[0]]
        for i in range(1, len(all_frames)):
            if _is_changed(all_frames[i-1], all_frames[i]):
                frames.append(all_frames[i])

    seq_name = os.path.basename(args.sg_dir.rstrip('/'))
    print(f'Sequence {seq_name}: {len(all_frames)} raw frames → '
          f'{len(frames)} {"(all)" if args.all_frames else "change-frames"} shown')
    print('Keys:  Space / → : next    ← : prev    Q : quit')

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f'Robot SG  {seq_name}', width=1280, height=800)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.08, 0.08, 0.12])
    opt.point_size = 4.0

    state = {'idx': 0, 'geoms': [], 'first': True}

    def refresh(vis):
        for g in state['geoms']:
            vis.remove_geometry(g, reset_bounding_box=False)
        state['geoms'].clear()

        fd    = frames[state['idx']]
        geoms = build_scene(fd)
        reset = state['first']
        for g in geoms:
            vis.add_geometry(g, reset_bounding_box=reset)
        state['geoms'] = geoms
        state['first'] = False
        _print_frame(fd, state['idx'], len(frames))
        return False

    def next_frame(vis):
        state['idx'] = min(state['idx'] + 1, len(frames) - 1)
        refresh(vis)
        return False

    def prev_frame(vis):
        state['idx'] = max(state['idx'] - 1, 0)
        refresh(vis)
        return False

    vis.register_key_callback(262, next_frame)  # Right arrow
    vis.register_key_callback(263, prev_frame)  # Left arrow
    vis.register_key_callback(32,  next_frame)  # Space

    refresh(vis)
    vis.run()
    vis.destroy_window()
    print()


if __name__ == '__main__':
    main()
