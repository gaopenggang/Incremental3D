#!/usr/bin/env python3
"""
Pre-compute [L, H, W] bounding-box sizes for every model in the 3D object
library and save to model_sizes.json alongside features.pkl.

Run ONCE; TeleShapeLibrary loads from JSON at runtime.

Strategy (in priority order):
  1. GT JSON scan  – scan all sg_frame_*.json files, match by model name.
     These sizes are ground-truth (URDF collision-box AABB × scale × scene
     rotation already applied) and are the most accurate.
  2. Mesh AABB     – open3d on the GLB file.  Convention: Y-up, forward=-Z
     → L = Z-extent, H = Y-extent, W = X-extent.
  3. Fallback      – [1.0, 1.0, 1.0] when neither source works.

Usage:
    # open3d must be imported before torch/numpy to avoid GLIBCXX conflict
    python scripts/precompute_model_sizes.py \
        --pkl  /media/gao/gao_ssd/3D\ objects/features.pkl \
        --gt   /media/gao/gao_ssd/3D\ Scene \
        --out  /media/gao/gao_ssd/3D\ objects/model_sizes.json
"""
# ── import open3d FIRST – before numpy/torch to avoid GLIBCXX conflict ────────
import open3d as o3d

import argparse
import glob
import json
import os
import pickle
import re

# numpy is imported AFTER open3d intentionally
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – build model_name → (L, H, W) map from GT JSON files
# ─────────────────────────────────────────────────────────────────────────────

def _norm_model_name(s: str) -> str:
    """Canonical model name for matching (lower, strip path/ext/trailing numbers)."""
    base = os.path.basename(s)
    # remove extension
    base = re.sub(r'\.(glb|urdf|json)$', '', base, flags=re.IGNORECASE)
    return base.strip().lower()


def scan_gt_sizes(gt_root: str) -> dict:
    """
    Scan all sg_frame_*.json in gt_root and build:
        {norm_model_name: (L, H, W)}

    Each scene keeps only the first occurrence of each model (sizes are fixed
    per model, so any frame gives the same values).
    """
    frames = sorted(glob.glob(os.path.join(gt_root, '*', 'sg_frame_*.json')))
    print(f'Scanning {len(frames)} GT frames in {gt_root} …')

    size_map = {}   # norm_model_name → (L, H, W)
    for fp in frames:
        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception:
            continue
        for node in data.get('nodes', []):
            sz = node.get('size', {})
            model = node.get('model', '')
            if not sz or not model:
                continue
            key = _norm_model_name(model)
            if key not in size_map:
                size_map[key] = (float(sz['L']), float(sz['H']), float(sz['W']))

    print(f'  Found sizes for {len(size_map)} distinct model names')
    for name, (L, H, W) in sorted(size_map.items()):
        print(f'    {name:35s}  L={L:.3f}  H={H:.3f}  W={W:.3f}')
    return size_map


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – fallback: compute from GLB mesh with open3d
# Convention: Y-up, forward = -Z  →  L = Z-extent, H = Y-extent, W = X-extent
# ─────────────────────────────────────────────────────────────────────────────

def size_from_glb(glb_path: str):
    """Return [L, H, W] from open3d AABB, or None on failure."""
    if not os.path.exists(glb_path):
        return None
    try:
        mesh = o3d.io.read_triangle_mesh(glb_path)
        if len(mesh.vertices) == 0:
            return None
        ext = mesh.get_axis_aligned_bounding_box().get_extent()   # [X, Y, Z]
        # L = Z (depth/length), H = Y (height, vertical), W = X (width)
        return [float(ext[2]), float(ext[1]), float(ext[0])]
    except Exception as e:
        print(f'    [WARN] open3d failed on {glb_path}: {e}')
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', default='/media/gao/gao_ssd/3D objects/features.pkl')
    parser.add_argument('--gt',  default='/media/gao/gao_ssd/3D Scene',
                        help='root directory containing scene subdirs with sg_frame_*.json')
    parser.add_argument('--out', default=None,
                        help='output JSON path (default: same dir as pkl)')
    args = parser.parse_args()

    out_path = args.out or os.path.join(os.path.dirname(args.pkl), 'model_sizes.json')

    # ── load features.pkl ────────────────────────────────────────────────────
    with open(args.pkl, 'rb') as f:
        raw = pickle.load(f)
    print(f'Library: {len(raw)} models')

    # ── scan GT sizes ────────────────────────────────────────────────────────
    gt_size_map = scan_gt_sizes(args.gt)

    # ── compute per-model size ───────────────────────────────────────────────
    result = {}   # glb_or_urdf_path → [L, H, W]
    stats = {'gt': 0, 'mesh': 0, 'fallback': 0}

    for rel_key, v in raw.items():
        glb_path = v['glb_path']
        label    = v['label']

        # Source 1: GT JSON match by model name
        key = _norm_model_name(glb_path)
        size = None
        if key in gt_size_map:
            L, H, W = gt_size_map[key]
            size = [L, H, W]
            source = 'GT'
            stats['gt'] += 1
        else:
            # Try alternate keys: just the label
            label_key = label.strip().lower()
            if label_key in gt_size_map:
                L, H, W = gt_size_map[label_key]
                size = [L, H, W]
                source = 'GT(label)'
                stats['gt'] += 1

        # Source 2: GLB mesh AABB (only for actual .glb files, not .urdf)
        if size is None and glb_path.lower().endswith('.glb'):
            size = size_from_glb(glb_path)
            if size is not None:
                source = 'mesh'
                stats['mesh'] += 1

        # Source 3: fallback
        if size is None:
            size = None   # stored as None → TeleShapeLibrary falls back to enc size
            source = 'FALLBACK(None)'
            stats['fallback'] += 1

        result[glb_path] = size
        L_s = f'{size[0]:.3f}' if size else 'None'
        H_s = f'{size[1]:.3f}' if size else 'None'
        W_s = f'{size[2]:.3f}' if size else 'None'
        print(f'  [{source:12s}]  {label:20s}  L={L_s}  H={H_s}  W={W_s}')

    # ── save ─────────────────────────────────────────────────────────────────
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\nSaved {len(result)} entries → {out_path}')
    print(f'  GT: {stats["gt"]}  mesh: {stats["mesh"]}  fallback: {stats["fallback"]}')


if __name__ == '__main__':
    main()
