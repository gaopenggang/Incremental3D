#!/usr/bin/env python3
"""
ROS 2 online-inference node for Sg2ScAEModelTeleIncremental.

Usage
-----
  python3 scripts/ros_telescene_inference.py \
      --exp   /home/gao/data_weight/tele_incremental \
      --epoch 199 \
      --enc_stats /path/to/boxes_stats_tele_enc_trainval.txt \
      --gt_stats  /path/to/boxes_stats_tele_gt_trainval.txt \
      --shape_features_pkl "/media/gao/gao_ssd/3D objects/features.pkl" \
      --obj_dir   "/media/gao/gao_ssd/3D objects" \
      --topic telescene

Caveats
-------
* Training always showed the model the FULL accumulated graph (OLD nodes as
  context). Here only the nodes of the current message are given, so the
  graph is much sparser than in training; only the CLS state carries history.
  Expect some accuracy loss versus offline evaluation.
* No ground truth exists on this topic: only pred_pos/pred_rot/pred_feat.
* Frame/axis convention: centre/bbox are assumed to be in the same Y-up map
  frame as the training data (header.frame_id is "map").
* Degenerate boxes (bbox_min == bbox_max, size 0) do occur in real messages;
  they are fed as-is, size then comes from shape retrieval.
"""
from __future__ import print_function

# ── open3d / trimesh must be imported before torch (GLIBCXX conflict) ───────
try:
    import open3d as o3d
    import trimesh
    import xml.etree.ElementTree as ET
    _VIS_OK = True
except Exception:
    _VIS_OK = False

import argparse
import glob
import math
import os
import sys
import time
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import rclpy
from rclpy.node import Node
from scene_msgs.msg import TeleScene

import torch

from dataset.Tele_dataset_incremental import TeleDatasetSceneGraphIncremental
from model.AE_tele_incremental import Sg2ScAEModelTeleIncremental, N_YAW_BINS
from helpers.util import bool_flag, normalize_box_params

OPE_OLD, OPE_ADD, OPE_UPDATE, OPE_REMOVE = 0, 1, 2, 3

TELE_CLASSES = TeleDatasetSceneGraphIncremental.TELE_CLASSES
TELE_RELATIONS = TeleDatasetSceneGraphIncremental.TELE_RELATIONS
LABEL_NORM = TeleDatasetSceneGraphIncremental.LABEL_NORM
CLASS_TO_ID = {c: i for i, c in enumerate(TELE_CLASSES)}
CLOSE_BY_PRED_ID = {r: i + 1 for i, r in enumerate(TELE_RELATIONS)}['close by']

# Post-hoc validated against GT: scanned every sg_frame_*.json in the dataset
# (12638 kitchenCupboard + 12495 kitchen_counter instances) — yaw is exactly
# 0.0° for every single one, zero deviation. Both are built-in fixtures placed
# flush against a wall in the source scenes, never at an arbitrary angle.
# Forcing the predicted yaw to 0 for these classes is therefore a correction,
# not an approximation — any nonzero value is model noise.
_ZERO_YAW_LABELS = {'kitchen_counter', 'kitchen_cupboard'}


def _apply_zero_yaw_prior(yaw, label):
    return 0.0 if label in _ZERO_YAW_LABELS else yaw


# ─────────────────────────────────────────────────────────────────────────────
# TeleScene delta → model encoder input (stateless, one message = one step)
# ─────────────────────────────────────────────────────────────────────────────
def convert_message(msg, enc_stats_path, enc_std, close_by_thresh, warned_labels):
    """Turn the nodes of ONE TeleScene message into the model's encoder dict.

    Only the nodes present in the message are used (no accumulated graph).
    Returns (enc, meta) — meta[i] = {'id','label','op'} aligned with enc rows —
    or (None, []) if no usable node.
    """
    rows, meta = [], []
    for node in msg.nodes:
        label = LABEL_NORM.get(node.label, node.label)
        if label not in CLASS_TO_ID:
            if label not in warned_labels:
                print(f'[WARN] unknown label "{label}" (id={node.id}) — skipping; '
                      f'not in TELE_CLASSES vocab')
                warned_labels.add(label)
            continue
        if node.ope_id not in (OPE_ADD, OPE_UPDATE, OPE_REMOVE):
            continue
        bmin, bmax, c = node.bbox_min, node.bbox_max, node.center
        dx, dy, dz = bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z
        p7 = np.array([dx, dy, dz, c.x, c.y, c.z, np.arctan2(dz, dx)], dtype=np.float64)
        dc, ds = node.delta_center, node.delta_size
        delta = (np.array([ds.x, ds.y, ds.z, dc.x, dc.y, dc.z, 0.0], dtype=np.float32)
                 if node.ope_id == OPE_UPDATE else np.zeros(7, dtype=np.float32))
        rows.append((CLASS_TO_ID[label], p7, node.ope_id, delta, float(node.confidence)))
        meta.append({'id': node.id, 'label': label, 'op': node.ope_id})
    if not rows:
        return None, []

    N = len(rows)
    boxes_n = np.zeros((N, 7), dtype=np.float32)
    for i, (_, p7, _, _, _) in enumerate(rows):
        b = normalize_box_params(p7.copy(), file=enc_stats_path)
        b[6] = p7[6]   # raw-radian yaw hint, matches dataset _build_dict
        boxes_n[i] = b

    op_types = torch.tensor([r[2] for r in rows], dtype=torch.long)
    delta_raw = np.stack([r[3] for r in rows], 0)
    centers = np.stack([r[1][3:6] for r in rows], 0)
    triples = []
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(centers[i] - centers[j]) < close_by_thresh:
                triples += [[i, CLOSE_BY_PRED_ID, j], [j, CLOSE_BY_PRED_ID, i]]
    triples_t = (torch.tensor(triples, dtype=torch.long) if triples
                 else torch.empty(0, 3, dtype=torch.long))

    enc = {
        'objs': torch.tensor([r[0] for r in rows], dtype=torch.long),
        'boxes': torch.from_numpy(boxes_n),
        'op_types': op_types,
        'delta_x': torch.from_numpy((3.0 * delta_raw / enc_std).astype(np.float32)),
        'confidence': torch.tensor([r[4] for r in rows], dtype=torch.float32),
        'active_mask': (op_types == OPE_ADD) | (op_types == OPE_UPDATE),
        'triples': triples_t,
        'obj_to_flat_step': torch.zeros(N, dtype=torch.long),
        'triple_to_flat_step': torch.zeros(triples_t.size(0), dtype=torch.long),
    }
    return enc, meta


def _make_batch(enc, scan_id):
    return {
        'encoder': enc,
        'step_meta': {
            'scan_id_str': [scan_id],
            # Always 0: each ROS callback builds a brand-new one-step batch, so
            # there is nothing to order it against (see eval_tele_incremental.py
            # run_online_checks / _single_step_batch, which does the same).
            # Using a monotonically growing step index here would make the
            # model's internal K_max loop scan over the whole session history
            # on every message.
            'step_to_k': torch.zeros(1, dtype=torch.long),
        },
    }


def _to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_device(v, device) for v in x]
    return x


def _load_stats(stats_file, dims):
    s = np.loadtxt(stats_file)
    return (torch.tensor(s[0, dims], dtype=torch.float32),
            torch.tensor(s[1, dims], dtype=torch.float32))


def _denorm(x_cpu, mean, std):
    if mean is None:
        return x_cpu
    return x_cpu * std / 3.0 + mean


def _checkpoint_rot_out_dim(state_dict):
    for key, value in state_dict.items():
        if key.startswith('mlp_rot.') and key.endswith('.weight') and torch.is_tensor(value):
            candidates = [(int(k.split('.')[1]), int(v.shape[0]))
                          for k, v in state_dict.items()
                          if k.startswith('mlp_rot.') and k.endswith('.weight')]
            return sorted(candidates)[-1][1]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers (mirrors scripts/eval_tele_incremental.py --slam)
# ─────────────────────────────────────────────────────────────────────────────
_LABEL_COLOR = {
    'cabinet': [0.70, 0.50, 0.30], 'beanbag': [0.90, 0.40, 0.40],
    'chair': [0.40, 0.65, 0.90], 'sofa': [0.55, 0.40, 0.80],
    'table': [0.50, 0.75, 0.50], 'tvstand': [0.80, 0.80, 0.30],
    'fridge': [0.60, 0.85, 0.85], 'bike': [0.85, 0.60, 0.20],
    'indoor_plant': [0.30, 0.70, 0.30], 'tv_screen': [0.85, 0.90, 0.95],
    'kitchen_counter': [0.75, 0.65, 0.55], 'kitchen_cupboard': [0.75, 0.65, 0.55],
    'chest_of_drawers': [0.70, 0.55, 0.35], 'wall_cabinet': [0.70, 0.55, 0.35],
    'clothes_hanger': [0.80, 0.70, 0.60], 'door': [0.65, 0.45, 0.25],
}
_DEFAULT_COLOR = [0.75, 0.75, 0.75]
_OP_COLOR = {OPE_ADD: [0.15, 0.90, 0.25], OPE_UPDATE: [0.20, 0.65, 1.00]}

_URDF_FILES = {
    'chest_of_drawers': 'chestOfDrawers_01.urdf',
    'kitchen_counter': 'kitchen_counter.urdf',
    'kitchen_cupboards': 'kitchenCupboard_01.urdf',
    'fridge': 'fridge.urdf',
}
_LABEL_ALIAS = {'kitchen_cupboard': 'kitchen_cupboards', 'tvstand': 'tv_stand', 'door': 'doors'}
_MESH_CANONICAL_PITCH = {'clothes_hanger': math.radians(90.0)}

_mesh_cache: dict = {}
_urdf_cache: dict = {}


def _canon(label):
    return _LABEL_ALIAS.get(label, label)


def _yaw_mat3(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def _pitch_mat3(pitch):
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def _obb_wireframe(center, L, H, W, yaw_rad, color):
    half = np.array([L / 2, H / 2, W / 2])
    local = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                       [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], np.float64) * half
    pts = local @ _yaw_mat3(yaw_rad).T + np.array(center, np.float64)
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(np.array(edges, np.int32))
    ls.paint_uniform_color(color)
    return ls


def _ground_grid(y=0.0, extent=20.0, step=1.0):
    pts, lines = [], []
    for v in np.arange(-2, extent + step, step):
        i = len(pts); pts += [[-2, y, v], [extent, y, v]]; lines.append([i, i + 1])
        i = len(pts); pts += [[v, y, -2], [v, y, extent]]; lines.append([i, i + 1])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts, np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.array(lines, np.int32))
    ls.paint_uniform_color([0.40, 0.40, 0.40])
    return ls


def _extract_vc(parts):
    chunks = []
    for p in parts:
        try:
            raw = p.visual.to_color().vertex_colors
            if raw.ndim == 1:
                vc = np.tile(raw[:3].astype(np.float64) / 255., (len(p.vertices), 1))
            else:
                vc = raw[:, :3].astype(np.float64) / 255.
            chunks.append(vc)
        except Exception:
            return None
    return np.concatenate(chunks, 0) if chunks else None


def _trimesh_to_o3d(mesh, vc, fallback_color):
    om = o3d.geometry.TriangleMesh()
    om.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, np.float64))
    om.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, np.int32))
    if vc is not None and len(vc) == len(mesh.vertices):
        om.vertex_colors = o3d.utility.Vector3dVector(vc)
    else:
        om.paint_uniform_color(fallback_color)
    om.compute_vertex_normals()
    return om


def _load_mesh_glb(obj_dir, label, glb_path=None):
    """Load a category's mesh. If glb_path is given (e.g. from shape-feature
    retrieval) and exists, load that specific model; otherwise fall back to
    the alphabetically-first .glb in the category directory."""
    use_specific = bool(glb_path) and os.path.exists(glb_path)
    key = f'glb_path:{glb_path}' if use_specific else f'glb:{label}'
    if key in _mesh_cache:
        mesh, vc = _mesh_cache[key]
        return (mesh.copy() if mesh is not None else None), vc
    if use_specific:
        glbs = [glb_path]
    else:
        cat_dir = os.path.join(obj_dir, _canon(label))
        glbs = sorted(glob.glob(os.path.join(cat_dir, '*.glb'))) if os.path.isdir(cat_dir) else []
    if not glbs:
        _mesh_cache[key] = (None, None); return None, None
    loaded = trimesh.load(glbs[0])
    if isinstance(loaded, trimesh.Scene):
        parts = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            _mesh_cache[key] = (None, None); return None, None
        vc = _extract_vc(parts); mesh = trimesh.util.concatenate(parts)
    elif isinstance(loaded, trimesh.Trimesh):
        vc = _extract_vc([loaded]); mesh = loaded
    else:
        _mesh_cache[key] = (None, None); return None, None
    pitch = _MESH_CANONICAL_PITCH.get(label, 0.0)
    if pitch != 0.0:
        v = np.asarray(mesh.vertices, np.float64)
        c = (v.min(0) + v.max(0)) / 2
        mesh.vertices = (v - c) @ _pitch_mat3(pitch).T + c
    _mesh_cache[key] = (mesh, vc)
    return mesh.copy(), vc


def _rpy_to_mat3(rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _origin_T(elem):
    if elem is None:
        return np.eye(4)
    xyz = list(map(float, elem.get('xyz', '0 0 0').split()))
    rpy = list(map(float, elem.get('rpy', '0 0 0').split()))
    T = np.eye(4); T[:3, :3] = _rpy_to_mat3(rpy); T[:3, 3] = xyz
    return T


def _load_urdf_inner(obj_dir, label, urdf_file):
    robot = ET.parse(os.path.join(obj_dir, label, urdf_file)).getroot()
    joint_map = {}
    for j in robot.findall('joint'):
        joint_map[j.find('child').get('link')] = (
            j.find('parent').get('link'), _origin_T(j.find('origin')))
    _SKIP = {'chest_of_drawers', 'kitchen_counter', 'kitchen_cupboards'}
    T_world = {'root': np.eye(4)}
    if label in _SKIP:
        T_world['body'] = np.eye(4)

    def world_T(n):
        if n in T_world:
            return T_world[n]
        parent, Tj = joint_map[n]
        T = world_T(parent) @ Tj; T_world[n] = T; return T

    for lnk in robot.findall('link'):
        world_T(lnk.get('name'))
    parts = []
    for lnk in robot.findall('link'):
        vis = lnk.find('visual')
        if vis is None:
            continue
        mel = vis.find('geometry/mesh')
        if mel is None:
            continue
        gp = os.path.join(obj_dir, label, mel.get('filename'))
        if not os.path.exists(gp):
            continue
        T = world_T(lnk.get('name')) @ _origin_T(vis.find('origin'))
        loaded = trimesh.load(gp)
        if isinstance(loaded, trimesh.Scene):
            sub = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
            if not sub:
                continue
            vc = _extract_vc(sub); mesh = trimesh.util.concatenate(sub)
        elif isinstance(loaded, trimesh.Trimesh):
            vc = _extract_vc([loaded]); mesh = loaded
        else:
            continue
        mesh.apply_transform(T)
        parts.append((mesh, vc))
    return parts


def _load_urdf(obj_dir, label):
    canon = _canon(label)
    if canon not in _urdf_cache:
        try:
            _urdf_cache[canon] = _load_urdf_inner(obj_dir, canon, _URDF_FILES[canon])
        except Exception as e:
            print(f'    [WARN] URDF load failed for {canon}: {e}')
            _urdf_cache[canon] = []
    return [(m.copy(), c) for m, c in _urdf_cache[canon]]


def _place_mesh(mesh, center, yaw_rad):
    v = np.asarray(mesh.vertices, np.float64)
    cur_c = (v.min(0) + v.max(0)) / 2
    mesh.vertices = (v - cur_c) @ _yaw_mat3(yaw_rad).T + np.array(center, np.float64)
    return mesh


def _place_parts(parts, center, L, H, W, yaw_rad):
    all_v = np.concatenate([m.vertices for m, _ in parts], 0)
    cur_c = (all_v.min(0) + all_v.max(0)) / 2
    cur_s = all_v.max(0) - all_v.min(0)
    scale = np.where(cur_s > 1e-6, np.array([L, H, W]) / cur_s, 1.0)
    R = _yaw_mat3(yaw_rad)
    result = []
    for mesh, vc in parts:
        m = mesh.copy()
        m.vertices = ((m.vertices - cur_c) * scale) @ R.T + np.array(center, np.float64)
        result.append((m, vc))
    return result


def build_node_geoms(label, L, H, W, center, yaw_rad, op, obj_dir, glb_path=None):
    """Predicted-pose geometry for one ADD/UPDATE node; OBB wireframe if no asset.

    glb_path, when given, is the specific model retrieved by shape-feature
    nearest-neighbour search (TeleShapeLibrary.retrieve_batch) — rendering it
    instead of the category's first-on-disk file is what makes the displayed
    mesh actually track the predicted shape feature, for categories with more
    than one model (table, chair, wall_cabinet, doors, ...). Ignored for URDF
    categories, which only ever have one asset.
    """
    color = _OP_COLOR.get(op, _LABEL_COLOR.get(label, _DEFAULT_COLOR))
    geoms = []
    if obj_dir:
        if _canon(label) in _URDF_FILES:
            parts = _load_urdf(obj_dir, label)
            if parts:
                for m, vc in _place_parts(parts, center, L, H, W, yaw_rad):
                    geoms.append(_trimesh_to_o3d(m, vc, color))
                return geoms
        mesh, vc = _load_mesh_glb(obj_dir, label, glb_path=glb_path)
        if mesh is not None:
            geoms.append(_trimesh_to_o3d(_place_mesh(mesh, center, yaw_rad), vc, color))
            return geoms
    geoms.append(_obb_wireframe(center, L, H, W, yaw_rad, color))
    return geoms


# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 node
# ─────────────────────────────────────────────────────────────────────────────
class TeleIncrementalInferenceNode(Node):
    def __init__(self, args):
        super().__init__('tele_incremental_inference')
        self.args = args
        self.device = torch.device(args.device)
        self.scan_id = args.scan_id
        self.step_counter = 0
        self._warned_labels: set = set()

        _es = np.loadtxt(args.enc_stats)
        self.enc_std7 = _es[1].astype(np.float32)
        self.enc_size_mu, self.enc_size_sg = _load_stats(args.enc_stats, slice(0, 3))
        self.gt_pos_mu, self.gt_pos_sg = _load_stats(args.gt_stats, slice(3, 6))

        train_args = {}
        args_json = os.path.join(args.exp, 'args.json')
        if os.path.exists(args_json):
            import json
            with open(args_json) as f:
                train_args = json.load(f)

        ckpt_path = os.path.join(args.exp, 'checkpoint', f'model_{args.epoch}.pth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        rot_out_dim = _checkpoint_rot_out_dim(ckpt['model'])
        if rot_out_dim is not None and rot_out_dim != N_YAW_BINS:
            raise RuntimeError(
                f'Checkpoint rotation head has output dim {rot_out_dim}; this '
                f'node supports only the current {N_YAW_BINS}-bin yaw-residual head.')

        vocab = {
            'object_idx_to_name': {i: c for i, c in enumerate(TELE_CLASSES)},
            'pred_idx_to_name': {i: r for i, r in enumerate(TELE_RELATIONS)},
        }
        self.model = Sg2ScAEModelTeleIncremental(
            vocab=vocab,
            node_embedding_dim=train_args.get('node_dim', 128),
            gconv_num_layers=train_args.get('num_layers', 5),
            shape_feat_dim=train_args.get('shape_feat_dim', 768),
            residual=train_args.get('residual', True),
            gconv_pooling=train_args.get('pooling', 'avg'),
        )
        self.model.load_state_dict(ckpt['model'])
        self.model.to(self.device)
        self.model.eval()
        self.get_logger().info(f'Loaded checkpoint {ckpt_path} (epoch {ckpt["epoch"]})')

        self.shape_lib = None
        shape_pkl = args.shape_features_pkl or train_args.get('shape_features_pkl', '')
        if shape_pkl and os.path.exists(shape_pkl):
            from helpers.tele_shape_features import TeleShapeLibrary
            self.shape_lib = TeleShapeLibrary(shape_pkl)
            self.get_logger().info(f'Shape library loaded: {len(self.shape_lib._all_paths)} models')

        self.vis = None
        self.persistent_geoms: dict = {}   # node_id -> list[o3d geometry]
        self._smoothed: dict = {}          # node_id -> {'center': np(3,), 'cos': f, 'sin': f}
        self._last_geom_pose: dict = {}    # node_id -> (center np(3,), yaw) actually rendered
        if args.visualize:
            if not _VIS_OK:
                self.get_logger().warn('open3d/trimesh import failed — disabling --visualize')
            else:
                self.vis = o3d.visualization.VisualizerWithKeyCallback()
                self.vis.create_window(window_name='Increment3D — live', width=1280, height=800)
                opt = self.vis.get_render_option()
                opt.background_color = np.array([0.08, 0.08, 0.12])
                opt.point_size = 4.0
                self.vis.add_geometry(_ground_grid(), reset_bounding_box=True)
                self.vis.register_key_callback(ord('Q'), self._on_quit)
                self._quit = False
                self._render_timer = self.create_timer(1.0 / 30.0, self._render_tick)

        self.sub = self.create_subscription(TeleScene, args.topic, self.on_scene, 10)
        self.get_logger().info(f'Listening on /{args.topic} ...')

    def _on_quit(self, vis):
        self._quit = True
        return False

    def _render_tick(self):
        if self.vis is None or self._quit:
            return
        self.vis.poll_events()
        self.vis.update_renderer()
        if self._quit:
            rclpy.shutdown()

    # ------------------------------------------------------------------
    def on_scene(self, msg):
        t_recv = self.get_clock().now()
        latency_ms = (t_recv - rclpy.time.Time.from_msg(msg.header.stamp)).nanoseconds / 1e6

        enc, meta = convert_message(msg, self.args.enc_stats, self.enc_std7,
                                    self.args.close_by_thresh, self._warned_labels)
        if enc is None:
            return
        removal_ids = [m['id'] for m in meta if m['op'] == OPE_REMOVE]
        N = len(meta)

        batch = _to_device(_make_batch(enc, self.scan_id), self.device)
        with torch.no_grad():
            preds = self.model(batch, reset_states=(self.step_counter == 0))
        self.step_counter += 1

        n_active = int(enc['active_mask'].sum())
        self.get_logger().info(
            f'frame={msg.frame}  latency={latency_ms:.1f}ms  '
            f'nodes={N} active={n_active} removed={len(removal_ids)}')

        active_idx = enc['active_mask'].nonzero(as_tuple=True)[0]
        results = []
        if active_idx.numel() > 0:
            pred_pos_n = preds['pred_pos'][active_idx].cpu()
            pred_pos_w = _denorm(pred_pos_n, self.gt_pos_mu, self.gt_pos_sg)
            pred_yaw = preds['pred_rot'][active_idx, 0].cpu()

            # Render size = the retrieved 3D-model-library size, treating every
            # category as a fixed-size real furniture piece (the network doesn't
            # predict size at all). The robot's own observed bbox is only used as a
            # fallback when a model has no size on record (see TeleShapeLibrary).
            enc_size_n = enc['boxes'][active_idx, :3].cpu()
            enc_size_w = _denorm(enc_size_n, self.enc_size_mu, self.enc_size_sg)
            pred_glb_paths = [None] * active_idx.numel()
            if self.shape_lib is not None:
                act_labels = [meta[i]['label'] for i in active_idx.tolist()]
                pred_feat = preds['pred_feat'][active_idx].cpu()
                pred_size_w = self.shape_lib.retrieve_sizes_batch(
                    pred_feat, tele_labels=act_labels, fallback=enc_size_w)
                # which specific model (within the category) the predicted shape
                # feature is nearest to — used so the displayed mesh actually
                # tracks the prediction instead of always showing the same
                # first-on-disk model for that category.
                pred_glb_paths = self.shape_lib.retrieve_batch(pred_feat, tele_labels=act_labels)
            else:
                pred_size_w = enc_size_w

            for k, i in enumerate(active_idx.tolist()):
                op, label = meta[i]['op'], meta[i]['label']
                cx, cy, cz = pred_pos_w[k].tolist()
                yaw = _apply_zero_yaw_prior(float(pred_yaw[k]), label)
                L, H, W = pred_size_w[k].tolist()
                results.append({'node_id': meta[i]['id'], 'op': op, 'label': label,
                                 'center': [cx, cy, cz], 'yaw': yaw, 'L': L, 'H': H, 'W': W,
                                 'glb_path': pred_glb_paths[k] or None})
                op_name = {OPE_ADD: 'ADD', OPE_UPDATE: 'UPD'}[op]
                self.get_logger().info(
                    f'  [{op_name}] {label:<18s} pos=({cx:+.2f},{cy:+.2f},{cz:+.2f}) '
                    f'yaw={math.degrees(yaw):+.1f}deg  size=({L:.2f},{H:.2f},{W:.2f})')

        if self.vis is not None:
            self._update_visualization(results, removal_ids)

    def _smooth_pose(self, nid, raw_center, raw_yaw, is_add):
        """EMA-smooth a node's pose across messages (yaw blended in cos/sin
        space to handle wraparound correctly). is_add resets the filter —
        an ADD is a fresh observation, not a continuation."""
        raw_center = np.asarray(raw_center, dtype=np.float64)
        if is_add or nid not in self._smoothed:
            self._smoothed[nid] = {'center': raw_center,
                                    'cos': math.cos(raw_yaw), 'sin': math.sin(raw_yaw)}
        else:
            a = self.args.smooth_alpha
            s = self._smoothed[nid]
            s['center'] = a * raw_center + (1 - a) * s['center']
            s['cos'] = a * math.cos(raw_yaw) + (1 - a) * s['cos']
            s['sin'] = a * math.sin(raw_yaw) + (1 - a) * s['sin']
        s = self._smoothed[nid]
        return s['center'].copy(), math.atan2(s['sin'], s['cos'])

    def _update_visualization(self, results, removal_ids):
        for nid in removal_ids:
            for g in self.persistent_geoms.pop(nid, []):
                self.vis.remove_geometry(g, reset_bounding_box=False)
            self._smoothed.pop(nid, None)
            self._last_geom_pose.pop(nid, None)
        for r in results:
            nid = r['node_id']
            center, yaw = self._smooth_pose(nid, r['center'], r['yaw'], is_add=(r['op'] == OPE_ADD))

            last = self._last_geom_pose.get(nid)
            if last is not None:
                dpos = float(np.linalg.norm(last[0] - center))
                dyaw = math.degrees(abs(math.atan2(math.sin(yaw - last[1]), math.cos(yaw - last[1]))))
                if dpos < self.args.pose_deadband_m and dyaw < self.args.yaw_deadband_deg:
                    continue   # change too small to be visible — skip the geometry rebuild

            for g in self.persistent_geoms.pop(nid, []):
                self.vis.remove_geometry(g, reset_bounding_box=False)
            geoms = build_node_geoms(r['label'], r['L'], r['H'], r['W'],
                                      center, yaw, r['op'], self.args.obj_dir,
                                      glb_path=r.get('glb_path'))
            for g in geoms:
                self.vis.add_geometry(g, reset_bounding_box=False)
            self.persistent_geoms[nid] = geoms
            self._last_geom_pose[nid] = (center, yaw)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--exp', required=True, help='training experiment dir (has checkpoint/, args.json)')
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--enc_stats', type=str, required=True, help='robot-SG normalization stats file')
    p.add_argument('--gt_stats', type=str, required=True, help='GT normalization stats file (denorm target)')
    p.add_argument('--shape_features_pkl', type=str, default='',
                    help='3D model library features.pkl (optional; enables shape-based size retrieval)')
    p.add_argument('--obj_dir', type=str, default='', help='3D objects directory for mesh visualisation')
    p.add_argument('--topic', type=str, default='telescene')
    p.add_argument('--scan_id', type=str, default='live')
    p.add_argument('--close_by_thresh', type=float, default=2.0)
    p.add_argument('--visualize', type=bool_flag, default=True)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--smooth_alpha', type=float, default=0.35,
                    help='EMA weight given to each new prediction, in (0,1]. '
                         'Lower = smoother/more lag, 1.0 = no smoothing (raw output).')
    p.add_argument('--pose_deadband_m', type=float, default=0.02,
                    help='skip re-rendering a node if its smoothed position moved less than this (m)')
    p.add_argument('--yaw_deadband_deg', type=float, default=3.0,
                    help='skip re-rendering a node if its smoothed yaw moved less than this (deg)')
    return p


def main():
    parser = build_arg_parser()
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = TeleIncrementalInferenceNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.vis is not None:
            node.vis.destroy_window()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
