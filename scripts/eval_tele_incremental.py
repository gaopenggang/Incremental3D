"""
Evaluation script for Sg2ScAEModelTeleIncremental.

Metrics (computed over active-node matched pairs):
  - Position L1  (m, if box_stats supplied; normalised otherwise)
  - Yaw error    (degrees)
  - Orientation % (yaw_err / 360)

Visualisation (--visualize True):
  Launches an interactive Open3D window per sequence showing the predicted 3D
  scene step-by-step.  Keys:  Space / → : next step    ← : prev step    Q : quit

  Colour coding:
    grey   wireframe  — OLD (context) nodes at enc-observed position
    green  mesh       — ADD  nodes at predicted position
    cyan   mesh       — UPDATE nodes at predicted position
    red    wireframe  — REMOVE nodes (vanishing this step)
    orange wireframe  — GT boxes for active nodes (ground truth)

Usage:
  python scripts/eval_tele_incremental.py \
      --exp   /home/gao/data_weight/tele_incremental \
      --epoch 99 \
      [--visualize True] [--obj_dir "/media/gao/gao_ssd/3D objects"]
"""
from __future__ import print_function

# ── open3d / trimesh MUST be imported before torch (GLIBCXX conflict) ─────────
try:
    import open3d as o3d
    import trimesh
    import xml.etree.ElementTree as ET
    _VIS_OK = True
except Exception:
    _VIS_OK = False

import sys, os, json, math, time, argparse, random, glob
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.Tele_dataset_incremental import (
    TeleDatasetSceneGraphIncremental, LengthGroupedCurriculumSampler)
from model.AE_tele_incremental import Sg2ScAEModelTeleIncremental, N_YAW_BINS
from helpers.util import bool_flag, denormalize_box_params
from helpers.metrics_3dfront import validate_box_accuracy, validate_shape_accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--robot_root', default='/media/gao/gao_ssd/3D scene graph')
parser.add_argument('--gt_root',    default='/media/gao/gao_ssd/3D Scene')
parser.add_argument('--exp',        default='/home/gao/data_weight/tele_incremental')
parser.add_argument('--epoch',      type=int, default=100)
parser.add_argument('--split',      type=str, default='val')
parser.add_argument('--visualize',  type=bool_flag, default=False)
parser.add_argument('--slam',       type=bool_flag, default=False,
                    help='SLAM-style incremental vis at 20 Hz (implies --visualize)')
parser.add_argument('--obj_dir',    default='/media/gao/gao_ssd/3D objects',
                    help='3D objects directory (used when --visualize / --slam)')
parser.add_argument('--enc_stats', type=str, default='',
                    help='encoder normalization stats file')
parser.add_argument('--gt_stats',  type=str, default='',
                    help='GT normalization stats file (used to denormalise predictions)')
parser.add_argument('--shape_features_pkl', type=str,
                    default='/media/gao/gao_ssd/3D objects/features.pkl',
                    help='path to 3D library features.pkl for size retrieval')
parser.add_argument('--seq', type=str, default='',
                    help='only process/visualize this scan_id (partial match ok)')
parser.add_argument('--splits_dir', type=str, default='',
                    help='directory with train/val/test_sequences.txt')
parser.add_argument('--online_benchmark', type=bool_flag, default=False,
                    help='benchmark online one-step-at-a-time model latency')
parser.add_argument('--consistency_test', type=bool_flag, default=False,
                    help='compare full-sequence forward vs online step-by-step forward')
parser.add_argument('--online_warmup_steps', type=int, default=5,
                    help='number of online step forwards to exclude from latency stats')
parser.add_argument('--online_max_sequences', type=int, default=0,
                    help='max sequences for online benchmark/consistency; 0 = all')
parser.add_argument('--bench_pipeline', type=bool_flag, default=False,
                    help='measure full pipeline latency per step: GCN + retrieval + mesh build '
                         '(no interactive window; requires --obj_dir)')
args = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────
RAD2DEG = 57.29577951308232

def to_cuda(x, device='cuda'):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_cuda(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [to_cuda(v, device) for v in x]
    return x


def circular_yaw_err_deg(pred_rad, gt_rad):
    """Mean absolute circular yaw error in degrees."""
    diff = (pred_rad - gt_rad).abs()
    diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
    return (diff * RAD2DEG).mean().item()


# n-fold rotational symmetry per class label.
# 0 = excluded from rotation evaluation (continuous symmetry or undefined GT yaw),
# 1 = no symmetry, 2/4 = discrete n-fold (table 2-fold).
_SYM_FOLD = {
    'indoor_plant': 0,   # fully rotationally symmetric; GT yaw is arbitrary
    'beanbag':      0,   # near-circular; GT yaw has no semantic meaning
    'table':        2,   # 2-fold (0° ≡ 180°)
    'wall_cabinet': 2,   # 2-fold (front/back equivalent when against wall)
}

# Post-hoc validated against GT: scanned every sg_frame_*.json in the dataset
# (12638 kitchenCupboard + 12495 kitchen_counter instances) — yaw is exactly
# 0.0° for every single one, zero deviation. Both are built-in fixtures placed
# flush against a wall in the source scenes, never at an arbitrary angle.
# Forcing the predicted yaw to 0 for these classes is therefore a correction,
# not an approximation — any nonzero value is model noise.
_ZERO_YAW_LABELS = {'kitchen_counter', 'kitchen_cupboard'}


def _apply_zero_yaw_prior(yaw, label):
    return 0.0 if label in _ZERO_YAW_LABELS else yaw


def sym_yaw_err_rad(pred, gt, folds):
    """Symmetry-aware per-sample yaw error (radians).

    pred, gt : (M,) tensors
    folds    : (M,) int tensor – n_fold per sample
    Returns  : (M,) non-negative error tensor in radians.
    """
    errs = torch.zeros_like(pred)
    for fold_val in folds.unique().tolist():
        mask = (folds == fold_val)
        if fold_val == 0:
            continue   # continuous symmetry: error = 0 by definition
        period = 2 * math.pi / fold_val
        d = (pred[mask] - gt[mask]) % (2 * math.pi)
        d = d % period
        d = torch.where(d > period / 2, period - d, d)
        errs[mask] = d
    return errs


def _load_stats(stats_file, dims):
    """Load mean/std tensors for selected dims from a 2-row stats file."""
    if not stats_file:
        return None, None
    s = np.loadtxt(stats_file)
    return (torch.tensor(s[0, dims], dtype=torch.float32),
            torch.tensor(s[1, dims], dtype=torch.float32))


def _require_stats(path, arg_name):
    if not path:
        raise ValueError(
            f'{arg_name} is required. Run scripts/compute_tele_box_stats.py '
            'and pass the generated uncentered stats file.')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{arg_name} not found: {path}')
    return path


def _denorm(x_cpu, mean, std):
    """Inverse normalisation: x_real = x_norm * std / 3 + mean."""
    if mean is None:
        return x_cpu
    return x_cpu * std / 3.0 + mean


def _checkpoint_rot_out_dim(state_dict):
    """Return the output dim of the checkpoint's final mlp_rot Linear layer."""
    candidates = []
    for key, value in state_dict.items():
        if not (key.startswith('mlp_rot.') and key.endswith('.weight')):
            continue
        if not torch.is_tensor(value) or value.ndim != 2:
            continue
        parts = key.split('.')
        try:
            layer_idx = int(parts[1])
        except (IndexError, ValueError):
            layer_idx = -1
        candidates.append((layer_idx, key, int(value.shape[0])))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: (x[0], x[1]))[-1][2]


def _sync_if_cuda(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def _single_step_batch(dataset, sample, step_idx):
    """Wrap one prebuilt step as a one-step model batch."""
    one = {
        'scan_id': sample['scan_id'],
        'steps': [sample['steps'][step_idx]],
        'scene_center': sample.get('scene_center', [0.0, 0.0, 0.0]),
    }
    return dataset.collate_fn_inc([one])


def _active_indices(batch_gpu, flat_sid=None):
    enc = batch_gpu['encoder']
    mask = enc['active_mask']
    if flat_sid is not None:
        mask = mask & (enc['obj_to_flat_step'] == flat_sid)
    return mask.nonzero(as_tuple=True)[0]


def _max_abs(a, b):
    if a.numel() == 0:
        return 0.0
    return float((a - b).abs().max().detach().cpu())


def run_online_checks(dataset, model, device):
    """
    Optional diagnostics:
      - online_benchmark: one forward per step with persistent scene state
      - consistency_test: compare online outputs with one full-sequence forward

    Timings are model-only: one-step batch construction and CPU->device transfer
    happen before the measured region.
    """
    if not (args.online_benchmark or args.consistency_test):
        return {}

    n_seq = len(dataset)
    if args.online_max_sequences > 0:
        n_seq = min(n_seq, args.online_max_sequences)

    times = []
    total_steps = 0
    measured_steps = 0
    total_pairs = 0
    compared_nodes = 0
    mismatch_steps = 0
    max_pos = 0.0
    max_rot = 0.0
    max_feat = 0.0

    print('\n--- Online diagnostics ---')
    print(f'Running {n_seq} sequence(s); timing excludes batch construction and transfer.')

    for seq_idx in range(n_seq):
        sample = dataset[seq_idx]
        steps = sample.get('steps', [])
        if not steps:
            continue

        full_gpu = None
        full_preds = None
        if args.consistency_test:
            full_gpu = to_cuda(dataset.collate_fn_inc([sample]), device)
            with torch.no_grad():
                full_preds = model(full_gpu, reset_states=True)

        model.reset_scene_states()
        for step_idx in range(len(steps)):
            one_cpu = _single_step_batch(dataset, sample, step_idx)
            one_gpu = to_cuda(one_cpu, device)
            total_steps += 1
            total_pairs += sum(len(m) for m in one_cpu['active_matches'])

            _sync_if_cuda(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                one_preds = model(one_gpu, reset_states=(step_idx == 0))
            _sync_if_cuda(device)
            dt = time.perf_counter() - t0

            if args.online_benchmark and total_steps > args.online_warmup_steps:
                times.append(dt)
                measured_steps += 1

            if args.consistency_test:
                full_idx = _active_indices(full_gpu, flat_sid=step_idx)
                one_idx = _active_indices(one_gpu)
                if full_idx.numel() != one_idx.numel():
                    mismatch_steps += 1
                    continue
                if one_idx.numel() == 0:
                    continue
                compared_nodes += int(one_idx.numel())
                max_pos = max(max_pos, _max_abs(
                    full_preds['pred_pos'][full_idx],
                    one_preds['pred_pos'][one_idx]))
                max_rot = max(max_rot, _max_abs(
                    full_preds['pred_rot'][full_idx],
                    one_preds['pred_rot'][one_idx]))
                max_feat = max(max_feat, _max_abs(
                    full_preds['pred_feat'][full_idx],
                    one_preds['pred_feat'][one_idx]))

    out = {}
    if args.online_benchmark:
        if times:
            arr = np.asarray(times, dtype=np.float64) * 1000.0
            stats = {
                'online_model_ms_mean': float(arr.mean()),
                'online_model_ms_median': float(np.median(arr)),
                'online_model_ms_p95': float(np.percentile(arr, 95)),
                'online_model_ms_p99': float(np.percentile(arr, 99)),
                'online_model_steps': int(measured_steps),
                'online_total_steps': int(total_steps),
                'online_total_active_pairs': int(total_pairs),
            }
            print('Online model-only latency:')
            print(f'  steps measured: {measured_steps} / {total_steps}'
                  f'  (warmup={args.online_warmup_steps})')
            print(f'  mean={stats["online_model_ms_mean"]:.3f} ms'
                  f'  median={stats["online_model_ms_median"]:.3f} ms'
                  f'  p95={stats["online_model_ms_p95"]:.3f} ms'
                  f'  p99={stats["online_model_ms_p99"]:.3f} ms')
            out.update(stats)
        else:
            print('Online benchmark: no measured steps after warmup.')

    if args.consistency_test:
        stats = {
            'consistency_nodes': int(compared_nodes),
            'consistency_mismatch_steps': int(mismatch_steps),
            'consistency_max_pos_abs': float(max_pos),
            'consistency_max_rot_abs': float(max_rot),
            'consistency_max_feat_abs': float(max_feat),
        }
        print('Full-sequence vs online consistency:')
        print(f'  compared active nodes: {compared_nodes}'
              f'  mismatch steps: {mismatch_steps}')
        print(f'  max_abs pos={max_pos:.6g}'
              f'  rot={max_rot:.6g}'
              f'  feat={max_feat:.6g}')
        out.update(stats)

    model.reset_scene_states()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation utilities  (only active when --visualize True and _VIS_OK)
# ─────────────────────────────────────────────────────────────────────────────

_LABEL_COLOR = {
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
_DEFAULT_COLOR = [0.75, 0.75, 0.75]

# op-type colour override for active nodes
_OP_COLOR = {
    0: None,                    # OLD    → use label colour, wireframe
    1: [0.15, 0.90, 0.25],     # ADD    → green mesh
    2: [0.20, 0.65, 1.00],     # UPDATE → cyan mesh
    3: [1.00, 0.20, 0.20],     # REMOVE → red wireframe
}

_URDF_FILES = {
    'chest_of_drawers':  'chestOfDrawers_01.urdf',
    'kitchen_counter':   'kitchen_counter.urdf',
    'kitchen_cupboards': 'kitchenCupboard_01.urdf',
    'fridge':            'fridge.urdf',
}

# Dataset label → canonical label used as key in _URDF_FILES and as obj_dir sub-folder.
# Mirrors visualize_gt_3d.py's LABEL_ALIAS for the same mismatches.
_LABEL_ALIAS = {
    'kitchen_cupboard': 'kitchen_cupboards',
    'tvstand':          'tv_stand',
    'door':             'doors',
}

# Canonical pitch (radians, around X axis, Rx convention) applied to GLB mesh
# at load time so that the mesh rests in the correct orientation before yaw is applied.
# Derived from GT orientation quaternions: all instances of these classes share
# the same non-zero pitch in the Habitat scene.
_MESH_CANONICAL_PITCH = {
    'clothes_hanger': math.radians(90.0),
}


def _canon(label: str) -> str:
    """Resolve a dataset label to its canonical form for _URDF_FILES lookup."""
    return _LABEL_ALIAS.get(label, label)

_mesh_cache: dict = {}
_urdf_cache: dict = {}


def _yaw_mat3(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def _pitch_mat3(pitch: float) -> np.ndarray:
    """Rotation around X axis (Rx)."""
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def _euler_yxz_mat3(yaw, pitch, roll):
    """Full YXZ Euler rotation (scipy convention used by _quat_to_euler_yup).
    R = R_Y(yaw) @ R_X(pitch) @ R_Z(roll)
    """
    Ry = _yaw_mat3(yaw)
    cx, sx = np.cos(pitch), np.sin(pitch)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=np.float64)
    cz, sz = np.cos(roll), np.sin(roll)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float64)
    return Ry @ Rx @ Rz


def _obb_wireframe(center, L, H, W, yaw_rad, color, pitch=0.0, roll=0.0):
    half = np.array([L / 2, H / 2, W / 2])
    local = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                       [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]],
                     dtype=np.float64) * half
    R   = _euler_yxz_mat3(yaw_rad, pitch, roll)
    pts = local @ R.T + np.array(center, np.float64)
    edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],
             [0,4],[1,5],[2,6],[3,7]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(np.array(edges, np.int32))
    ls.paint_uniform_color(color)
    return ls


def _ground_grid(y=0.0, extent=20.0, step=1.0):
    pts, lines = [], []
    for v in np.arange(-2, extent + step, step):
        i = len(pts); pts += [[-2,y,v],[extent,y,v]]; lines.append([i,i+1])
        i = len(pts); pts += [[v,y,-2],[v,y,extent]]; lines.append([i,i+1])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts, np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines, np.int32))
    ls.paint_uniform_color([0.40, 0.40, 0.40])
    return ls


def _extract_vc(parts):
    chunks = []
    for p in parts:
        try:
            raw = p.visual.to_color().vertex_colors
            if raw.ndim == 1:
                rgb = raw[:3].astype(np.float64) / 255.
                vc  = np.tile(rgb, (len(p.vertices), 1))
            else:
                vc = raw[:, :3].astype(np.float64) / 255.
            chunks.append(vc)
        except Exception:
            return None
    return np.concatenate(chunks, 0) if chunks else None


def _trimesh_to_o3d(mesh, vc, fallback_color):
    om = o3d.geometry.TriangleMesh()
    om.vertices  = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, np.float64))
    om.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces,    np.int32))
    if vc is not None and len(vc) == len(mesh.vertices):
        om.vertex_colors = o3d.utility.Vector3dVector(vc)
    else:
        om.paint_uniform_color(fallback_color)
    om.compute_vertex_normals()
    return om


def _load_mesh_glb(obj_dir: str, label: str, glb_path: str = None):
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

    # Apply canonical pitch pre-rotation so the mesh is in the correct
    # horizontal/vertical orientation before yaw is applied at placement time.
    pitch = _MESH_CANONICAL_PITCH.get(label, 0.0)
    if pitch != 0.0:
        v = np.asarray(mesh.vertices, np.float64)
        c = (v.min(0) + v.max(0)) / 2
        mesh.vertices = (v - c) @ _pitch_mat3(pitch).T + c

    _mesh_cache[key] = (mesh, vc)
    return mesh.copy(), vc


def _rpy_to_mat3(rpy):
    r, p, y = rpy
    Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
    return Rz @ Ry @ Rx


def _origin_T(elem):
    if elem is None: return np.eye(4)
    xyz = list(map(float, elem.get('xyz', '0 0 0').split()))
    rpy = list(map(float, elem.get('rpy', '0 0 0').split()))
    T = np.eye(4); T[:3, :3] = _rpy_to_mat3(rpy); T[:3, 3] = xyz
    return T


def _load_urdf_inner(obj_dir: str, label: str, urdf_file: str):
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
        if n in T_world: return T_world[n]
        parent, Tj = joint_map[n]
        T = world_T(parent) @ Tj; T_world[n] = T; return T

    for lnk in robot.findall('link'):
        world_T(lnk.get('name'))

    parts = []
    for lnk in robot.findall('link'):
        vis = lnk.find('visual')
        if vis is None: continue
        mel = vis.find('geometry/mesh')
        if mel is None: continue
        gp = os.path.join(obj_dir, label, mel.get('filename'))
        if not os.path.exists(gp): continue
        T = world_T(lnk.get('name')) @ _origin_T(vis.find('origin'))
        loaded = trimesh.load(gp)
        if isinstance(loaded, trimesh.Scene):
            sub = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
            if not sub: continue
            vc = _extract_vc(sub); mesh = trimesh.util.concatenate(sub)
        elif isinstance(loaded, trimesh.Trimesh):
            vc = _extract_vc([loaded]); mesh = loaded
        else: continue
        mesh.apply_transform(T)
        parts.append((mesh, vc))
    return parts


def _load_urdf(obj_dir: str, label: str):
    canon = _canon(label)
    if canon not in _urdf_cache:
        try:
            _urdf_cache[canon] = _load_urdf_inner(obj_dir, canon, _URDF_FILES[canon])
        except Exception as e:
            print(f'    [WARN] URDF load failed for {canon}: {e}')
            _urdf_cache[canon] = []
    return [(m.copy(), c) for m, c in _urdf_cache[canon]]


def _place_mesh(mesh, center, yaw_rad):
    """Center the mesh at its geometric centroid, apply yaw, translate to center.

    GT JSON convention: L = mesh Z-axis extent, W = mesh X-axis extent.
    OBB convention (after fix): W along world X, L along world Z.
    With both conventions aligned there is no need for any pre-rotation —
    the mesh natural orientation already agrees with the corrected OBB.
    """
    v = np.asarray(mesh.vertices, np.float64)
    cur_c = (v.min(0) + v.max(0)) / 2
    mesh.vertices = (v - cur_c) @ _yaw_mat3(yaw_rad).T + np.array(center, np.float64)
    return mesh


def _place_parts(parts, center, L, H, W, yaw_rad):
    """Scale + rotate assembled URDF parts to L×H×W, translate to center."""
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


def _build_step_geoms(nodes_vis, gt_vis, obj_dir):
    """Build Open3D geometry list for one visualisation step.

    nodes_vis : [{label, op, L, H, W, cx, cy, cz, yaw}]
    gt_vis    : [{L, H, W, cx, cy, cz}]   L/H/W are world AABB extents
    """
    geoms = [_ground_grid()]

    for node in nodes_vis:
        label  = node['label']
        L, H, W = node['L'], node['H'], node['W']
        center = [node['cx'], node['cy'], node['cz']]
        yaw    = node['yaw']
        op     = node['op']

        label_color = _LABEL_COLOR.get(label, _DEFAULT_COLOR)
        op_color    = _OP_COLOR.get(op)
        color       = op_color if op_color is not None else label_color

        if op in (0, 3):
            pass  # OLD / REMOVE → no geometry
        else:
            # ADD / UPDATE → show mesh at predicted pose
            placed = False
            if _canon(label) in _URDF_FILES:
                parts = _load_urdf(obj_dir, label)
                if parts:
                    for m, vc in _place_parts(parts, center, L, H, W, yaw):
                        geoms.append(_trimesh_to_o3d(m, vc, color))
                    placed = True
            if not placed:
                mesh, vc = _load_mesh_glb(obj_dir, label, glb_path=node.get('glb_path'))
                if mesh is not None:
                    placed_mesh = _place_mesh(mesh, center, yaw)
                    geoms.append(_trimesh_to_o3d(placed_mesh, vc, color))
                    placed = True

    return geoms


def _show_sequence(scan_id: str, vis_steps: list, obj_dir: str):
    """Interactive Open3D step viewer for one predicted sequence."""
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f'Pred 3D Scene — {scan_id}', width=1280, height=800)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.08, 0.08, 0.12])
    opt.point_size = 4.0

    state = {'idx': 0, 'geoms': [], 'first': True}

    def _console_print():
        step = vis_steps[state['idx']]
        op_name = {0: 'OLD', 1: 'ADD', 2: 'UPD', 3: 'REM'}
        print(f'\n  [{scan_id}]  step {step["step_id"]:02d}  '
              f'[{state["idx"]+1}/{len(vis_steps)}]')
        for n in step['nodes']:
            print(f'    [{op_name.get(n["op"],"?")}] {n["label"]:<20s}  '
                  f'pos=({n["cx"]:+.2f},{n["cy"]:+.2f},{n["cz"]:+.2f})  '
                  f'yaw={math.degrees(n["yaw"]):+.1f}°  '
                  f'L={n["L"]:.2f} H={n["H"]:.2f} W={n["W"]:.2f}')
        if step['gt']:
            print(f'  GT ({len(step["gt"])} boxes):')
            for g in step['gt']:
                print(f'    [GT ] pos=({g["cx"]:+.2f},{g["cy"]:+.2f},{g["cz"]:+.2f})  '
                      f'L={g["L"]:.2f} H={g["H"]:.2f} W={g["W"]:.2f}')

    def refresh(vis):
        for g in state['geoms']:
            vis.remove_geometry(g, reset_bounding_box=False)
        state['geoms'].clear()
        step  = vis_steps[state['idx']]
        geoms = _build_step_geoms(step['nodes'], step['gt'], obj_dir)
        reset = state['first']
        for g in geoms:
            vis.add_geometry(g, reset_bounding_box=reset)
        state['geoms'] = geoms
        state['first'] = False
        _console_print()
        return False

    def next_step(vis):
        state['idx'] = min(state['idx'] + 1, len(vis_steps) - 1)
        refresh(vis); return False

    def prev_step(vis):
        state['idx'] = max(state['idx'] - 1, 0)
        refresh(vis); return False

    vis.register_key_callback(262, next_step)   # Right arrow
    vis.register_key_callback(263, prev_step)   # Left arrow
    vis.register_key_callback(32,  next_step)   # Space

    print('\n  Keys:  Space / → : next step    ← : prev step    Q : quit')
    refresh(vis)
    vis.run()
    vis.destroy_window()


def _build_single_geoms(node: dict, obj_dir: str) -> list:
    """Build Open3D geometries for a single ADD/UPDATE node at its predicted pose."""
    label  = node['label']
    L, H, W = node['L'], node['H'], node['W']
    center = [node['cx'], node['cy'], node['cz']]
    yaw    = node['yaw']
    op     = node['op']

    color = [0.75, 0.75, 0.75]   # neutral grey fallback; vertex colors used when available

    geoms = []
    if op in (0, 3):
        return geoms

    placed = False
    if _canon(label) in _URDF_FILES:
        parts = _load_urdf(obj_dir, label)
        if parts:
            for m, vc in _place_parts(parts, center, L, H, W, yaw):
                geoms.append(_trimesh_to_o3d(m, vc, color))
            placed = True
    if not placed:
        mesh, vc = _load_mesh_glb(obj_dir, label, glb_path=node.get('glb_path'))
        if mesh is not None:
            pm = _place_mesh(mesh, center, yaw)
            geoms.append(_trimesh_to_o3d(pm, vc, color))
            placed = True
    return geoms


def _show_slam(scan_id: str, vis_steps: list, obj_dir: str, hz: float = 20.0):
    """SLAM-style step-by-step viewer.

    Objects persist across steps once placed:
      ADD    → mesh appears at predicted position (green)
      UPDATE → mesh moves to new predicted position (cyan)
      OLD    → stays in scene unchanged (already placed in a prior step)
      REMOVE → removed from scene

    GT orange wireframes are shown only for the current step's active objects.

    Keys:
      Space / →   – next step
      ←           – previous step (replays from start)
      Q           – skip to next sequence
    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f'SLAM — {scan_id}', width=1280, height=800)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.08, 0.08, 0.12])
    opt.point_size = 4.0

    state = {'next': False, 'prev': False, 'quit': False}

    def _next(_): state['next'] = True; return False
    def _prev(_): state['prev'] = True; return False
    def _quit(v): state['quit'] = True; v.close(); return False

    vis.register_key_callback(32,        _next)   # Space
    vis.register_key_callback(262,       _next)   # →
    vis.register_key_callback(263,       _prev)   # ←
    vis.register_key_callback(ord('Q'),  _quit)

    grid = _ground_grid()
    vis.add_geometry(grid, reset_bounding_box=True)
    vis.poll_events(); vis.update_renderer()

    persistent: dict = {}   # gi → list[o3d_geom]
    gt_active:  list = []

    op_name = {0: 'OLD', 1: 'ADD', 2: 'UPD', 3: 'REM'}

    def _clear_gt():
        for g in gt_active:
            vis.remove_geometry(g, reset_bounding_box=False)
        gt_active.clear()

    def _apply_step(step_data, silent=False):
        _clear_gt()
        step_id = step_data['step_id']
        nodes   = step_data['nodes']
        if not silent:
            print(f'\n  [{scan_id}]  step {step_id:02d}  [{len(nodes)} nodes]')
        for node in nodes:
            # node_id is the robot-side id, stable across steps for the same
            # real object; `gi` is only this step's position in the flattened
            # batch and differs every time the same object reappears, so it
            # must NOT be used as the persistent-geometry key (using it made
            # every UPDATE add a new copy on top instead of replacing one).
            node_id = node['node_id']
            op      = node['op']
            if not silent:
                print(f'    [{op_name.get(op,"?")}] {node["label"]:<20s}  '
                      f'pos=({node["cx"]:+.2f},{node["cy"]:+.2f},{node["cz"]:+.2f})  '
                      f'yaw={math.degrees(node["yaw"]):+.1f}°')
            if op == 0:
                pass
            elif op == 3:  # REMOVE: delete, nothing to replace it with
                for g in persistent.pop(node_id, []):
                    vis.remove_geometry(g, reset_bounding_box=False)
            else:  # ADD or UPDATE: (re)place mesh at the predicted pose/model
                for g in persistent.pop(node_id, []):
                    vis.remove_geometry(g, reset_bounding_box=False)
                geoms = _build_single_geoms(node, obj_dir)
                for g in geoms:
                    vis.add_geometry(g, reset_bounding_box=False)
                persistent[node_id] = geoms
        # GT wireframes disabled

    def _rebuild_to(target_idx):
        """Replay steps 0..target_idx-1 to reconstruct scene state."""
        for geoms in persistent.values():
            for g in geoms:
                vis.remove_geometry(g, reset_bounding_box=False)
        persistent.clear()
        _clear_gt()
        for i in range(target_idx):
            _apply_step(vis_steps[i], silent=(i < target_idx - 1))

    # show first step immediately
    cur = 0
    if vis_steps:
        _apply_step(vis_steps[cur])
        cur = 1
        vis.poll_events(); vis.update_renderer()

    n = len(vis_steps)
    print(f'  [{scan_id}]  {n} steps  (Space/→: next   ←: prev   Q: next sequence)')

    while not state['quit']:
        if state['next']:
            state['next'] = False
            if cur < n:
                _apply_step(vis_steps[cur])
                cur += 1
                print(f'  step {cur}/{n}')
                vis.poll_events(); vis.update_renderer()
            else:
                print(f'  [{scan_id}] done — press Q for next sequence')
        elif state['prev']:
            state['prev'] = False
            if cur > 1:
                cur -= 1
                _rebuild_to(cur)
                print(f'  step {cur}/{n}')
                vis.poll_events(); vis.update_renderer()
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.02)

    vis.destroy_window()


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation function
# ─────────────────────────────────────────────────────────────────────────────
def evaluate():
    torch.manual_seed(48); random.seed(48); np.random.seed(48)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Evaluation device: {device}')

    if args.visualize and not _VIS_OK:
        print('[WARN] open3d / trimesh import failed — disabling --visualize')
        args.visualize = False

    # load training args
    args_json = os.path.join(args.exp, 'args.json')
    train_args = {}
    if os.path.exists(args_json):
        with open(args_json) as f:
            train_args = json.load(f)
        print(f'Loaded training args from {args_json}')

    enc_stats = args.enc_stats or train_args.get('enc_stats', '') or None
    gt_stats  = args.gt_stats  or train_args.get('gt_stats',  '') or None
    enc_stats = _require_stats(enc_stats, '--enc_stats')
    gt_stats  = _require_stats(gt_stats,  '--gt_stats')

    # GT denorm tensors
    gt_size_mu, gt_size_sg = _load_stats(gt_stats, slice(0, 3))
    gt_pos_mu,  gt_pos_sg  = _load_stats(gt_stats, slice(3, 6))

    # Enc denorm tensors (all 7 dims for vis; separate size for metric code)
    enc_box_mu,  enc_box_sg  = _load_stats(enc_stats, slice(0, 7))
    enc_size_mu, enc_size_sg = _load_stats(enc_stats, slice(0, 3))

    dataset = TeleDatasetSceneGraphIncremental(
        robot_sg_root        = args.robot_root,
        gt_scene_root        = args.gt_root,
        split                = args.split,
        shuffle_objs         = False,
        eval                 = True,
        seed                 = True,
        enc_normalized_stats = enc_stats,
        gt_normalized_stats  = gt_stats,
        splits_dir           = args.splits_dir or None,
    )
    loader = DataLoader(
        dataset,
        batch_sampler = LengthGroupedCurriculumSampler(dataset, batch_size=1),
        collate_fn    = dataset.collate_fn_inc,
        num_workers   = 0,
    )

    vocab = {
        'object_idx_to_name': {i: c for i, c in
                                enumerate(TeleDatasetSceneGraphIncremental.TELE_CLASSES)},
        'pred_idx_to_name':   {i: r for i, r in
                                enumerate(TeleDatasetSceneGraphIncremental.TELE_RELATIONS)},
    }
    ckpt_path = os.path.join(args.exp, 'checkpoint', f'model_{args.epoch}.pth')
    assert os.path.exists(ckpt_path), f'Checkpoint not found: {ckpt_path}'
    ckpt = torch.load(ckpt_path, map_location='cpu')

    rot_out_dim = _checkpoint_rot_out_dim(ckpt['model'])
    if rot_out_dim != N_YAW_BINS:
        raise RuntimeError(
            f'Checkpoint rotation head has output dim {rot_out_dim}; '
            f'this evaluator supports only the current {N_YAW_BINS}-bin '
            'yaw-residual head. Retrain or use the matching old evaluator.')

    model = Sg2ScAEModelTeleIncremental(
        vocab              = vocab,
        node_embedding_dim = train_args.get('node_dim', 128),
        gconv_num_layers   = train_args.get('num_layers', 5),
        shape_feat_dim     = train_args.get('shape_feat_dim', 768),
        residual           = train_args.get('residual', True),
        gconv_pooling      = train_args.get('pooling', 'avg'),
    )
    model.load_state_dict(ckpt['model'])
    print(f'Loaded checkpoint: {ckpt_path}  (epoch {ckpt["epoch"]})')

    model = model.to(device)
    model.eval()

    # shape library for size retrieval
    shape_lib = None
    shape_pkl = args.shape_features_pkl or train_args.get('shape_features_pkl', '')
    if shape_pkl and os.path.exists(shape_pkl):
        from helpers.tele_shape_features import TeleShapeLibrary
        shape_lib = TeleShapeLibrary(shape_pkl)
        print(f'Shape library loaded: {len(shape_lib._all_paths)} models')

    classes = TeleDatasetSceneGraphIncremental.TELE_CLASSES

    # URDF labels (need special yaw formula + explicit scaling in visualisation)
    _URDF_LABELS = {'chest_of_drawers', 'kitchen_counter', 'kitchen_cupboards',
                    'kitchen_cupboard', 'fridge'}

    extra_results = run_online_checks(dataset, model, device)

    # metric accumulators (global + per-class)
    box_keys = ['position', 'Overlap_GT(%)']
    box_sum       = {k: 0.0 for k in box_keys}
    n_steps       = 0
    n_pairs       = 0
    n_rot_pairs   = 0   # pairs counted in yaw error (excludes fold=0 classes)
    total_pos_err     = 0.0
    total_enc_pos_err = 0.0   # robot-side (encoder) position as baseline
    total_yaw_err = 0.0
    # full vs partial observation split (enc footprint / GT footprint >= 0.8 → full)
    _OBS_THRESH = 0.9
    acc_obs = {k: {'pos': 0.0, 'yaw': 0.0, 'n': 0, 'n_rot': 0}
               for k in ('full', 'partial')}
    shape_sum_obs = {k: {'CD': 0.0, 'F-score': 0.0, 'Vox_IoU': 0.0, 'n': 0}
                     for k in ('full', 'partial')}
    from collections import defaultdict
    shape_cls_obs = {'full': defaultdict(int), 'partial': defaultdict(int)}
    # per-class: {label: {'pos': float, 'yaw': float, 'n': int, 'n_rot': int}}
    per_class = defaultdict(lambda: {'pos': 0.0, 'yaw': 0.0, 'n': 0, 'n_rot': 0})
    all_infer_t      = 0.0
    n_total_steps    = 0   # 所有 sequence 的总步数（含无 active pair 的步）
    single_step_t     = None   # 首个只含 1 个 active pair 的 sequence 推理时间
    single_step_nstep = None   # 该 sequence 的总步数
    # shape evaluation accumulators (object-weighted; only active when shape_lib is provided)
    shape_sum  = {'CD': 0.0, 'F-score': 0.0, 'Vox_IoU': 0.0}
    n_shape_pairs = 0

    # pipeline benchmark accumulators (active when --bench_pipeline)
    _bp_stages   = ['to_cuda', 'gcn', 'postproc', 'sz_ret', 'sh_ret', 'mesh_build']
    _bp_totals   = {s: 0.0 for s in _bp_stages}
    _bp_nsteps   = 0
    _bp_nseqs    = 0
    _BP_WARMUP   = 2   # skip first N sequences from bench stats (GPU + disk warmup)

    for batch in loader:
        scan_id      = batch['step_meta']['scan_id_str'][0]
        scene_center = batch['step_meta']['scene_centers'][0].cpu().numpy()  # (3,)
        if args.seq and args.seq not in scan_id:
            continue
        print(f'\n── {scan_id} ──')
        # ── bench_pipeline: time to_cuda separately ───────────────────────────
        if args.bench_pipeline:
            if device.type == 'cuda': torch.cuda.synchronize()
            _bp_t0 = time.perf_counter()
        batch_gpu = to_cuda(batch, device)
        if args.bench_pipeline:
            if device.type == 'cuda': torch.cuda.synchronize()
            _bp_totals['to_cuda'] += time.perf_counter() - _bp_t0
        enc       = batch_gpu['encoder']

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            preds = model(batch_gpu, reset_states=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        seq_t = time.time() - t0
        if args.bench_pipeline:
            _bp_totals['gcn'] += seq_t
        all_infer_t   += seq_t
        seq_steps      = len(batch['active_matches'])
        n_total_steps += seq_steps
        n_active_in_seq = sum(1 for m in batch['active_matches'] if m)
        if n_active_in_seq == 1 and single_step_t is None:
            single_step_t     = seq_t
            single_step_nstep = seq_steps

        pred_pos = preds['pred_pos']          # (N, 3) normalised
        pred_rot = preds['pred_rot']          # (N, 1) radians [yaw]
        gt_boxes = batch_gpu['gt']['boxes']   # (G, 7) – positions/sizes normalised, yaw normalised
        gt_rot   = batch_gpu['gt'].get('rot') # (G, 3) [yaw,pitch,roll] raw radians; None if absent

        def _active_pred_rw(gi_list):
            """Render data for active (ADD/UPDATE) nodes that have NO Hungarian GT
            match this step. The model still predicted pred_pos/pred_rot/pred_feat
            for them — active_mask (what the model used) is independent of
            active_matches (what the scoring/GT-pairing step used) — so these are
            genuine predictions, not a fallback to the raw encoder box.
            """
            if not gi_list:
                return {}
            idx = torch.tensor(gi_list, dtype=torch.long, device=pred_pos.device)
            pos_m = _denorm(pred_pos[idx].cpu(), gt_pos_mu, gt_pos_sg)
            yaw   = pred_rot[idx, 0].cpu()
            labels = [classes[enc['objs'][gi].item()] for gi in gi_list]
            size_m = _denorm(enc['boxes'][idx, :3].cpu(), enc_size_mu, enc_size_sg)
            glb_paths = [None] * len(gi_list)
            if shape_lib is not None:
                feat   = preds['pred_feat'][idx].cpu()
                size_m = shape_lib.retrieve_sizes_batch(feat, tele_labels=labels, fallback=size_m)
                glb_paths = shape_lib.retrieve_batch(feat, tele_labels=labels)
            return {
                gi: {
                    'L': float(size_m[k, 0]), 'H': float(size_m[k, 1]), 'W': float(size_m[k, 2]),
                    'cx': float(pos_m[k, 0]) + scene_center[0],
                    'cy': float(pos_m[k, 1]) + scene_center[1],
                    'cz': float(pos_m[k, 2]) + scene_center[2],
                    'yaw': _apply_zero_yaw_prior(float(yaw[k]), labels[k]),
                    'glb_path': glb_paths[k] or None,
                }
                for k, gi in enumerate(gi_list)
            }

        vis_steps = []   # per-step data for this sequence
        _bp_seq_steps = len(batch['active_matches'])

        for flat_sid, matches in enumerate(batch['active_matches']):
            if not matches:
                # no GT-matched nodes this step — any active (ADD/UPDATE) nodes here
                # failed Hungarian matching, not inference; they still get the
                # model's real prediction via _active_pred_rw, not the raw box.
                if args.visualize or args.slam:
                    obj_mask = (enc['obj_to_flat_step'] == flat_sid)
                    step_gi  = obj_mask.nonzero(as_tuple=True)[0].cpu().tolist()
                    active_gi = [gi for gi in step_gi
                                 if int(enc['op_types'][gi].item()) in (1, 2)]
                    pred_rw = _active_pred_rw(active_gi)
                    nodes_vis = []
                    for gi in step_gi:
                        label = classes[enc['objs'][gi].item()]
                        op    = int(enc['op_types'][gi].item())
                        if gi in pred_rw:
                            nodes_vis.append({'gi': gi, 'node_id': enc['node_id'][gi],
                                               'label': label, 'op': op, **pred_rw[gi]})
                            continue
                        brw   = _denorm(enc['boxes'][gi].cpu().unsqueeze(0),
                                        enc_box_mu, enc_box_sg)[0]
                        nodes_vis.append({
                            'gi': gi, 'node_id': enc['node_id'][gi], 'label': label, 'op': op,
                            'L': float(brw[0]), 'H': float(brw[1]), 'W': float(brw[2]),
                            'cx': float(brw[3]) + scene_center[0],
                            'cy': float(brw[4]) + scene_center[1],
                            'cz': float(brw[5]) + scene_center[2],
                            'yaw': float(brw[6]),
                        })
                    vis_steps.append({'step_id': flat_sid, 'nodes': nodes_vis, 'gt': []})
                continue

            if args.bench_pipeline: _bp_t0 = time.perf_counter()
            enc_idx = torch.tensor([m[0] for m in matches],
                                   dtype=torch.long, device=pred_pos.device)
            gt_idx  = torch.tensor([m[1] for m in matches],
                                   dtype=torch.long, device=pred_pos.device)

            # position in GT-normalised space → real world
            p_pos_n = pred_pos[enc_idx].cpu()
            g_pos_n = gt_boxes[gt_idx, 3:6].cpu()
            p_pos_m = _denorm(p_pos_n, gt_pos_mu, gt_pos_sg)
            g_pos_m = _denorm(g_pos_n, gt_pos_mu, gt_pos_sg)

            obj_labels = [classes[enc['objs'][enc_idx[i]].item()] for i in range(len(enc_idx))]

            p_yaw = pred_rot[enc_idx, 0].clone()
            zero_yaw_mask = torch.tensor([lbl in _ZERO_YAW_LABELS for lbl in obj_labels],
                                         dtype=torch.bool, device=p_yaw.device)
            p_yaw[zero_yaw_mask] = 0.0
            # Use raw-radian GT yaw (gt_rot) when available; fall back to
            # denormalising gt_boxes[:, 6] with the yaw stats.
            if gt_rot is not None:
                g_yaw = gt_rot[gt_idx, 0]
            else:
                g_yaw = gt_boxes[gt_idx, 6]   # normalised – imprecise fallback

            # symmetry-aware yaw error
            folds = torch.tensor([_SYM_FOLD.get(lbl, 1) for lbl in obj_labels],
                                 dtype=torch.long)
            pair_yaw_rad = sym_yaw_err_rad(p_yaw.cpu(), g_yaw.cpu(), folds)  # (M,) rad
            pair_yaw     = pair_yaw_rad * RAD2DEG                             # (M,) deg

            pair_pos = (p_pos_m - g_pos_m).norm(dim=1).cpu()   # (M,)
            pos_err = pair_pos.mean().item()

            # encoder (robot-side) position error — baseline comparison
            enc_pos_n = enc['boxes'][enc_idx, 3:6].cpu()
            enc_pos_m = _denorm(enc_pos_n, enc_box_mu[3:6], enc_box_sg[3:6])
            pair_enc_pos = (enc_pos_m - g_pos_m).norm(dim=1)
            total_enc_pos_err += float(pair_enc_pos.sum())
            rot_mask = (folds != 0)                              # exclude fold=0 from yaw stats
            yaw_err = pair_yaw[rot_mask].mean().item() if rot_mask.any() else float('nan')
            n_this_pairs = len(matches)
            total_pos_err += float(pair_pos.sum())
            total_yaw_err += float(pair_yaw[rot_mask].sum())
            n_pairs       += n_this_pairs
            n_rot_pairs   += int(rot_mask.sum())

            # per-pair per-class accumulation
            for i in range(len(matches)):
                lbl = classes[enc['objs'][enc_idx[i]].item()]
                per_class[lbl]['pos'] += float(pair_pos[i])
                per_class[lbl]['n']   += 1
                if _SYM_FOLD.get(lbl, 1) != 0:
                    per_class[lbl]['yaw']   += float(pair_yaw[i])
                    per_class[lbl]['n_rot'] += 1
            if args.bench_pipeline:
                _bp_totals['postproc'] += time.perf_counter() - _bp_t0

            # full real-world predicted box [L,H,W, cx,cy,cz, yaw_deg]
            robot_enc_cpu = enc['boxes'][enc_idx].cpu()
            pred_full_rw  = robot_enc_cpu.clone()
            enc_sizes_rw  = _denorm(robot_enc_cpu[:, :3], enc_size_mu, enc_size_sg)
            pred_glb_paths = [None] * len(enc_idx)
            if shape_lib is not None:
                act_labels = [classes[enc['objs'][enc_idx[i]].item()]
                              for i in range(len(enc_idx))]
                if args.bench_pipeline: _bp_t0 = time.perf_counter()
                pred_feat_vis = preds['pred_feat'][enc_idx].cpu()
                pred_full_rw[:, :3] = shape_lib.retrieve_sizes_batch(
                    pred_feat_vis,
                    tele_labels=act_labels,
                    fallback=enc_sizes_rw)
                # which specific model (within the category) the predicted shape
                # feature is nearest to — used so the displayed mesh actually
                # tracks the prediction instead of always showing the same
                # first-on-disk model for that category.
                pred_glb_paths = shape_lib.retrieve_batch(pred_feat_vis, tele_labels=act_labels)
                if args.bench_pipeline:
                    _bp_totals['sz_ret'] += time.perf_counter() - _bp_t0
            else:
                pred_full_rw[:, :3] = enc_sizes_rw
            pred_full_rw[:, 3:6] = p_pos_m
            pred_full_rw[:, 6]   = p_yaw.cpu() * RAD2DEG   # store as degrees

            # full real-world GT box
            gt_full_rw = gt_boxes[gt_idx].cpu().clone()
            gt_full_rw[:, :3]  = _denorm(gt_boxes[gt_idx, :3].cpu(), gt_size_mu, gt_size_sg)
            gt_full_rw[:, 3:6] = g_pos_m
            if gt_rot is not None:
                gt_full_rw[:, 6] = gt_rot[gt_idx, 0].cpu() * RAD2DEG
            else:
                gt_full_rw[:, 6] = gt_boxes[gt_idx, 6].cpu() * RAD2DEG  # normalised fallback

            m_acc = validate_box_accuracy(pred_full_rw, gt_full_rw)
            for k in box_keys:
                box_sum[k] += m_acc.get(k, 0.0) * n_this_pairs
            n_steps += 1

            # ── full vs partial observation split ──────────────────────────────
            enc_fp  = enc_sizes_rw[:, 0] * enc_sizes_rw[:, 2]          # enc L*W footprint
            gt_fp   = gt_full_rw[:, 0]   * gt_full_rw[:, 2]            # GT  L*W footprint
            fp_ratio = (enc_fp / gt_fp.clamp(min=1e-6)).cpu()          # (M,)
            is_full  = fp_ratio >= _OBS_THRESH                          # bool (M,)
            for tag, sel in (('full', is_full), ('partial', ~is_full)):
                if not sel.any():
                    continue
                acc_obs[tag]['pos']   += float(pair_pos[sel].sum())
                acc_obs[tag]['n']     += int(sel.sum())
                sel_rot = sel & rot_mask
                if sel_rot.any():
                    acc_obs[tag]['yaw']   += float(pair_yaw[sel_rot].sum())
                    acc_obs[tag]['n_rot'] += int(sel_rot.sum())

            # ── shape evaluation (requires shape_lib) ──────────────────────
            shape_str = ''
            try:
                import trimesh as _trimesh
            except ImportError:
                _trimesh = None
            if shape_lib is not None and _trimesh is not None:
                act_labels  = [classes[enc['objs'][enc_idx[i]].item()]
                               for i in range(len(enc_idx))]
                gt_models   = batch.get('gt', {}).get('models', [])

                # separate mesh lists for full / partial observation
                meshes = {'full': ([], [], 0), 'partial': ([], [], 0)}
                mp_list = {'full': [], 'partial': []}
                mg_list = {'full': [], 'partial': []}
                n_urdf  = {'full': 0,  'partial': 0}

                for i_obj, label in enumerate(act_labels):
                    lib_label = shape_lib.map_label(label)
                    if lib_label is None:
                        continue   # background / unknown class

                    tag = 'full' if is_full[i_obj].item() else 'partial'
                    shape_cls_obs[tag][label] += 1

                    # Pred: retrieve nearest model by predicted feature (within class)
                    pred_feat_i  = preds['pred_feat'][enc_idx[i_obj]].cpu()
                    if args.bench_pipeline: _bp_t0 = time.perf_counter()
                    pred_results = shape_lib.retrieve(pred_feat_i,
                                                      label_filter=lib_label)
                    if args.bench_pipeline:
                        _bp_totals['sh_ret'] += time.perf_counter() - _bp_t0
                    pp = pred_results[0][0] if pred_results else None

                    # GT: find the actual model file from the scene graph's model name
                    gi        = gt_idx[i_obj].item()
                    gt_model  = gt_models[gi] if gi < len(gt_models) else ''
                    gp = None
                    if gt_model:
                        # search label sub-folder first, then any sub-folder
                        candidates = glob.glob(
                            os.path.join(args.obj_dir, label, f'{gt_model}.glb'))
                        if not candidates:
                            candidates = glob.glob(
                                os.path.join(args.obj_dir, '**', f'{gt_model}.glb'),
                                recursive=True)
                        gp = candidates[0] if candidates else None

                    # URDF: trimesh cannot load → count as CD=0, F=1, IoU=1
                    if (pp and pp.endswith('.urdf')) or (gp is None and gt_model and
                            glob.glob(os.path.join(args.obj_dir, '**',
                                                   f'{gt_model}.urdf'), recursive=True)):
                        n_urdf[tag] += 1
                        continue

                    if pp and gp and os.path.exists(pp) and os.path.exists(gp):
                        try:
                            mp = _trimesh.load(pp, force='mesh')
                            mg = _trimesh.load(gp, force='mesh')
                            if isinstance(mp, _trimesh.Trimesh) and isinstance(mg, _trimesh.Trimesh):
                                mp_list[tag].append(mp)
                                mg_list[tag].append(mg)
                        except Exception:
                            pass

                # Accumulate shape metrics per tag (full / partial) + combined
                cd_sum_all = f_sum_all = iou_sum_all = 0.0
                n_total_this = 0
                for tag in ('full', 'partial'):
                    n_mesh = len(mp_list[tag])
                    n_u    = n_urdf[tag]
                    n_tot  = n_mesh + n_u
                    if n_tot == 0:
                        continue
                    if n_mesh > 0:
                        s = validate_shape_accuracy(mp_list[tag], mg_list[tag])
                        cd  = s['CD']      * n_mesh
                        fs  = s['F-score'] * n_mesh + n_u
                        iou = s['Vox_IoU'] * n_mesh + n_u
                        # obs split: URDF excluded — only non-URDF meshes
                        shape_sum_obs[tag]['CD']      += s['CD']      * n_mesh
                        shape_sum_obs[tag]['F-score'] += s['F-score'] * n_mesh
                        shape_sum_obs[tag]['Vox_IoU'] += s['Vox_IoU'] * n_mesh
                        shape_sum_obs[tag]['n']       += n_mesh
                    else:
                        cd, fs, iou = 0.0, float(n_u), float(n_u)
                        # all URDF: nothing to add to obs split
                    cd_sum_all  += cd;  f_sum_all  += fs
                    iou_sum_all += iou; n_total_this += n_tot

                if n_total_this > 0:
                    cd_step  = cd_sum_all  / n_total_this
                    f_step   = f_sum_all   / n_total_this
                    iou_step = iou_sum_all / n_total_this
                    shape_sum['CD']      += cd_sum_all
                    shape_sum['F-score'] += f_sum_all
                    shape_sum['Vox_IoU'] += iou_sum_all
                    n_shape_pairs += n_total_this
                    shape_str = (f'  CD={cd_step:.4f}'
                                 f'  F={f_step:.3f}'
                                 f'  VoxIoU={iou_step:.3f}')

            pos_unit = 'm' if gt_pos_mu is not None else 'norm'
            print(f'  step {flat_sid:02d}: M={len(matches)}  '
                  f'pos_L1={pos_err:.4f}{pos_unit}  yaw_err={yaw_err:.1f}°  '
                  f'overlap={m_acc.get("Overlap_GT(%)", 0.):.1f}%{shape_str}')

            # ── collect visualisation data ──────────────────────────────────
            if args.visualize or args.slam or args.bench_pipeline:
                obj_mask    = (enc['obj_to_flat_step'] == flat_sid)
                all_gi      = obj_mask.nonzero(as_tuple=True)[0].cpu().tolist()
                gi_to_pred  = {gi: j for j, gi in enumerate(enc_idx.cpu().tolist())}
                # active nodes this step that matched no GT (see _active_pred_rw) —
                # they still get the model's real prediction, not the raw encoder box
                unmatched_active_gi = [
                    gi for gi in all_gi
                    if gi not in gi_to_pred and int(enc['op_types'][gi].item()) in (1, 2)
                ]
                unmatched_pred_rw = _active_pred_rw(unmatched_active_gi)

                nodes_vis = []
                for gi in all_gi:
                    label = classes[enc['objs'][gi].item()]
                    op    = int(enc['op_types'][gi].item())

                    node_id = enc['node_id'][gi]
                    if gi in gi_to_pred:
                        j   = gi_to_pred[gi]
                        prw = pred_full_rw[j]   # yaw in degrees, position in world/map frame
                        nodes_vis.append({
                            'gi': gi, 'node_id': node_id, 'label': label, 'op': op,
                            'L': float(prw[0]), 'H': float(prw[1]), 'W': float(prw[2]),
                            'cx': float(prw[3]) + scene_center[0],
                            'cy': float(prw[4]) + scene_center[1],
                            'cz': float(prw[5]) + scene_center[2],
                            'yaw': float(prw[6]) / RAD2DEG,   # → radians
                            'glb_path': pred_glb_paths[j],
                        })
                    elif gi in unmatched_pred_rw:
                        nodes_vis.append({'gi': gi, 'node_id': node_id, 'label': label, 'op': op,
                                           **unmatched_pred_rw[gi]})
                    else:
                        # OLD or REMOVE: use enc box denormed to real world
                        brw = _denorm(enc['boxes'][gi].cpu().unsqueeze(0),
                                      enc_box_mu, enc_box_sg)[0]
                        nodes_vis.append({
                            'gi': gi, 'node_id': node_id, 'label': label, 'op': op,
                            'L': float(brw[0]), 'H': float(brw[1]), 'W': float(brw[2]),
                            'cx': float(brw[3]) + scene_center[0],
                            'cy': float(brw[4]) + scene_center[1],
                            'cz': float(brw[5]) + scene_center[2],
                            'yaw': float(brw[6]),   # radians after denorm
                        })

                gt_vis = [
                    {
                        'L': float(gt_full_rw[j, 0]), 'H': float(gt_full_rw[j, 1]),
                        'W': float(gt_full_rw[j, 2]),
                        'cx': float(gt_full_rw[j, 3]) + scene_center[0],
                        'cy': float(gt_full_rw[j, 4]) + scene_center[1],
                        'cz': float(gt_full_rw[j, 5]) + scene_center[2],
                    }
                    for j in range(len(matches))
                ]

                vis_steps.append({'step_id': flat_sid, 'nodes': nodes_vis, 'gt': gt_vis})

                # bench_pipeline: time mesh load + place (no window)
                if args.bench_pipeline and (args.visualize is False and args.slam is False):
                    _bp_t0 = time.perf_counter()
                    _build_step_geoms(nodes_vis, gt_vis, args.obj_dir)
                    _bp_totals['mesh_build'] += time.perf_counter() - _bp_t0

        if args.bench_pipeline:
            _bp_nseqs += 1
            if _bp_nseqs == _BP_WARMUP + 1:
                # first measured sequence: reset totals (clear warmup contamination)
                for s in _bp_stages:
                    _bp_totals[s] = 0.0
            if _bp_nseqs > _BP_WARMUP:
                _bp_nsteps += _bp_seq_steps

        # launch interactive viewer for this sequence
        if (args.visualize or args.slam) and vis_steps:
            if args.slam:
                _show_slam(scan_id, vis_steps, args.obj_dir, hz=20.0)
            else:
                _show_sequence(scan_id, vis_steps, args.obj_dir)

    # ── pipeline benchmark summary ────────────────────────────────────────────
    if args.bench_pipeline and _bp_nsteps > 0:
        print('\n' + '='*60)
        print(f'Pipeline latency breakdown  ({_bp_nseqs} sequences, {_bp_nsteps} steps total)')
        print(f'  {"Stage":<15} {"ms/step":>10}  {"Hz":>8}')
        print(f'  {"-"*35}')
        for s in _bp_stages:
            ms = _bp_totals[s] / _bp_nsteps * 1000
            hz = 1000.0 / ms if ms > 0 else float('inf')
            print(f'  {s:<15} {ms:>10.2f}  {hz:>8.0f}')
        total_ms = sum(_bp_totals[s] for s in _bp_stages) / _bp_nsteps * 1000
        total_hz = 1000.0 / total_ms if total_ms > 0 else float('inf')
        print(f'  {"─"*35}')
        print(f'  {"TOTAL":<15} {total_ms:>10.2f}  {total_hz:>8.0f}  Hz')
        print('='*60)

    # ── summary ──────────────────────────────────────────────────────────────
    if n_pairs == 0:
        print('No active pairs evaluated.')
        return

    print('\n' + '='*60)
    pos_unit = 'm' if gt_pos_mu is not None else 'norm'
    print(f'Split: {args.split}  |  Sequences: {len(dataset)}  |  Non-empty steps: {n_steps}')
    print(f'Avg position L1   : {total_pos_err / n_pairs:.4f} {pos_unit}'
          f'  (enc baseline: {total_enc_pos_err / n_pairs:.4f} {pos_unit})')
    print(f'Avg yaw error     : {total_yaw_err / max(n_rot_pairs, 1):.2f}°'
          f'  (excl. beanbag/indoor_plant; {n_rot_pairs}/{n_pairs} pairs)')
    for k in box_keys:
        print(f'  {k:20s}: {box_sum[k] / n_pairs:.4f}')
    print(f'Total active pairs: {n_pairs}')

    print(f'\n--- Full vs Partial observation (enc/GT footprint threshold={_OBS_THRESH}) ---')
    for tag in ('full', 'partial'):
        a = acc_obs[tag]
        n, nr = a['n'], a['n_rot']
        pos_s = f"{a['pos']/n:.4f} {pos_unit}" if n  > 0 else '  —  '
        yaw_s = f"{a['yaw']/nr:.2f}°"          if nr > 0 else '  —  '
        print(f'  {tag:8s}  n={n:4d}  pos={pos_s}  yaw={yaw_s}'
              f'  (n_rot={nr})')
    if any(shape_sum_obs[t]['n'] > 0 for t in ('full', 'partial')):
        print(f'\n--- Shape: Full vs Partial observation (mesh-only, excl. URDF) ---')
        mesh_total = {'CD': 0.0, 'F-score': 0.0, 'Vox_IoU': 0.0, 'n': 0}
        for tag in ('full', 'partial'):
            s = shape_sum_obs[tag]
            ns = s['n']
            if ns == 0:
                print(f'  {tag:8s}  n=   0  —')
            else:
                print(f'  {tag:8s}  n={ns:4d}'
                      f'  CD={s["CD"]/ns:.4f}'
                      f'  F={s["F-score"]/ns:.4f}'
                      f'  VoxIoU={s["Vox_IoU"]/ns:.4f}')
            for k in ('CD', 'F-score', 'Vox_IoU'):
                mesh_total[k] += s[k]
            mesh_total['n'] += ns
        nm = mesh_total['n']
        if nm > 0:
            print(f'  {"overall":8s}  n={nm:4d}'
                  f'  CD={mesh_total["CD"]/nm:.4f}'
                  f'  F={mesh_total["F-score"]/nm:.4f}'
                  f'  VoxIoU={mesh_total["Vox_IoU"]/nm:.4f}  (mesh-only)')
    t_per_seq  = all_infer_t / len(dataset)
    t_per_step = all_infer_t / max(n_total_steps, 1)
    t_per_pair = all_infer_t / max(n_pairs, 1)
    print(f'Avg inference time: {t_per_seq:.4f} s/sequence')
    print(f'  per step  : {t_per_step*1000:.2f} ms  →  {1/t_per_step:.0f} Hz  (total {n_total_steps} steps)')
    print(f'  per active: {t_per_pair*1000:.2f} ms  →  {1/t_per_pair:.0f} Hz  (total {n_pairs} active pairs)')
    if single_step_t is not None:
        t_1step = single_step_t / single_step_nstep
        print(f'  single-change-event seq: {single_step_t*1000:.1f} ms total, {single_step_nstep} steps'
              f'  →  {t_1step*1000:.2f} ms/step  ({1/t_1step:.0f} Hz throughput)')
        print(f'  NOTE: model processes full sequence in one call; above is batch throughput, not online latency.')
    if n_shape_pairs > 0:
        print(f'\n--- Shape evaluation ({n_shape_pairs} objects) ---')
        for k in shape_sum:
            print(f'  {k:12s}: {shape_sum[k] / n_shape_pairs:.4f}')
        if any(shape_cls_obs[t] for t in ('full', 'partial')):
            print('  Class composition (non-URDF, excl. background):')
            all_cls = sorted(set(shape_cls_obs['full']) | set(shape_cls_obs['partial']))
            for cls in all_cls:
                nf = shape_cls_obs['full'].get(cls, 0)
                np_ = shape_cls_obs['partial'].get(cls, 0)
                print(f'    {cls:<22s}  full={nf:3d}  partial={np_:3d}')

    # ── per-class breakdown ───────────────────────────────────────────────
    print('\n--- Per-class errors (pos m / yaw °) ---')
    rows = sorted(per_class.items(),
                  key=lambda kv: kv[1]['yaw'] / max(kv[1]['n_rot'], 1),
                  reverse=True)
    urdf_pos = urdf_yaw = urdf_n = urdf_nr = 0.0
    nonu_pos = nonu_yaw = nonu_n = nonu_nr = 0.0
    for lbl, acc in rows:
        n = acc['n']
        if n == 0:
            continue
        p  = acc['pos'] / n
        nr = acc['n_rot']
        y  = acc['yaw'] / nr if nr > 0 else float('nan')
        tag = '[URDF]' if lbl in _URDF_LABELS else '      '
        yaw_str = f'{y:.1f}°' if nr > 0 else '  —  (excluded)'
        print(f'  {tag} {lbl:25s}  n={n:4d}  pos={p:.3f}m  yaw={yaw_str}')
        if lbl in _URDF_LABELS:
            urdf_pos += acc['pos']; urdf_yaw += acc['yaw']
            urdf_n  += n;           urdf_nr  += nr
        else:
            nonu_pos += acc['pos']; nonu_yaw += acc['yaw']
            nonu_n  += n;           nonu_nr  += nr
    print(f'\n  [URDF]  avg  pos={urdf_pos/max(urdf_n,1):.3f}m'
          f'  yaw={urdf_yaw/max(urdf_nr,1):.1f}°  (n={int(urdf_n)}, n_rot={int(urdf_nr)})')
    print(f'  [nonU]  avg  pos={nonu_pos/max(nonu_n,1):.3f}m'
          f'  yaw={nonu_yaw/max(nonu_nr,1):.1f}°  (n={int(nonu_n)}, n_rot={int(nonu_nr)})')

    result = {
        'split': args.split, 'epoch': args.epoch,
        'n_sequences': len(dataset), 'n_steps': n_steps,
        'n_pairs': n_pairs, 'n_rot_pairs': n_rot_pairs,
        f'pos_L1_{pos_unit}': total_pos_err / n_pairs,
        f'enc_pos_L1_{pos_unit}': total_enc_pos_err / n_pairs,
        'yaw_deg':  total_yaw_err / max(n_rot_pairs, 1),
        **{k: box_sum[k] / n_pairs for k in box_keys},
        'avg_infer_s': all_infer_t / len(dataset),
        **({'n_shape_pairs': n_shape_pairs,
            **{k: shape_sum[k] / n_shape_pairs for k in shape_sum}}
           if n_shape_pairs > 0 else {}),
        **extra_results,
    }
    out_path = os.path.join(args.exp, f'eval_{args.split}_ep{args.epoch}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Results saved to {out_path}')


if __name__ == '__main__':
    evaluate()
