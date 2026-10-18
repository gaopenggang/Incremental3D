"""
Evaluation script for Sg2ScAEModelTeleIncremental.

Metrics (computed over active-node matched pairs):
  - Position L1  (m, if box_stats supplied; normalised otherwise)
  - Yaw error    (degrees)
  - Position %   (pos_err / box diagonal)
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
from model.AE_tele_incremental import Sg2ScAEModelTeleIncremental
from helpers.util import bool_flag, denormalize_box_params
from helpers.metrics_3dfront import validate_box_accuracy


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
parser.add_argument('--obj_dir',    default='/media/gao/gao_ssd/3D objects',
                    help='3D objects directory (used when --visualize True)')
parser.add_argument('--enc_stats', type=str, default='',
                    help='encoder normalization stats file')
parser.add_argument('--gt_stats',  type=str, default='',
                    help='GT normalization stats file (used to denormalise predictions)')
parser.add_argument('--shape_features_pkl', type=str,
                    default='/media/gao/gao_ssd/3D objects/features.pkl',
                    help='path to 3D library features.pkl for size retrieval')
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


def _load_stats(stats_file, dims):
    """Load mean/std tensors for selected dims from a 2-row stats file."""
    if not stats_file:
        return None, None
    s = np.loadtxt(stats_file)
    return (torch.tensor(s[0, dims], dtype=torch.float32),
            torch.tensor(s[1, dims], dtype=torch.float32))


def _denorm(x_cpu, mean, std):
    """Inverse normalisation: x_real = x_norm * std / 3 + mean."""
    if mean is None:
        return x_cpu
    return x_cpu * std / 3.0 + mean


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
}


def _canon(label: str) -> str:
    """Resolve a dataset label to its canonical form for _URDF_FILES lookup."""
    return _LABEL_ALIAS.get(label, label)

_mesh_cache: dict = {}
_urdf_cache: dict = {}


def _yaw_mat3(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


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
            vc = p.visual.to_color().vertex_colors[:, :3].astype(np.float64) / 255.
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


def _load_mesh_glb(obj_dir: str, label: str):
    key = f'glb:{label}'
    if key in _mesh_cache:
        mesh, vc = _mesh_cache[key]
        return (mesh.copy() if mesh is not None else None), vc

    cat_dir = os.path.join(obj_dir, label)
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
            # OLD / REMOVE → wireframe only (no mesh load to keep it fast)
            # Enc boxes are axis-aligned AABBs: [X_ext, Y_ext, Z_ext] → no L/W swap
            geoms.append(_obb_wireframe(center, L, H, W, yaw, color))
        else:
            # ADD / UPDATE → show mesh at predicted pose
            placed = False
            if _canon(label) in _URDF_FILES:
                parts = _load_urdf(obj_dir, label)
                if parts:
                    for m, vc in _place_parts(parts, center, L, H, W, yaw):
                        geoms.append(_trimesh_to_o3d(m, vc, color))
                    placed = True
                    # URDF parts are explicitly scaled to L×H×W, so the box matches.
                    geoms.append(_obb_wireframe(center, L, H, W, yaw, color))
            if not placed:
                mesh, vc = _load_mesh_glb(obj_dir, label)
                if mesh is not None:
                    placed_mesh = _place_mesh(mesh, center, yaw)
                    geoms.append(_trimesh_to_o3d(placed_mesh, vc, color))
                    placed = True
                    # Wireframe from the placed mesh's actual AABB (axis-aligned),
                    # so the box always matches what is rendered regardless of .glb orientation.
                    v = np.asarray(placed_mesh.vertices, np.float64)
                    mn, mx = v.min(0), v.max(0)
                    ext = mx - mn
                    mesh_center = (mn + mx) / 2
                    geoms.append(_obb_wireframe(mesh_center, ext[0], ext[1], ext[2], 0.0, color))
            if not placed:
                # No mesh available: fall back to predicted-box wireframe
                geoms.append(_obb_wireframe(center, L, H, W, yaw, color))

    # GT boxes – orange wireframe.
    # The stored L=wx, H=wy, W=wz are WORLD AABB extents already incorporating the
    # full object rotation (computed via world_aabb_extents in the GT generation script).
    # Draw as an axis-aligned box (yaw=0): L along world X, H along world Y, W along
    # world Z.  Applying any rotation would double-count the baked-in rotation.
    for gt in gt_vis:
        geoms.append(_obb_wireframe(
            [gt['cx'], gt['cy'], gt['cz']],
            gt['L'], gt['H'], gt['W'], 0.0, [1.0, 0.60, 0.0]))

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


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation function
# ─────────────────────────────────────────────────────────────────────────────
def evaluate():
    torch.manual_seed(48); random.seed(48); np.random.seed(48)

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

    # Detect rot output dim from checkpoint so we handle both yaw-only (1) and full (3) models.
    _rot_w = ckpt['model'].get('mlp_rot.1.weight', ckpt['model'].get('mlp_rot.2.weight'))
    rot_out_dim = int(_rot_w.shape[0]) if _rot_w is not None else 3

    model = Sg2ScAEModelTeleIncremental(
        vocab              = vocab,
        node_embedding_dim = train_args.get('node_dim', 128),
        gconv_num_layers   = train_args.get('num_layers', 5),
        shape_feat_dim     = train_args.get('shape_feat_dim', 256),
        residual           = train_args.get('residual', True),
        gconv_pooling      = train_args.get('pooling', 'avg'),
        rot_out_dim        = rot_out_dim,
    )
    model.load_state_dict(ckpt['model'])
    print(f'Loaded checkpoint: {ckpt_path}  (epoch {ckpt["epoch"]})')

    if torch.cuda.is_available():
        model = model.cuda()
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

    # metric accumulators (global + per-class)
    box_keys = ['position', 'orientation', 'Overlap_GT(%)',
                'orientation_p(%)']
    box_sum       = {k: 0.0 for k in box_keys}
    n_steps       = 0
    n_pairs       = 0
    total_pos_err = 0.0
    total_yaw_err = 0.0
    # per-class: {label: {'pos': float, 'yaw': float, 'n': int}}
    from collections import defaultdict
    per_class = defaultdict(lambda: {'pos': 0.0, 'yaw': 0.0, 'n': 0})
    all_infer_t   = 0.0

    for batch in loader:
        scan_id      = batch['step_meta']['scan_id_str'][0]
        scene_center = batch['step_meta']['scene_centers'][0].cpu().numpy()  # (3,)
        print(f'\n── {scan_id} ──')
        batch_gpu = to_cuda(batch)
        enc       = batch_gpu['encoder']

        t0 = time.time()
        with torch.no_grad():
            preds = model(batch_gpu, reset_states=True)
        all_infer_t += time.time() - t0

        pred_pos = preds['pred_pos']          # (N, 3) normalised
        pred_rot = preds['pred_rot']          # (N, 3) radians [yaw, pitch, roll]
        gt_boxes = batch_gpu['gt']['boxes']   # (G, 7) – positions/sizes normalised, yaw normalised
        gt_rot   = batch_gpu['gt'].get('rot') # (G, 3) [yaw,pitch,roll] raw radians; None if absent

        vis_steps = []   # per-step data for this sequence

        for flat_sid, matches in enumerate(batch['active_matches']):
            if not matches:
                # no active nodes – still record accumulated scene for vis
                if args.visualize:
                    obj_mask = (enc['obj_to_flat_step'] == flat_sid)
                    nodes_vis = []
                    for gi in obj_mask.nonzero(as_tuple=True)[0].cpu().tolist():
                        label = classes[enc['objs'][gi].item()]
                        op    = int(enc['op_types'][gi].item())
                        brw   = _denorm(enc['boxes'][gi].cpu().unsqueeze(0),
                                        enc_box_mu, enc_box_sg)[0]
                        nodes_vis.append({
                            'label': label, 'op': op,
                            'L': float(brw[0]), 'H': float(brw[1]), 'W': float(brw[2]),
                            'cx': float(brw[3]) + scene_center[0],
                            'cy': float(brw[4]) + scene_center[1],
                            'cz': float(brw[5]) + scene_center[2],
                            'yaw': float(brw[6]),
                        })
                    vis_steps.append({'step_id': flat_sid, 'nodes': nodes_vis, 'gt': []})
                continue

            enc_idx = torch.tensor([m[0] for m in matches],
                                   dtype=torch.long, device=pred_pos.device)
            gt_idx  = torch.tensor([m[1] for m in matches],
                                   dtype=torch.long, device=pred_pos.device)

            # position in GT-normalised space → real world
            p_pos_n = pred_pos[enc_idx].cpu()
            g_pos_n = gt_boxes[gt_idx, 3:6].cpu()
            p_pos_m = _denorm(p_pos_n, gt_pos_mu, gt_pos_sg)
            g_pos_m = _denorm(g_pos_n, gt_pos_mu, gt_pos_sg)

            p_yaw = pred_rot[enc_idx, 0]
            # Use raw-radian GT yaw (gt_rot) when available; fall back to
            # denormalising gt_boxes[:, 6] with the yaw stats.
            if gt_rot is not None:
                g_yaw = gt_rot[gt_idx, 0]
            else:
                g_yaw = gt_boxes[gt_idx, 6]   # normalised – imprecise fallback

            pos_err = (p_pos_m - g_pos_m).norm(dim=1).mean().item()
            yaw_err = circular_yaw_err_deg(p_yaw, g_yaw)
            total_pos_err += pos_err
            total_yaw_err += yaw_err
            n_pairs       += len(matches)

            # per-pair per-class accumulation
            pair_pos = (p_pos_m - g_pos_m).norm(dim=1).cpu()   # (M,)
            dy       = p_yaw.cpu() - g_yaw.cpu()
            pair_yaw = torch.atan2(torch.sin(dy), torch.cos(dy)).abs() * (180.0 / math.pi)
            for i in range(len(matches)):
                lbl = classes[enc['objs'][enc_idx[i]].item()]
                per_class[lbl]['pos'] += float(pair_pos[i])
                per_class[lbl]['yaw'] += float(pair_yaw[i])
                per_class[lbl]['n']   += 1

            # full real-world predicted box [L,H,W, cx,cy,cz, yaw_deg]
            robot_enc_cpu = enc['boxes'][enc_idx].cpu()
            pred_full_rw  = robot_enc_cpu.clone()
            enc_sizes_rw  = _denorm(robot_enc_cpu[:, :3], enc_size_mu, enc_size_sg)
            if shape_lib is not None:
                act_labels = [classes[enc['objs'][enc_idx[i]].item()]
                              for i in range(len(enc_idx))]
                pred_full_rw[:, :3] = shape_lib.retrieve_sizes_batch(
                    preds['pred_feat'][enc_idx].cpu(),
                    tele_labels=act_labels,
                    fallback=enc_sizes_rw)
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
                box_sum[k] += m_acc.get(k, 0.0)
            n_steps += 1

            pos_unit = 'm' if gt_pos_mu is not None else 'norm'
            print(f'  step {flat_sid:02d}: M={len(matches)}  '
                  f'pos_L1={pos_err:.4f}{pos_unit}  yaw_err={yaw_err:.1f}°  '
                  f'overlap={m_acc.get("Overlap_GT(%)", 0.):.1f}%')

            # ── collect visualisation data ──────────────────────────────────
            if args.visualize:
                obj_mask    = (enc['obj_to_flat_step'] == flat_sid)
                all_gi      = obj_mask.nonzero(as_tuple=True)[0].cpu().tolist()
                gi_to_pred  = {gi: j for j, gi in enumerate(enc_idx.cpu().tolist())}

                nodes_vis = []
                for gi in all_gi:
                    label = classes[enc['objs'][gi].item()]
                    op    = int(enc['op_types'][gi].item())

                    if gi in gi_to_pred:
                        j   = gi_to_pred[gi]
                        prw = pred_full_rw[j]   # yaw in degrees, position in world/map frame
                        nodes_vis.append({
                            'label': label, 'op': op,
                            'L': float(prw[0]), 'H': float(prw[1]), 'W': float(prw[2]),
                            'cx': float(prw[3]) + scene_center[0],
                            'cy': float(prw[4]) + scene_center[1],
                            'cz': float(prw[5]) + scene_center[2],
                            'yaw': float(prw[6]) / RAD2DEG,   # → radians
                        })
                    else:
                        # OLD or REMOVE: use enc box denormed to real world
                        brw = _denorm(enc['boxes'][gi].cpu().unsqueeze(0),
                                      enc_box_mu, enc_box_sg)[0]
                        nodes_vis.append({
                            'label': label, 'op': op,
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

        # launch interactive viewer for this sequence
        if args.visualize and vis_steps:
            _show_sequence(scan_id, vis_steps, args.obj_dir)

    # ── summary ──────────────────────────────────────────────────────────────
    if n_steps == 0:
        print('No steps evaluated.')
        return

    print('\n' + '='*60)
    pos_unit = 'm' if gt_pos_mu is not None else 'norm'
    print(f'Split: {args.split}  |  Sequences: {len(dataset)}  |  Steps: {n_steps}')
    print(f'Avg position L1   : {total_pos_err / n_steps:.4f} {pos_unit}')
    print(f'Avg yaw error     : {total_yaw_err / n_steps:.2f}°')
    for k in box_keys:
        print(f'  {k:20s}: {box_sum[k] / n_steps:.4f}')
    print(f'Total active pairs: {n_pairs}')
    print(f'Avg inference time: {all_infer_t / len(dataset):.4f} s/sequence')

    # ── per-class breakdown ───────────────────────────────────────────────
    print('\n--- Per-class errors (pos m / yaw °) ---')
    rows = sorted(per_class.items(), key=lambda kv: kv[1]['yaw'] / max(kv[1]['n'], 1),
                  reverse=True)
    urdf_pos = urdf_yaw = urdf_n = 0.0
    nonu_pos = nonu_yaw = nonu_n = 0.0
    for lbl, acc in rows:
        n = acc['n']
        if n == 0:
            continue
        p = acc['pos'] / n
        y = acc['yaw'] / n
        tag = '[URDF]' if lbl in _URDF_LABELS else '      '
        print(f'  {tag} {lbl:25s}  n={n:4d}  pos={p:.3f}m  yaw={y:.1f}°')
        if lbl in _URDF_LABELS:
            urdf_pos += acc['pos']; urdf_yaw += acc['yaw']; urdf_n += n
        else:
            nonu_pos += acc['pos']; nonu_yaw += acc['yaw']; nonu_n += n
    print(f'\n  [URDF]  avg  pos={urdf_pos/max(urdf_n,1):.3f}m  yaw={urdf_yaw/max(urdf_n,1):.1f}°  (n={int(urdf_n)})')
    print(f'  [nonU]  avg  pos={nonu_pos/max(nonu_n,1):.3f}m  yaw={nonu_yaw/max(nonu_n,1):.1f}°  (n={int(nonu_n)})')

    result = {
        'split': args.split, 'epoch': args.epoch,
        'n_sequences': len(dataset), 'n_steps': n_steps, 'n_pairs': n_pairs,
        f'pos_L1_{pos_unit}': total_pos_err / n_steps,
        'yaw_deg':  total_yaw_err / n_steps,
        **{k: box_sum[k] / n_steps for k in box_keys},
        'avg_infer_s': all_infer_t / len(dataset),
    }
    out_path = os.path.join(args.exp, f'eval_{args.split}_ep{args.epoch}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Results saved to {out_path}')


if __name__ == '__main__':
    evaluate()
