"""
Measure end-to-end latency of the inference + visualisation pipeline.

Stages timed per step:
  1. to_cuda       – host-to-device transfer
  2. gcn           – model.forward()
  3. postproc      – denorm / pos / yaw / match bookkeeping  (CPU)
  4. sz_retrieval  – shape_lib.retrieve_sizes_batch()
  5. sh_retrieval  – per-active-node shape_lib.retrieve()
  6. mesh_load     – GLB / URDF loading from disk + trimesh ops + pose application
                     (subsequent calls hit the in-process cache, as in the real pipeline)

Usage:
    python scripts/bench_pipeline.py --seq 31_1 --n_repeat 5
    python scripts/bench_pipeline.py              # first 10 val sequences, 3 repeats
"""

import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from dataset.Tele_dataset_incremental import TeleDatasetSceneGraphIncremental
from model.AE_tele_incremental import Sg2ScAEModelTeleIncremental
from helpers.tele_shape_features import TeleShapeLibrary


# ── helpers ───────────────────────────────────────────────────────────────────
def _denorm(x, mu, sg):
    return x * sg + mu


def _load_norm_stats(path, dims):
    s = np.loadtxt(path)
    return (torch.tensor(s[0, dims], dtype=torch.float32),
            torch.tensor(s[1, dims], dtype=torch.float32))


def to_cuda(x, device):
    if torch.is_tensor(x): return x.to(device, non_blocking=True)
    if isinstance(x, dict): return {k: to_cuda(v, device) for k, v in x.items()}
    if isinstance(x, list): return [to_cuda(v, device) for v in x]
    return x


# ── mesh loading (mirrors eval_tele_incremental.py) ──────────────────────────
import glob, xml.etree.ElementTree as ET
import trimesh

_URDF_FILES = {
    'fridge':             'fridge.urdf',
    'kitchen_counter':    'kitchen_counter.urdf',
    'kitchen_cupboards':  'kitchen_cupboards.urdf',
    'chest_of_drawers':   'chestOfDrawers_01.urdf',
}
_MESH_CANONICAL_PITCH = {'clothes_hanger': math.radians(90.0)}
_mesh_cache = {}
_urdf_cache = {}


def _canon(label):
    return label.replace(' ', '_').lower()


def _pitch_mat3(pitch):
    c, s = math.cos(pitch), math.sin(pitch)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def _yaw_mat3(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


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


def _load_mesh_glb(obj_dir, label):
    canon = _canon(label)
    if canon in _mesh_cache:
        r = _mesh_cache[canon]
        return (r[0].copy() if r[0] is not None else None, r[1])
    glbs = glob.glob(os.path.join(obj_dir, canon, '*.glb'))
    if not glbs:
        _mesh_cache[canon] = (None, None); return None, None
    loaded = trimesh.load(glbs[0])
    if isinstance(loaded, trimesh.Scene):
        parts = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            _mesh_cache[canon] = (None, None); return None, None
        vc = _extract_vc(parts); mesh = trimesh.util.concatenate(parts)
    elif isinstance(loaded, trimesh.Trimesh):
        vc = _extract_vc([loaded]); mesh = loaded
    else:
        _mesh_cache[canon] = (None, None); return None, None
    pitch = _MESH_CANONICAL_PITCH.get(label, 0.0)
    if pitch != 0.0:
        v = np.asarray(mesh.vertices, np.float64)
        c = (v.min(0) + v.max(0)) / 2
        mesh.vertices = (v - c) @ _pitch_mat3(pitch).T + c
    _mesh_cache[canon] = (mesh, vc)
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


def _load_urdf(obj_dir, label):
    canon = _canon(label)
    if canon in _urdf_cache:
        return [(m.copy(), c) for m, c in _urdf_cache[canon]]
    try:
        robot = ET.parse(os.path.join(obj_dir, canon, _URDF_FILES[canon])).getroot()
    except Exception:
        _urdf_cache[canon] = []; return []
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
        gp = os.path.join(obj_dir, canon, mel.get('filename'))
        if not os.path.exists(gp): continue
        T = world_T(lnk.get('name')) @ _origin_T(vis.find('origin'))
        loaded = trimesh.load(gp)
        if isinstance(loaded, trimesh.Scene):
            sub = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
            if not sub: continue
            mesh = trimesh.util.concatenate(sub)
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        else:
            continue
        mesh.apply_transform(T)
        parts.append((mesh, None))
    _urdf_cache[canon] = parts
    return [(m.copy(), c) for m, c in parts]


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
        result.append(m)
    return result


def _build_step_meshes(nodes_vis, obj_dir):
    """Load + place meshes for one step. Returns list of trimesh.Trimesh."""
    meshes = []
    for node in nodes_vis:
        label = node['label']
        op    = node['op']
        if op in (0, 3):      # OLD / REMOVE
            continue
        L, H, W = node['L'], node['H'], node['W']
        center  = [node['cx'], node['cy'], node['cz']]
        yaw     = node['yaw']
        canon   = _canon(label)
        if canon in _URDF_FILES:
            parts = _load_urdf(obj_dir, label)
            if parts:
                meshes.extend(_place_parts(parts, center, L, H, W, yaw))
                continue
        mesh, _ = _load_mesh_glb(obj_dir, label)
        if mesh is not None:
            meshes.append(_place_mesh(mesh, center, yaw))
    return meshes


# ── args ──────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp',        default='/home/gao/data_weight/tele_incremental_12')
    p.add_argument('--epoch',      type=int, default=199)
    p.add_argument('--robot_root', default='/media/gao/gao_ssd/3D scene graph')
    p.add_argument('--gt_root',    default='/media/gao/gao_ssd/3D Scene')
    p.add_argument('--enc_stats',  default='/media/gao/gao_ssd/3D Scene/boxes_stats_tele_enc_trainval.txt')
    p.add_argument('--gt_stats',   default='/media/gao/gao_ssd/3D Scene/boxes_stats_tele_gt_trainval.txt')
    p.add_argument('--splits_dir', default='/media/gao/gao_ssd/3D Scene/splits')
    p.add_argument('--shape_pkl',  default='/media/gao/gao_ssd/3D objects/features.pkl')
    p.add_argument('--obj_dir',    default='/media/gao/gao_ssd/3D objects')
    p.add_argument('--seq',        default=None)
    p.add_argument('--n_seq',      type=int, default=10)
    p.add_argument('--n_repeat',   type=int, default=3)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    device = torch.device('cuda')

    with open(os.path.join(args.exp, 'args.json')) as f:
        ta = json.load(f)

    classes = TeleDatasetSceneGraphIncremental.TELE_CLASSES
    vocab = {
        'object_idx_to_name': {i: c + '\n' for i, c in enumerate(classes)},
        'pred_idx_to_name':   {i: r for i, r in
                               enumerate(TeleDatasetSceneGraphIncremental.TELE_RELATIONS)},
    }
    model = Sg2ScAEModelTeleIncremental(
        vocab=vocab,
        node_embedding_dim=ta['node_dim'],
        gconv_num_layers=ta['num_layers'],
        shape_feat_dim=ta['shape_feat_dim'],
        residual=ta['residual'],
        gconv_pooling=ta['pooling'],
    ).to(device)
    ckpt_path = os.path.join(args.exp, f'checkpoint/model_{args.epoch}.pth')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    shape_lib = TeleShapeLibrary(args.shape_pkl)
    enc_box_mu,  enc_box_sg  = _load_norm_stats(args.enc_stats, slice(0, 7))
    enc_size_mu, enc_size_sg = _load_norm_stats(args.enc_stats, slice(0, 3))
    gt_pos_mu,   gt_pos_sg   = _load_norm_stats(args.gt_stats,  slice(3, 6))

    ds = TeleDatasetSceneGraphIncremental(
        robot_sg_root=args.robot_root,
        gt_scene_root=args.gt_root,
        split='val',
        enc_normalized_stats=args.enc_stats,
        gt_normalized_stats=args.gt_stats,
        splits_dir=args.splits_dir,
    )
    loader = DataLoader(ds, batch_size=1, collate_fn=ds.collate_fn_inc,
                        shuffle=False, num_workers=0)

    # warmup
    print('Warming up...')
    for i, batch in enumerate(loader):
        b_gpu = to_cuda(batch, device)
        with torch.no_grad():
            preds = model(b_gpu, reset_states=True)
        enc = batch['encoder']
        n   = enc['objs'].shape[0]
        lbls = [classes[enc['objs'][j].item()] for j in range(n)]
        _ = shape_lib.retrieve_sizes_batch(preds['pred_feat'].cpu(), tele_labels=lbls)
        for j in range(min(n, 3)):
            lbl = shape_lib.map_label(lbls[j])
            if lbl:
                _ = shape_lib.retrieve(preds['pred_feat'][j].cpu(), label_filter=lbl)
        # warmup mesh cache
        for j in range(min(n, 5)):
            _load_mesh_glb(args.obj_dir, lbls[j])
        if i >= 4:
            break
    print('Done.\n')

    STAGES = ['to_cuda', 'gcn', 'postproc', 'sz_retrieval', 'sh_retrieval', 'mesh_load']
    totals = {s: 0.0 for s in STAGES}
    n_seq_done = 0

    hdr = (f'{"Seq":<8} {"Steps":>5} | '
           + ' '.join(f'{s:>13}' for s in STAGES)
           + f' | {"Total":>9} {"Hz":>7}')
    print(hdr)
    print('-' * len(hdr))

    for batch in loader:
        sid = batch['step_meta']['scan_id_str'][0]
        if args.seq and args.seq not in sid:
            continue
        n_steps = len(batch['active_matches'])
        if n_steps == 0:
            continue

        enc_cpu  = batch['encoder']
        n_obj    = enc_cpu['objs'].shape[0]
        all_lbls = [classes[enc_cpu['objs'][j].item()] for j in range(n_obj)]

        stage_t = {s: 0.0 for s in STAGES}

        for _ in range(args.n_repeat):

            # 1) to_cuda
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            b_gpu = to_cuda(batch, device)
            torch.cuda.synchronize()
            stage_t['to_cuda'] += time.perf_counter() - t0

            # 2) GCN
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                preds = model(b_gpu, reset_states=True)
            torch.cuda.synchronize()
            stage_t['gcn'] += time.perf_counter() - t0

            feats_cpu = preds['pred_feat'].cpu()
            pred_pos  = preds['pred_pos'].cpu()
            pred_rot  = preds['pred_rot'].cpu()
            enc_gpu   = b_gpu['encoder']
            gt_boxes  = b_gpu['gt']['boxes'].cpu()

            # 3) post-processing
            t0 = time.perf_counter()
            for flat_sid, matches in enumerate(batch['active_matches']):
                if not matches:
                    continue
                enc_idx = torch.tensor([m[0] for m in matches], dtype=torch.long)
                gt_idx  = torch.tensor([m[1] for m in matches], dtype=torch.long)
                # m[2] is dist_m (unused here)
                p_pos_m = _denorm(pred_pos[enc_idx], gt_pos_mu, gt_pos_sg)
                g_pos_m = _denorm(gt_boxes[gt_idx, 3:6], gt_pos_mu, gt_pos_sg)
                _       = (p_pos_m - g_pos_m).norm(dim=1)
                _       = _denorm(enc_gpu['boxes'][enc_idx].cpu()[:, :3],
                                  enc_size_mu, enc_size_sg)
            stage_t['postproc'] += time.perf_counter() - t0

            # 4) size retrieval
            t0 = time.perf_counter()
            _ = shape_lib.retrieve_sizes_batch(feats_cpu, tele_labels=all_lbls)
            stage_t['sz_retrieval'] += time.perf_counter() - t0

            # 5) shape retrieval (per active node)
            t0 = time.perf_counter()
            for matches in batch['active_matches']:
                for enc_i, _gi, _dist in matches:
                    lbl = shape_lib.map_label(all_lbls[enc_i])
                    if lbl:
                        _ = shape_lib.retrieve(feats_cpu[enc_i], label_filter=lbl)
            stage_t['sh_retrieval'] += time.perf_counter() - t0

            # 6) mesh load + place (per step, mimics real-time SLAM vis loop)
            t0 = time.perf_counter()
            for flat_sid, matches in enumerate(batch['active_matches']):
                nodes_vis = []
                obj_mask  = (enc_cpu['obj_to_flat_step'] == flat_sid)
                for gi in obj_mask.nonzero(as_tuple=True)[0].tolist():
                    lbl = all_lbls[gi]
                    op  = int(enc_cpu['op_types'][gi].item())
                    brw = _denorm(enc_cpu['boxes'][gi].unsqueeze(0),
                                  enc_box_mu, enc_box_sg)[0]
                    nodes_vis.append({
                        'label': lbl, 'op': op,
                        'L': float(brw[0]), 'H': float(brw[1]), 'W': float(brw[2]),
                        'cx': float(brw[3]), 'cy': float(brw[4]), 'cz': float(brw[5]),
                        'yaw': float(brw[6]),
                    })
                _build_step_meshes(nodes_vis, args.obj_dir)
            stage_t['mesh_load'] += time.perf_counter() - t0

        r = args.n_repeat
        per_step = {s: stage_t[s] / r / n_steps * 1000 for s in STAGES}
        total_ms = sum(per_step.values())
        hz = 1000.0 / total_ms if total_ms > 0 else 0.0

        row = (f'{sid:<8} {n_steps:>5} | '
               + ' '.join(f'{per_step[s]:>13.2f}' for s in STAGES)
               + f' | {total_ms:>9.2f} {hz:>7.1f}')
        print(row)

        for s in STAGES:
            totals[s] += per_step[s]
        n_seq_done += 1
        if n_seq_done >= args.n_seq:
            break

    if n_seq_done == 0:
        print('No sequences matched.')
        return

    print('-' * len(hdr))
    mean_per = {s: totals[s] / n_seq_done for s in STAGES}
    mean_tot = sum(mean_per.values())
    mean_hz  = 1000.0 / mean_tot if mean_tot > 0 else 0.0
    row = (f'{"MEAN":<8} {"":>5} | '
           + ' '.join(f'{mean_per[s]:>13.2f}' for s in STAGES)
           + f' | {mean_tot:>9.2f} {mean_hz:>7.1f}')
    print(row)
    print('\nAll units: ms/step')


if __name__ == '__main__':
    main()
