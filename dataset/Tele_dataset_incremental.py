from __future__ import print_function
import sys
import warnings
warnings.filterwarnings('ignore', message='Gimbal lock detected', category=UserWarning)
sys.path.append("..")
sys.path.append(".")

import os
import os.path
import glob
import json
import random

import numpy as np
import torch
import torch.utils.data as data
from collections import defaultdict

from helpers.util import normalize_box_params

# URDF objects placed in Habitat use Z-up quaternions; use a dedicated yaw
# formula rather than the generic Y-up Euler decomposition.
_ZUPYAW_LABELS = {'kitchen_counter', 'kitchen_cupboards', 'kitchen_cupboard',
                   'chest_of_drawers', 'fridge'}


class TeleDatasetSceneGraphIncremental(data.Dataset):
    """
    Per-sequence incremental dataset: noisy robot scene graph → GT 3D scene.
    Positions stay in the original world/map coordinate frame; no scene
    centering is applied.

    Data sources
    ------------
    robot_sg_root  /media/gao/gao_ssd/3D scene graph/<seq>/frame_XXXXXX.json
        Cumulative scene graph built by the robot.
        Nodes: {id, label, centroid, bbox_max, bbox_min}

    gt_scene_root  /media/gao/gao_ssd/3D Scene/<seq>/sg_frame_XXXXX.json
        Per-frame GT annotation (currently-visible objects).
        Nodes: {node_id, label, model, center, orientation[w,x,y,z], size{L,W,H}}

    Frame correspondence:  robot frame N  →  GT frame N.
    First frame of every robot sequence is skipped (always empty).

    Sample / step structure
    -----------------------
    One sample = one sequence.
    One step   = a frame at which the robot SG node-id set changed.

    Active-node set  (O_active = O_add ∪ O_update)
    ------------------------------------------------
    O_add     newly seen node_ids at this step
    O_update  same node_id but robot-side label/box changed
              (e.g. after SLAM loop-closure correction)
    O_remove  node_ids dropped from the current frame (robot corrected a
              false positive / over-segmentation).  Visible to the GCN
              with op_type=3 but excluded from active_matches.  Removed
              from the accumulated state after this step.

    Per-step dict keys
    ------------------
    encoder
      objs        (N,)    int64   class ids for all accumulated nodes
      boxes       (N,7)   float32 [l,h,w, cx,cy,cz, yaw_hint]  world/map frame,
                                   robot observations (noisy; yaw_hint only
                                   initialises rotation prediction)
      triples     (T,3)   int64   close-by spatial relations
      valid_mask  (N,)    bool    True for real objects
      op_types    (N,)    int64   0=OLD  1=ADD  2=UPDATE  3=REMOVE
      delta_x     (N,7)   float32 v_cur − v_old  (zeros for OLD/ADD/REMOVE)
      confidence  (N,)    float32 real confidence from robot SG
      active_mask (N,)    bool    True for ADD or UPDATE nodes (not REMOVE)

    gt            clean GT scene for this frame
      objs        (G,)    int64
      boxes       (G,7)   float32 [l,h,w, cx,cy,cz, yaw]  (yaw from quaternion)
      triples     (T2,3)  int64

    active_matches  list of (enc_idx, gt_idx, dist_m)  # dist_m: raw 3D distance in metres
        Hungarian-matched pairs restricted to active encoder nodes.
        Provides per-object supervision for position, yaw, and shape feature.

    unmatched_enc   enc indices with no GT match  (E5 false positives)
    unmatched_gt    GT  indices with no enc match (E4 missed detections)

    new_idx   (K_add,) int64  indices of ADD nodes inside encoder
    old_idx   (N-K,)   int64  indices of all non-ADD nodes inside encoder

    Loss targets (derived from active_matches)
    ------------------------------------------
    For each matched pair (enc_idx, gt_idx):
      GT position: gt['boxes'][gt_idx, 3:6]          world/map frame
      GT yaw:      gt['boxes'][gt_idx, 6]             radians
      (size not supervised)

    Perception errors handled
    -------------------------
    E1 position noise  0.1–0.3 m centroid offset
    E2 size noise      bbox smaller than GT (occlusion)
    E3 label error     cross-class mismatch → E4+E5
    E4 missed detect   unmatched_gt
    E5 false positive  unmatched_enc
    E6 no orientation  robot yaw uses AABB hint only → GT yaw supervised via loss
    """

    LABEL_NORM = {
        'chestOfDrawers': 'chest_of_drawers',
        'kitchenCupboard': 'kitchen_cupboard',
    }

    TELE_CLASSES = sorted([
        '_scene_', 'beanbag', 'bike', 'cabinet', 'chair', 'chest_of_drawers',
        'clothes_hanger', 'door', 'fridge', 'indoor_plant', 'kitchen_counter',
        'kitchen_cupboard', 'sofa', 'table', 'tv_screen', 'tvstand', 'wall_cabinet',
        'floor',
    ])

    TELE_RELATIONS = ['close by', 'left', 'right', 'front', 'behind', 'above', 'standing on']

    UPDATE_POS_THRESH  = 0.05   # metres: centroid L2 shift to count as UPDATE
    UPDATE_SIZE_THRESH = 0.10   # metres: L2 of (dl,dh,dw) to count as UPDATE
    LABEL_MISMATCH_PENALTY = 0.5  # metres added to matching cost when labels differ

    def __init__(self, robot_sg_root, gt_scene_root,
                 split='train', shuffle_objs=False,
                 data_len=None, eval=False, seed=True,
                 with_feats=False,
                 enc_normalized_stats=None,
                 gt_normalized_stats=None,
                 shape_features_pkl=None,
                 close_by_thresh=2.0, match_max_dist=1.5,
                 splits_dir=None):

        self.robot_sg_root      = robot_sg_root
        self.gt_scene_root      = gt_scene_root
        self.shuffle_objs       = shuffle_objs
        self.data_len           = data_len
        self.eval               = eval
        self.with_feats         = with_feats
        self.enc_normalized_stats = enc_normalized_stats   # robot SG stats
        self.gt_normalized_stats  = gt_normalized_stats    # GT stats
        self.close_by_thresh    = close_by_thresh
        self.match_max_dist     = match_max_dist

        self.shape_lib = None
        if shape_features_pkl:
            from helpers.tele_shape_features import TeleShapeLibrary
            self.shape_lib = TeleShapeLibrary(shape_features_pkl)

        # Pre-load encoder std for delta_x normalisation (Bug 4 fix)
        self._enc_std = None
        if enc_normalized_stats:
            _stats = np.loadtxt(enc_normalized_stats)
            self._enc_std = _stats[1].astype(np.float32)   # (7,) std values

        if eval and seed:
            np.random.seed(47); torch.manual_seed(47); random.seed(47)

        # Vocabulary
        self.classes             = {c: i for i, c in enumerate(self.TELE_CLASSES)}
        self.classes_r           = {i: c for c, i in self.classes.items()}
        self.fine_grained_classes = dict(self.classes)
        self.relationships       = self.TELE_RELATIONS
        self.relationships_dict  = {r: i + 1 for i, r in enumerate(self.relationships)}
        self.relationships_dict_r = {i: r for r, i in self.relationships_dict.items()}
        self.vocab = {
            'object_idx_to_name': [c + '\n' for c in self.TELE_CLASSES],
            'pred_idx_to_name':   ['in\n'] + [r + '\n' for r in self.relationships],
        }

        # ---- build sequence index (with disk cache to avoid re-scanning 44K files) ----
        # Cache key: mtime of robot_sg_root dir + thresholds.
        import pickle, hashlib
        _cache_key = hashlib.md5(
            f'{robot_sg_root}|{gt_scene_root}|'
            f'{close_by_thresh}|{match_max_dist}'.encode()).hexdigest()[:12]
        _cache_path = os.path.join(robot_sg_root, f'.seq_index_{_cache_key}.pkl')

        if os.path.exists(_cache_path):
            with open(_cache_path, 'rb') as _f:
                self.sequences = pickle.load(_f)
            print(f'[TeleDataset] loaded index from cache ({len(self.sequences)} seqs total)')
        else:
            self.sequences = []
            seq_dirs = sorted(glob.glob(os.path.join(robot_sg_root, '*/')))
            for seq_dir in seq_dirs:
                seq = os.path.basename(seq_dir.rstrip('/'))
                if seq in ('old', 'tmp'):
                    continue                         # ignore legacy / scratch data
                robot_files = sorted(glob.glob(os.path.join(seq_dir, 'frame_*.json')))
                if len(robot_files) < 2:
                    continue

                change_frames = []
                prev_ids:  set  = set()
                prev_cen:  dict = {}   # nid → centroid ndarray(3)
                prev_size: dict = {}   # nid → [l,h,w]  ndarray(3)

                for rf in robot_files:
                    with open(rf) as fp:
                        rd = json.load(fp)
                    cur_nodes = rd.get('nodes', [])
                    cur_ids   = {n['id'] for n in cur_nodes}
                    cur_cen   = {n['id']: np.array(n['center'],  dtype=np.float64)
                                 for n in cur_nodes}
                    cur_size  = {n['id']: np.array(n['bbox_max'],  dtype=np.float64) -
                                           np.array(n['bbox_min'],  dtype=np.float64)
                                 for n in cur_nodes}

                    if not prev_ids and not cur_ids:
                        prev_ids = cur_ids; prev_cen = cur_cen; prev_size = cur_size
                        continue

                    added   = cur_ids - prev_ids
                    removed = prev_ids - cur_ids
                    # also capture frames where existing nodes drifted significantly
                    # (e.g. SLAM loop-closure corrects positions without add/remove)
                    attr_changed = any(
                        np.linalg.norm(cur_cen[nid] - prev_cen[nid]) > self.UPDATE_POS_THRESH
                        or np.linalg.norm(cur_size[nid] - prev_size[nid]) > self.UPDATE_SIZE_THRESH
                        for nid in (cur_ids & prev_ids) if nid in prev_cen
                    )

                    if added or removed or attr_changed:
                        robot_fid = int(rd['frame'])
                        # GT frame is the SAME frame (no -1 offset)
                        gt_path = os.path.join(gt_scene_root, seq,
                                               f'sg_frame_{robot_fid:05d}.json')
                        if os.path.exists(gt_path):
                            change_frames.append({
                                'robot_path':  rf,
                                'gt_path':     gt_path,
                                'added_ids':   added,
                                'removed_ids': removed,
                            })
                    prev_ids = cur_ids; prev_cen = cur_cen; prev_size = cur_size

                if change_frames:
                    self.sequences.append({'seq': seq, 'change_frames': change_frames})

            with open(_cache_path, 'wb') as _f:
                pickle.dump(self.sequences, _f)
            print(f'[TeleDataset] index built and cached → {_cache_path}')

        # train / val / test split
        # If splits_dir is provided, load sequence names from {split}_sequences.txt.
        # Otherwise fall back to automatic room-based splitting.
        split_key = split.replace('_scans', '')   # 'train_scans' → 'train'
        if splits_dir is not None:
            split_file = os.path.join(splits_dir, f'{split_key}_sequences.txt')
            if os.path.isfile(split_file):
                with open(split_file) as f:
                    allowed = {l.strip() for l in f if l.strip()}
                self.sequences = [s for s in self.sequences if s['seq'] in allowed]
                print(f'[TeleDataset] split loaded from {split_file}')
            else:
                raise FileNotFoundError(f'Split file not found: {split_file}')
        else:
            all_rooms = sorted(
                {s['seq'].split('_')[0] for s in self.sequences},
                key=lambda x: int(x))
            n_test = max(1, len(all_rooms) // 10)
            n_val  = max(1, len(all_rooms) // 10)
            _test_rooms  = set(all_rooms[-n_test:])
            _val_rooms   = set(all_rooms[-(n_val + n_test):-n_test])
            _train_rooms = set(all_rooms) - _val_rooms - _test_rooms

            if split_key == 'train':
                self.sequences = [s for s in self.sequences
                                  if s['seq'].split('_')[0] in _train_rooms]
            elif split_key == 'val':
                self.sequences = [s for s in self.sequences
                                  if s['seq'].split('_')[0] in _val_rooms]
            elif split_key == 'test':
                self.sequences = [s for s in self.sequences
                                  if s['seq'].split('_')[0] in _test_rooms]

        self.scans        = [s['seq'] for s in self.sequences]
        self.scan_name2idx = {s: i for i, s in enumerate(self.scans)}

        self.n_steps = [len(s['change_frames']) for s in self.sequences]
        print(f'[TeleDataset] {len(self.sequences)} seqs ({split}), '
              f'steps/seq: avg {np.mean(self.n_steps):.1f}  '
              f'min {min(self.n_steps)}  max {max(self.n_steps)}')

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _norm_label(label):
        return TeleDatasetSceneGraphIncremental.LABEL_NORM.get(label, label)

    @staticmethod
    def _quat_yaw_yup(wxyz):
        """Yaw (rad) from Y-up quaternion [w, x, y, z]."""
        w, x, y, z = wxyz
        return float(np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (x * x + y * y)))

    @staticmethod
    def _quat_to_euler_yup(wxyz):
        """
        Full Euler angles (yaw, pitch, roll) in radians from Y-up quaternion.

        Convention: YXZ intrinsic (yaw around Y first, then pitch around X,
        then roll around Z).  Most furniture has pitch≈roll≈0; leaning objects
        (e.g. tables against walls) have non-zero pitch.

        Returns np.ndarray([yaw, pitch, roll]).
        """
        from scipy.spatial.transform import Rotation
        w, x, y, z = wxyz
        r = Rotation.from_quat([x, y, z, w])   # scipy uses [x,y,z,w]
        yaw, pitch, roll = r.as_euler('YXZ', degrees=False)
        return np.array([yaw, pitch, roll], dtype=np.float32)

    # ------------------------------------------------------------------
    # Frame loading
    # ------------------------------------------------------------------

    def _load_robot_frame(self, path):
        """[(node_id, label, param7, confidence)]  param7=[l,h,w,cx,cy,cz,yaw_hint]."""
        with open(path) as fp:
            d = json.load(fp)
        out = []
        for n in d.get('nodes', []):
            label = self._norm_label(n['label'])
            if label not in self.classes:
                continue
            bmin = np.array(n['bbox_min'], dtype=np.float64)
            bmax = np.array(n['bbox_max'], dtype=np.float64)
            cen  = np.array(n['center'], dtype=np.float64)
            dx, dy, dz = bmax[0]-bmin[0], bmax[1]-bmin[1], bmax[2]-bmin[2]
            yaw_hint = float(np.arctan2(dz, dx))   # >0 → wider in Z (≈±90°), <0 → wider in X (≈0°/180°)
            p7   = np.array([dx, dy, dz, cen[0], cen[1], cen[2], yaw_hint])
            conf = float(n.get('confidence', 1.0))
            out.append((int(n['id']), label, p7, conf))
        return out

    def _load_gt_frame(self, path):
        """[(label, param7, rot3)]
        param7 = [L,H,W,cx,cy,cz,yaw]
        rot3   = [yaw, pitch, roll]  full Euler angles (YXZ, radians)
        """
        with open(path) as fp:
            d = json.load(fp)
        out = []
        for n in d.get('nodes', []):
            label = self._norm_label(n['label'])
            if label not in self.classes:
                continue
            sz   = n['size']
            cen  = np.array(n['center'], dtype=np.float64)
            if label in _ZUPYAW_LABELS:
                # Habitat places these objects with a Z-up quaternion.
                # Extract yaw around Z rather than the Y-up formula to avoid
                # the ~103° error that results from decoding with the wrong convention.
                w, x, y, z = n['orientation']
                primary_yaw = float(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
                rot3 = np.array([primary_yaw, 0.0, 0.0], dtype=np.float32)
            else:
                rot3 = self._quat_to_euler_yup(n['orientation'])  # [yaw, pitch, roll]
            p7    = np.array([sz['L'], sz['H'], sz['W'],
                               cen[0], cen[1], cen[2], rot3[0]])
            model = n.get('model', '')
            out.append((label, p7, rot3, model))
        return out

    # ------------------------------------------------------------------
    # Hungarian matching
    # ------------------------------------------------------------------

    @staticmethod
    def match_nodes(robot_objs, gt_objs, max_dist=1.5,
                    label_mismatch_penalty=0.5):
        """
        Hungarian matching with same-class preference.

        Base cost is L2 centroid distance.  A fixed penalty is added when
        robot and GT labels differ, so same-class pairs are preferred but
        cross-class matching is still allowed when the distance difference
        outweighs the penalty (i.e. the robot label is genuinely wrong).

        robot_objs : [(nid, label, p7), …]
        gt_objs    : [(label, p7), …]

        Returns
        -------
        matches        [(enc_idx, gt_idx), …]
        unmatched_enc  [enc_idx, …]   E5 false positives
        unmatched_gt   [gt_idx,  …]   E4 missed detections
        """
        from scipy.optimize import linear_sum_assignment

        R = len(robot_objs)
        G = len(gt_objs)
        if R == 0 or G == 0:
            return [], list(range(R)), list(range(G))

        r_pos = np.array([p7[3:6] for _, _, p7 in robot_objs])   # (R, 3)
        g_pos = np.array([p7[3:6] for _,    p7 in gt_objs])      # (G, 3)
        cost  = np.linalg.norm(r_pos[:, None] - g_pos[None, :], axis=-1)  # (R, G)

        # penalise cross-class pairs so same-class matches are preferred
        for ri, (_, r_lb, _) in enumerate(robot_objs):
            for gi, (g_lb, _) in enumerate(gt_objs):
                if r_lb != g_lb:
                    cost[ri, gi] += label_mismatch_penalty

        ri_a, gi_a = linear_sum_assignment(cost)

        matches   = []
        matched_r = set()
        matched_g = set()
        for ri, gi in zip(ri_a, gi_a):
            # filter by raw distance (before penalty) to keep max_dist meaningful
            raw_dist = np.linalg.norm(r_pos[ri] - g_pos[gi])
            if raw_dist <= max_dist:
                matches.append((ri, gi, float(raw_dist)))
                matched_r.add(ri)
                matched_g.add(gi)

        return (matches,
                [i for i in range(R) if i not in matched_r],
                [j for j in range(G) if j not in matched_g])

    # ------------------------------------------------------------------
    # Build graph dict from node list
    # ------------------------------------------------------------------

    def _build_dict(self, labels, p7s, stats_file=None):
        """
        labels     : list[str]
        p7s        : list[np.ndarray(7,)]
        stats_file : path to the normalization stats for THIS side
                     (enc_normalized_stats or gt_normalized_stats)
        Returns raw dict (not yet torchified).
        """
        cat_ids = []
        boxes   = []
        centers = []
        words   = []

        for label, p7 in zip(labels, p7s):
            cat_ids.append(self.classes[label])
            box = p7.copy()
            centers.append(box[3:6].copy())
            yaw = box[6]
            if stats_file:
                box = normalize_box_params(box, file=stats_file)
            box[6] = yaw      # restore raw-radian yaw (not normalized)
            boxes.append(box)

        centers = np.array(centers) if centers else np.zeros((0, 3))
        pred    = self.relationships_dict['close by']
        triples = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                if np.linalg.norm(centers[i] - centers[j]) < self.close_by_thresh:
                    triples += [[i, pred, j], [j, pred, i]]
                    words   += [f'{labels[i]} close by {labels[j]}',
                                f'{labels[j]} close by {labels[i]}']

        out = {
            'objs':         cat_ids,
            'objs_grained': list(cat_ids),
            'triples':      triples,
            'boxes':        boxes,
            'words':        words,
        }
        if self.with_feats:
            feats = [np.zeros([1, 768], dtype=np.float32) for _ in labels]
            feats += [np.zeros([1, 768], dtype=np.float32)]
            out['feats'] = list(np.concatenate(feats, axis=0))
        return out

    @staticmethod
    def _torchify(d):
        d['objs']         = torch.from_numpy(np.array(d['objs'],         dtype=np.int64))
        d['objs_grained'] = torch.from_numpy(np.array(d['objs_grained'], dtype=np.int64))
        d['boxes']        = torch.from_numpy(np.array(d['boxes'],        dtype=np.float32))
        t = (np.array(d['triples'], dtype=np.int64).reshape(-1, 3)
             if d['triples'] else np.empty((0, 3), dtype=np.int64))
        d['triples'] = torch.from_numpy(t)
        if 'feats' in d:
            d['feats'] = torch.from_numpy(np.array(d['feats'], dtype=np.float32))

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        seq_info = self.sequences[index]

        # Accumulated robot SG (grows over steps, updated in-place for UPDATE)
        acc_labels:  list = []         # str per accumulated node
        acc_p7s:     list = []         # np.ndarray(7,) per accumulated node
        acc_confs:   list = []         # float confidence per accumulated node
        id_to_idx:  dict = {}          # node_id → enc_idx
        idx_to_id:  list = []         # enc_idx  → node_id
        prev_p7:    dict = {}          # node_id  → last observed p7

        # Keep world/map coordinates.  scene_center remains in the sample for
        # compatibility with collate/eval visualisation, but is always zero.
        scene_center = np.zeros(3, dtype=np.float64)

        steps = []

        for cf in seq_info['change_frames']:
            robot_raw = self._load_robot_frame(cf['robot_path'])
            gt_raw    = self._load_gt_frame(cf['gt_path'])

            if not robot_raw:
                continue

            # ---- step 1: robot-side classification (O_add / O_update) ────────
            O_add         = set()
            O_update      = set()
            O_update_phys = set()   # UPDATE triggered by physical box change
            nid_to_cur_p7   = {nid: p7   for nid, _, p7, _    in robot_raw}
            nid_to_cur_conf = {nid: conf for nid, _, _,  conf in robot_raw}

            for nid, label, p7, conf in robot_raw:
                if nid in cf['added_ids']:
                    O_add.add(nid)
                    id_to_idx[nid] = len(acc_labels)
                    idx_to_id.append(nid)
                    acc_labels.append(label)
                    acc_p7s.append(p7.copy())
                    acc_confs.append(conf)
                elif nid in id_to_idx:
                    ei   = id_to_idx[nid]
                    prev = prev_p7.get(nid)
                    changed = False

                    # robot SG self-corrected its own label estimate
                    if label != acc_labels[ei]:
                        acc_labels[ei] = label
                        changed = True

                    if prev is not None:
                        pos_shift  = np.linalg.norm(p7[3:6] - prev[3:6])
                        size_shift = np.linalg.norm(p7[0:3] - prev[0:3])
                        if (pos_shift  > self.UPDATE_POS_THRESH or
                                size_shift > self.UPDATE_SIZE_THRESH):
                            changed = True
                            O_update_phys.add(nid)   # physical change

                    if changed:
                        O_update.add(nid)
                        acc_p7s[ei]  = p7.copy()
                        acc_confs[ei] = conf          # refresh confidence

            # ---- detect REMOVE nodes ────────────────────────────────────────
            # Nodes present in prev step but absent now: robot corrected a
            # false positive (e.g. one object mis-detected as two).
            O_remove = set()
            for nid in cf['removed_ids']:
                if nid in id_to_idx:
                    O_remove.add(nid)

            # ---- step 2: distance-first GT matching (exclude REMOVE nodes) ──
            # REMOVE nodes are false positives; including them in matching would
            # "steal" GT slots from genuinely active (ADD/UPDATE) nodes.
            non_rm_idxs   = [ei for ei in range(len(acc_labels))
                             if idx_to_id[ei] not in O_remove]
            acc_for_match = [(0, acc_labels[ei], acc_p7s[ei]) for ei in non_rm_idxs]
            gt_for_match  = [(lb, p7) for lb, p7, _, _ in gt_raw]
            raw_matches, unmatched_sub, unmatched_gt = self.match_nodes(
                acc_for_match, gt_for_match,
                max_dist=self.match_max_dist,
                label_mismatch_penalty=self.LABEL_MISMATCH_PENALTY)
            all_matches   = [(non_rm_idxs[ri], gi, dist) for ri, gi, dist in raw_matches]
            unmatched_enc = [non_rm_idxs[ri]   for ri in unmatched_sub]

            # ---- step 3: keep robot-side labels ─────────────────────────────
            # GT labels are supervision targets only.  Do not copy them into the
            # encoder labels; otherwise class embeddings would leak GT identity.

            # ---- step 4: per-node metadata (uses robot-side O_update) ───────
            N           = len(acc_labels)
            op_types    = np.zeros(N, dtype=np.int64)
            delta_xs    = np.zeros((N, 7), dtype=np.float32)
            confidences = np.array(acc_confs, dtype=np.float32)   # real values

            for nid in O_add:
                ei = id_to_idx[nid]
                op_types[ei] = 1
                delta_xs[ei] = 0.0

            for nid in O_update:
                ei = id_to_idx[nid]
                op_types[ei] = 2
                if nid in O_update_phys and nid in prev_p7:
                    # physical box change: delta_x in the same scaled-normalised
                    # coordinate system as encoder boxes.
                    d = (acc_p7s[ei] - prev_p7[nid]).astype(np.float32)
                    if self._enc_std is not None:
                        d = 3.0 * d / self._enc_std
                    delta_xs[ei] = d

            for nid in O_remove:
                ei = id_to_idx[nid]
                op_types[ei] = 3   # delta_x stays 0 (node is vanishing)

            # ---- step 5: update prev_p7 for next step ───────────────────────
            for nid, label, p7, conf in robot_raw:
                prev_p7[nid] = p7.copy()

            # ---- step 6: build dicts with robot labels and GT targets ───────
            enc_raw = self._build_dict(
                acc_labels, acc_p7s,
                stats_file=self.enc_normalized_stats)
            # Stable robot-side node id per row, aligned with acc_labels/enc_raw at
            # this point. Kept as a plain list (not torchified) so consumers that
            # need to track the same real-world object across steps (e.g. the
            # --slam visualiser) don't have to rely on `gi`, which is only a
            # position in this step's own flattened block and differs for the same
            # object between steps.
            enc_raw['node_id'] = list(idx_to_id)
            gt_labels  = [lb    for lb, _,   _,    _     in gt_raw]
            gt_rot3s   = [rot3  for _,  _,   rot3, _     in gt_raw]
            gt_models  = [model for _,  _,   _,    model in gt_raw]
            gt_raw_d   = self._build_dict(
                gt_labels,
                [p7 for _, p7, _, _ in gt_raw],
                stats_file=self.gt_normalized_stats)
            gt_raw_d['models'] = gt_models   # list[str] of model names per GT node

            # full Euler angles [yaw, pitch, roll] for rotation supervision
            if gt_rot3s:
                gt_raw_d['rot'] = torch.from_numpy(
                    np.stack(gt_rot3s).astype(np.float32))   # (G, 3)
            else:
                gt_raw_d['rot'] = torch.zeros(0, 3)

            # attach GT shape features (OpenShape, 768-dim) if library loaded
            # use per-instance model features (from scene graph 'model' field);
            # unresolved model names raise immediately to catch data/library drift.
            if self.shape_lib is not None:
                gt_raw_d['feats'] = self.shape_lib.gt_features_for_models(
                    gt_models, tele_labels=gt_labels)

            # ---- step 7: active matches (restrict to ADD / UPDATE) ──────────
            active_set     = {id_to_idx[nid] for nid in O_add | O_update}
            active_matches = [(ei, gi, dist) for ei, gi, dist in all_matches
                              if ei in active_set]

            # ---- torchify ----
            self._torchify(enc_raw)
            self._torchify(gt_raw_d)

            active_mask = torch.from_numpy((op_types == 1) | (op_types == 2))
            valid_mask  = torch.tensor(
                [self.classes_r[int(c)] not in ('floor', '_scene_')
                 for c in enc_raw['objs']], dtype=torch.bool)

            enc_raw['op_types']    = torch.from_numpy(op_types)
            enc_raw['delta_x']     = torch.from_numpy(delta_xs)
            enc_raw['confidence']  = torch.from_numpy(confidences)
            enc_raw['active_mask'] = active_mask
            enc_raw['valid_mask']  = valid_mask

            # ---- new / old bookkeeping ----
            new_enc_idx = sorted(id_to_idx[nid] for nid in O_add)
            old_enc_idx = [i for i in range(N) if i not in set(new_enc_idx)]

            steps.append({
                'encoder':        enc_raw,
                'gt':             gt_raw_d,
                # supervision: active matched pairs only
                'active_matches': active_matches,   # [(enc_idx, gt_idx)]
                # diagnostics
                'unmatched_enc':  unmatched_enc,    # E5 false positives
                'unmatched_gt':   unmatched_gt,     # E4 missed detections
                # index bookkeeping
                'new_idx':        torch.tensor(new_enc_idx, dtype=torch.long),
                'old_idx':        torch.tensor(old_enc_idx, dtype=torch.long),
            })

            # ---- purge REMOVE nodes from accumulated state ──────────────────
            # Must happen AFTER appending the step (so GCN sees them this step)
            # and BEFORE the next step (so they don't persist as OLD nodes).
            if O_remove:
                keep       = [ei for ei in range(len(acc_labels))
                              if idx_to_id[ei] not in O_remove]
                acc_labels = [acc_labels[i] for i in keep]
                acc_p7s    = [acc_p7s[i]    for i in keep]
                acc_confs  = [acc_confs[i]  for i in keep]
                idx_to_id  = [idx_to_id[i]  for i in keep]
                id_to_idx  = {nid: new_ei for new_ei, nid in enumerate(idx_to_id)}
                for nid in O_remove:
                    prev_p7.pop(nid, None)

        return {'scan_id': seq_info['seq'], 'steps': steps,
                'scene_center': scene_center.tolist()}

    # ------------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------------

    def collate_fn_inc(self, batch):
        """
        Flatten all steps from all sequences in the batch.

        Returned dict keys
        ------------------
        encoder
          objs, objs_g, boxes          concatenated over all steps × scenes
          op_types, delta_x            same
          confidence, active_mask      same
          valid_mask                   same
          obj_to_step, obj_to_scene    mapping tensors
          new_mask, old_mask           bool masks
          triples, triple_to_step, triple_to_scene

        gt
          objs, objs_g, boxes
          triples
          gt_to_step, gt_to_scene

        active_matches   list[list[(enc_global, gt_global, dist_m)]]
                         one entry per flattened step
        step_meta
          step_to_batch  (S,) which batch-item each step belongs to
          scan_id_str    (S,) sequence names
        """
        flat_steps     = []
        step_idx_vec   = []
        step_batch_idx = []
        for bidx, scene in enumerate(batch):
            for k, step in enumerate(scene['steps']):
                flat_steps.append(step)
                step_idx_vec.append(k)
                step_batch_idx.append(bidx)

        # encoder accumulators
        enc_objs, enc_objs_g, enc_boxes        = [], [], []
        enc_op, enc_dx, enc_conf               = [], [], []
        enc_act, enc_valid                     = [], []
        enc_triples                            = []
        enc_triple_to_step, enc_triple_to_scene = [], []
        enc_triple_to_flat                     = []
        obj_to_step, obj_to_scene              = [], []
        obj_to_flat_step                       = []
        new_mask, old_mask                     = [], []
        enc_feats                              = []
        enc_node_id                            = []   # list[int] concatenated, parallel to enc_objs

        # gt accumulators
        gt_objs, gt_objs_g, gt_boxes           = [], [], []
        gt_triples                             = []
        gt_to_step, gt_to_scene               = [], []
        gt_rots                                = []
        gt_feats                               = []
        gt_models_all                          = []   # list[str] concatenated

        active_matches_all = []   # per flattened step

        enc_off = 0
        gt_off  = 0

        for flat_sid, step in enumerate(flat_steps):
            k    = step_idx_vec[flat_sid]
            bidx = step_batch_idx[flat_sid]

            # --- encoder ---
            enc    = step['encoder']
            n_enc  = enc['objs'].size(0)

            enc_objs.append(enc['objs'])
            enc_objs_g.append(enc['objs_grained'])
            enc_boxes.append(enc['boxes'])
            enc_op.append(enc['op_types'])
            enc_dx.append(enc['delta_x'])
            enc_conf.append(enc['confidence'])
            enc_act.append(enc['active_mask'])
            enc_valid.append(enc['valid_mask'])
            enc_node_id.extend(enc['node_id'])

            obj_to_step.append(torch.full((n_enc,), k,        dtype=torch.long))
            obj_to_scene.append(torch.full((n_enc,), bidx,    dtype=torch.long))
            obj_to_flat_step.append(torch.full((n_enc,), flat_sid, dtype=torch.long))

            nm = torch.zeros(n_enc, dtype=torch.bool); nm[step['new_idx']] = True
            om = torch.zeros(n_enc, dtype=torch.bool); om[step['old_idx']] = True
            new_mask.append(nm); old_mask.append(om)

            t = enc['triples']
            if t.size(0) > 0:
                t = t.clone(); t[:, 0] += enc_off; t[:, 2] += enc_off
                enc_triples.append(t)
                enc_triple_to_step.append(torch.full((t.size(0),), k,        dtype=torch.long))
                enc_triple_to_scene.append(torch.full((t.size(0),), bidx,    dtype=torch.long))
                enc_triple_to_flat.append(torch.full((t.size(0),), flat_sid, dtype=torch.long))
            if enc.get('feats') is not None:
                enc_feats.append(enc['feats'])

            # --- gt ---
            gt   = step['gt']
            n_gt = gt['objs'].size(0)

            gt_objs.append(gt['objs'])
            gt_objs_g.append(gt['objs_grained'])
            gt_boxes.append(gt['boxes'])
            gt_to_step.append(torch.full((n_gt,), k,    dtype=torch.long))
            gt_to_scene.append(torch.full((n_gt,), bidx, dtype=torch.long))

            t = gt['triples']
            if t.size(0) > 0:
                t = t.clone(); t[:, 0] += gt_off; t[:, 2] += gt_off
                gt_triples.append(t)
            if gt.get('rot') is not None:
                gt_rots.append(gt['rot'])
            if gt.get('feats') is not None:
                gt_feats.append(gt['feats'])
            if gt.get('models') is not None:
                gt_models_all.extend(gt['models'])

            # --- active matches with global offsets ---
            active_matches_all.append(
                [(ei + enc_off, gi + gt_off, dist)
                 for ei, gi, dist in step['active_matches']])

            enc_off += n_enc
            gt_off  += n_gt

        def _cat(lst):
            return torch.cat(lst, 0) if lst else torch.empty(0, dtype=torch.long)

        def _cat3(lst):
            return torch.cat(lst, 0) if lst else torch.empty(0, 3, dtype=torch.long)

        out_enc = {
            'objs':            torch.cat(enc_objs,   0),
            'objs_g':          torch.cat(enc_objs_g, 0),
            'boxes':           torch.cat(enc_boxes,  0),
            'op_types':        torch.cat(enc_op,     0),
            'delta_x':         torch.cat(enc_dx,     0),
            'confidence':      torch.cat(enc_conf,   0),
            'active_mask':     torch.cat(enc_act,    0),
            'valid_mask':      torch.cat(enc_valid,  0),
            'obj_to_step':     _cat(obj_to_step),
            'obj_to_scene':    _cat(obj_to_scene),
            'new_mask':        torch.cat(new_mask,   0),
            'old_mask':        torch.cat(old_mask,   0),
            'triples':              _cat3(enc_triples),
            'triple_to_step':       _cat(enc_triple_to_step),
            'triple_to_scene':      _cat(enc_triple_to_scene),
            'obj_to_flat_step':     _cat(obj_to_flat_step),
            'triple_to_flat_step':  _cat(enc_triple_to_flat),
            'node_id':              enc_node_id,   # list[int], NOT a tensor
        }
        if enc_feats:
            out_enc['feats'] = torch.cat(enc_feats, 0)

        out_gt = {
            'objs':       torch.cat(gt_objs,   0),
            'objs_g':     torch.cat(gt_objs_g, 0),
            'boxes':      torch.cat(gt_boxes,  0),
            'triples':    _cat3(gt_triples),
            'gt_to_step': _cat(gt_to_step),
            'gt_to_scene':_cat(gt_to_scene),
        }
        if gt_rots:
            out_gt['rot'] = torch.cat(gt_rots, 0)    # (G_total, 3) [yaw,pitch,roll]
        if gt_feats:
            out_gt['feats'] = torch.cat(gt_feats, 0)
        if gt_models_all:
            out_gt['models'] = gt_models_all          # list[str] length G_total

        unique_bidx   = sorted(set(step_batch_idx))
        scene_centers = [batch[i].get('scene_center', [0.0, 0.0, 0.0]) for i in unique_bidx]

        return {
            'encoder':        out_enc,
            'gt':             out_gt,
            # active_matches[s] = [(enc_global, gt_global), …] for flattened step s
            'active_matches': active_matches_all,
            'step_meta': {
                'step_to_batch': torch.tensor(step_batch_idx, dtype=torch.long),
                'step_to_k':     torch.tensor(step_idx_vec,   dtype=torch.long),
                'scan_id_str':   [batch[i]['scan_id'] for i in step_batch_idx],
                # (B, 3) kept for compatibility; zero because boxes are not centred
                'scene_centers': torch.tensor(scene_centers, dtype=torch.float64),
            },
        }

    def __len__(self):
        return self.data_len if self.data_len is not None else len(self.sequences)


# ---------------------------------------------------------------------------
# Batch samplers (group sequences by step count to minimise padding)
# ---------------------------------------------------------------------------

class LengthGroupedCurriculumSampler(data.BatchSampler):
    """Short sequences first (curriculum), shuffle within each bucket."""

    def __init__(self, dataset, batch_size: int, drop_last: bool = False):
        self.batch_size = batch_size
        self.drop_last  = drop_last
        buckets = defaultdict(list)
        for idx in range(len(dataset)):
            buckets[dataset.n_steps[idx]].append(idx)   # O(1); no __getitem__ call
        self.sorted_buckets = sorted(buckets.items(), key=lambda x: x[0])

    def __iter__(self):
        for _, idx_list in self.sorted_buckets:
            random.shuffle(idx_list)
            for i in range(0, len(idx_list), self.batch_size):
                chunk = idx_list[i:i + self.batch_size]
                if len(chunk) == self.batch_size or not self.drop_last:
                    yield chunk

    def __len__(self):
        total = 0
        for _, idx_list in self.sorted_buckets:
            total += len(idx_list) // self.batch_size
            if len(idx_list) % self.batch_size and not self.drop_last:
                total += 1
        return total


class RandomStepGroupedSampler(data.BatchSampler):
    """Random order; each batch keeps the same step count."""

    def __init__(self, dataset, batch_size: int, drop_last: bool = False):
        self.batch_size = batch_size
        self.drop_last  = drop_last
        self.buckets = defaultdict(list)
        for idx in range(len(dataset)):
            self.buckets[dataset.n_steps[idx]].append(idx)   # O(1); no __getitem__ call
        for lst in self.buckets.values():
            random.shuffle(lst)

    def __iter__(self):
        remain = {k: v.copy() for k, v in self.buckets.items()}
        active = [k for k, v in remain.items() if v]
        while active:
            k      = random.choice(active)
            bucket = remain[k]
            if len(bucket) >= self.batch_size:
                yield [bucket.pop() for _ in range(self.batch_size)]
            else:
                if not self.drop_last:
                    yield [bucket.pop() for _ in range(len(bucket))]
                bucket.clear()
            active = [k for k, v in remain.items() if v]

    def __len__(self):
        total = 0
        for lst in self.buckets.values():
            total += len(lst) // self.batch_size
            if len(lst) % self.batch_size and not self.drop_last:
                total += 1
        return total
