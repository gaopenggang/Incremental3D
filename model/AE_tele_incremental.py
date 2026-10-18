import math as _math
import torch
import torch.nn as nn
from model.graph import GraphTripleConvNet, _init_weights, make_mlp

# Yaw discretisation: 8 bins at 45° intervals, ordered 0→315° then wrapped to (-π,π].
N_YAW_BINS = 8
YAW_BIN_RAD = torch.tensor(
    [0., _math.pi/4, _math.pi/2, 3*_math.pi/4,
     _math.pi, -3*_math.pi/4, -_math.pi/2, -_math.pi/4],
    dtype=torch.float32)   # [0°, 45°, 90°, 135°, ±180°, -135°, -90°, -45°]


def _wrap_angle(x):
    """Wrap radians to (-pi, pi]."""
    return torch.atan2(torch.sin(x), torch.cos(x))


class Sg2ScAEModelTeleIncremental(nn.Module):
    """
    Incremental scene graph refinement model for the Tele robot dataset.

    Each forward() call processes one batch of sequences.  Within a sequence
    steps are processed in temporal order (k=0 → k=1 → …) so that the CLS
    state captured at step k is available as context for step k+1.

    Input  (encoder side)  : accumulated robot SG with per-node metadata
                             op_types ∈ {0=OLD, 1=ADD, 2=UPDATE}
    Output (active nodes)  : pred_pos  (M,3)  – corrected centroid
                             pred_rot  (M,1)  – corrected yaw angle
                             pred_feat (M,F)  – shape feature

    Only ADD and UPDATE nodes (active_mask==True) receive predictions.
    Loss targets are provided by batch['gt']['boxes'] indexed via
    batch['active_matches'][flat_sid] = [(enc_global, gt_global, dist_m), …].
    """

    GLOBAL_PID = 0   # predicate id used for CLS ↔ node edges

    def __init__(self,
                 vocab,
                 node_embedding_dim: int = 128,
                 gconv_pooling: str = 'avg',
                 gconv_num_layers: int = 5,
                 mlp_normalization: str = 'none',
                 shape_feat_dim: int = 768,
                 residual: bool = True,
                 rot_out_dim: int = N_YAW_BINS):
        super().__init__()
        if rot_out_dim != N_YAW_BINS:
            raise ValueError(
                f'Tele incremental yaw head is fixed to {N_YAW_BINS} bins; '
                f'got rot_out_dim={rot_out_dim}. Old rotation-regression '
                'checkpoints are not compatible with this model.')
        D = node_embedding_dim
        self.D = D
        self.shape_feat_dim = shape_feat_dim

        n_obj = len(set(vocab['object_idx_to_name'])) + 1
        n_rel = len(set(vocab['pred_idx_to_name'])) + 1
        self.n_obj = n_obj

        # ── node input projections ──────────────────────────────────────────
        self.emb_obj    = nn.Embedding(n_obj, D)   # semantic class
        self.emb_box    = nn.Linear(6, D)           # noisy robot size + position
        self.emb_op     = nn.Embedding(4, D)        # op type (0=OLD/1=ADD/2=UPDATE/3=REMOVE)
        self.emb_delta  = nn.Linear(6, D)           # δx size + position (zero for OLD/ADD)
        self.emb_conf   = nn.Linear(1, D)           # confidence scalar
        self.node_proj  = nn.Linear(5 * D, 2 * D)  # fuse → GCN width

        # ── CLS token (global scene state, persists across steps) ───────────
        self.cls_tok   = nn.Parameter(torch.randn(1, 2 * D))

        # ── GCN encoder ─────────────────────────────────────────────────────
        self.emb_pred  = nn.Embedding(n_rel, 2 * D)
        self.gcn = GraphTripleConvNet(
            input_dim_obj     = 2 * D,
            input_dim_pred    = 2 * D,
            hidden_dim        = 4 * D,
            pooling           = gconv_pooling,
            num_layers        = gconv_num_layers,
            mlp_normalization = mlp_normalization,
            residual          = residual,
        )

        # ── LayerNorm before prediction heads ────────────────────────────────
        # Prevents GCN residual-accumulation from blowing up active features.
        H = 4 * D
        self.active_norm = nn.LayerNorm(H)

        # ── CLS-state normalisation ───────────────────────────────────────────
        # The CLS token flows through up to 60 GCN steps (RNN-like recurrence).
        # Without normalisation it can grow exponentially, causing NaN for long
        # sequences.  LayerNorm on the CLS output before storing prevents this.
        self.cls_norm = nn.LayerNorm(2 * D)

        # ── prediction heads (input: active_node_out(2D) ‖ CLS(2D) = 4D) ──
        self.mlp_pos       = make_mlp([H, 2*D, 3],          mlp_normalization, norelu=True)
        self.rot_out_dim   = N_YAW_BINS
        # Coarse: N_YAW_BINS-class classification of yaw-delta bin
        self.mlp_rot       = make_mlp([H, 2*D, N_YAW_BINS], mlp_normalization, norelu=True)
        # Fine: per-bin intra-bin residual in (-π/N, +π/N); only the predicted bin is used
        self.mlp_rot_delta = make_mlp([H, 2*D, N_YAW_BINS], mlp_normalization, norelu=True)
        self.mlp_feat      = make_mlp([H, 4*D, 3*D, shape_feat_dim], mlp_normalization, norelu=True)
        # Label correction: predict the true object class using GT supervision on matched pairs
        self.mlp_cls       = make_mlp([H, 2*D, n_obj],      mlp_normalization, norelu=True)

        # ── per-sequence CLS state (reset at the start of each forward) ─────
        self._scene_states: dict = {}

        self.apply(_init_weights)

        # Zero-init pos/rot heads: initial predictions start near 0.
        for mlp in (self.mlp_pos, self.mlp_rot, self.mlp_rot_delta):
            last = [m for m in mlp.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

        # Scale mlp_feat last layer so initial pred_feat norm ≈ 1.
        # Kaiming init across 3 ReLU layers gives norm ≈ 2^1.5 * sqrt(shape_feat_dim) ≈ 78
        # for shape_feat_dim=768.  Dividing by that factor aligns pred_feat with the
        # unit-normed OpenShape GT features and avoids normalize() instability.
        feat_last = [m for m in self.mlp_feat.modules() if isinstance(m, nn.Linear)][-1]
        expected_norm = (2 ** 1.5) * (shape_feat_dim ** 0.5)
        feat_last.weight.data /= expected_norm
        feat_last.bias.data   /= expected_norm

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _node_feat(self, objs, boxes, op_types, delta_x, confidence):
        """Build 2D node feature vectors from raw encoder fields."""
        h = torch.cat([
            self.emb_obj(objs),
            self.emb_box(boxes[:, :6]),
            self.emb_op(op_types),
            self.emb_delta(delta_x[:, :6]),
            self.emb_conf(confidence.unsqueeze(-1)),
        ], dim=-1)                   # (N, 5D)
        return self.node_proj(h)     # (N, 2D)

    def _gcn_step(self, scan_id: str, node_feats, triples):
        """
        One GCN forward for a single (sequence, step) graph.

        Prepends the CLS token (loaded from _scene_states or initialised from
        cls_tok), adds bidirectional CLS ↔ node edges, runs the GCN, and
        returns the updated CLS vector and the per-node output vectors.

        triples : (T, 3) local-indexed [s, pred_id, o], or empty
        """
        device = node_feats.device
        N = node_feats.size(0)

        cls = self._scene_states.get(scan_id, self.cls_tok.to(device))  # (1, 2D)

        # Build mega-node tensor: [CLS | nodes]
        all_nodes = torch.cat([cls, node_feats], 0)   # (1+N, 2D)

        # CLS ↔ every node (bidirectional)
        node_idx = torch.arange(1, 1 + N, device=device)
        cls_idx  = torch.zeros(N, dtype=torch.long, device=device)
        e_cls = torch.cat([
            torch.stack([cls_idx, node_idx], 1),   # CLS → node
            torch.stack([node_idx, cls_idx], 1),   # node → CLS
        ], 0)                                       # (2N, 2)
        p_cls = torch.full((2 * N,), self.GLOBAL_PID, dtype=torch.long, device=device)

        # Object–object triples (shift +1 for CLS offset)
        if triples.numel() > 0:
            t = triples.clone()
            t[:, 0] += 1; t[:, 2] += 1
            edges = torch.cat([e_cls, t[:, [0, 2]]], 0)
            pids  = torch.cat([p_cls, t[:, 1]], 0)
        else:
            edges, pids = e_cls, p_cls

        pred_emb = self.emb_pred(pids)                      # (E, 2D)
        node_out, _ = self.gcn(all_nodes, pred_emb, edges)  # (1+N, 2D)

        cls_out = self.cls_norm(node_out[0:1])   # normalise to prevent recurrent explosion
        return cls_out, node_out[1:]            # cls_out (1,2D), node_out (N,2D)

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, batch, reset_states: bool = True):
        """
        Parameters
        ----------
        batch : dict produced by TeleDatasetSceneGraphIncremental.collate_fn_inc
        reset_states : clear per-sequence CLS cache before processing
                       (True for training, False for live inference)

        Returns
        -------
        dict with keys:
          pred_pos  (N_total, 3)   – predicted centroids (active nodes only)
          pred_rot  (N_total, 1)   – predicted yaw angle
          pred_feat (N_total, F)   – predicted shape features
        Non-active entries are zero.  Use batch['active_matches'] to index them.
        """
        enc       = batch['encoder']
        meta      = batch['step_meta']
        scan_ids  = meta['scan_id_str']       # list[str], length S
        step_to_k = meta['step_to_k']         # (S,) k-within-sequence
        device    = enc['objs'].device
        N_total   = enc['objs'].size(0)
        S         = len(scan_ids)

        if reset_states:
            for sid in set(scan_ids):
                self._scene_states.pop(sid, None)

        # obj_to_flat_step: (N,) direct mapping, added by collate_fn_inc
        obj_to_flat  = enc['obj_to_flat_step'].to(device)
        triple_to_flat = enc.get('triple_to_flat_step')   # (T,) or None

        # Output buffers – filled in-place for active nodes
        pred_pos        = torch.zeros(N_total, 3,                   device=device)
        pred_rot        = torch.zeros(N_total, 1,                   device=device)
        pred_yaw_logits = torch.zeros(N_total, N_YAW_BINS,          device=device)
        pred_yaw_deltas = torch.zeros(N_total, N_YAW_BINS,          device=device)
        pred_feat       = torch.zeros(N_total, self.shape_feat_dim,  device=device)
        pred_cls_logits = torch.zeros(N_total, self.n_obj,           device=device)

        # Process steps in ascending k order so CLS state propagates correctly.
        # At each k, all sequences that have a step k are processed (in parallel
        # within the same Python loop iteration).
        K_max = int(step_to_k.max()) + 1
        for k in range(K_max):
            for flat_sid in range(S):
                if int(step_to_k[flat_sid]) != k:
                    continue
                scan_id  = scan_ids[flat_sid]

                # ── select objects for this (sequence, step) ──────────────
                obj_mask = (obj_to_flat == flat_sid)          # (N_total,) bool
                if not obj_mask.any():
                    continue
                obj_idx = obj_mask.nonzero(as_tuple=True)[0]  # (M,) global indices

                # ── node features ─────────────────────────────────────────
                node_feats = self._node_feat(
                    enc['objs'][obj_mask],
                    enc['boxes'][obj_mask],
                    enc['op_types'][obj_mask],
                    enc['delta_x'][obj_mask],
                    enc['confidence'][obj_mask],
                )  # (M, 2D)

                # ── triples for this step (remap global → local) ──────────
                triples = self._local_triples(
                    enc['triples'], triple_to_flat, flat_sid, obj_idx, N_total, device)

                # ── GCN forward ───────────────────────────────────────────
                cls_out, node_out = self._gcn_step(scan_id, node_feats, triples)
                self._scene_states[scan_id] = cls_out.detach()

                # ── prediction heads on active nodes ─────────────────────
                active_local = enc['active_mask'][obj_mask]   # (M,) bool
                if active_local.any():
                    A = int(active_local.sum())
                    h_act = torch.cat([
                        node_out[active_local],        # (A, 2D)
                        cls_out.expand(A, -1),         # (A, 2D)
                    ], dim=-1)                         # (A, 4D)
                    h_act = self.active_norm(h_act)    # stabilise GCN variance

                    active_global = obj_idx[active_local]
                    enc_pos_n  = enc['boxes'][obj_mask][active_local, 3:6]   # (A,3) enc-norm
                    pred_pos[active_global]     = self.mlp_pos(h_act) + enc_pos_n
                    yaw_logits = self.mlp_rot(h_act)                         # (A, N_YAW_BINS)
                    yaw_deltas = self.mlp_rot_delta(h_act)                   # (A, N_YAW_BINS)
                    bins = YAW_BIN_RAD.to(device)
                    yaw_init   = torch.zeros(A, device=device)               # predict absolute yaw (no prior)
                    pred_bin   = yaw_logits.argmax(dim=-1)                   # (A,) coarse bin
                    pred_delta = yaw_deltas.gather(
                        1, pred_bin.unsqueeze(1)).squeeze(1)                 # (A,) intra-bin delta
                    pred_yaw   = _wrap_angle(
                        yaw_init + bins[pred_bin] + pred_delta)              # (A,) fine yaw
                    pred_rot[active_global, 0]     = pred_yaw
                    pred_yaw_logits[active_global] = yaw_logits
                    pred_yaw_deltas[active_global] = yaw_deltas
                    pred_feat[active_global]       = self.mlp_feat(h_act)
                    pred_cls_logits[active_global] = self.mlp_cls(h_act)

        return {
            'pred_pos':        pred_pos,          # (N, 3)  predicted centroid
            'pred_rot':        pred_rot,          # (N, 1)  final yaw (rad)
            'pred_yaw_logits': pred_yaw_logits,   # (N, 8)  coarse bin logits
            'pred_yaw_deltas': pred_yaw_deltas,   # (N, 8)  per-bin intra-bin delta (rad)
            'pred_feat':       pred_feat,         # (N, F)
            'pred_cls_logits': pred_cls_logits,   # (N, n_obj)  label correction logits
        }

    @staticmethod
    def _local_triples(all_triples, triple_to_flat, flat_sid, obj_idx, N_total, device):
        """Filter and remap global triples to local indices for one step."""
        if all_triples.numel() == 0:
            return torch.empty(0, 3, dtype=torch.long, device=device)

        if triple_to_flat is not None:
            t_mask = (triple_to_flat.to(device) == flat_sid)
            if not t_mask.any():
                return torch.empty(0, 3, dtype=torch.long, device=device)
            t_global = all_triples[t_mask]
        else:
            t_global = all_triples

        # Build global → local lookup
        g2l = torch.full((N_total,), -1, dtype=torch.long, device=device)
        g2l[obj_idx] = torch.arange(len(obj_idx), device=device)

        t_local = t_global.clone()
        t_local[:, 0] = g2l[t_global[:, 0]]
        t_local[:, 2] = g2l[t_global[:, 2]]
        valid = (t_local[:, 0] >= 0) & (t_local[:, 2] >= 0)
        return t_local[valid] if valid.any() else torch.empty(0, 3, dtype=torch.long, device=device)

    def reset_scene_states(self):
        self._scene_states.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Loss utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_active_loss(preds, batch,
                        feat_weight:    float = 0.5,
                        delta_weight:   float = 1.0,
                        cls_weight:     float = 1.0,
                        cls_dist_thresh: float = 0.75,
                        enc_size_mu=None, enc_size_sg=None,
                        gt_size_mu=None,  gt_size_sg=None):
    """
    Compute position + rotation + feature loss over active matched nodes.

    Denominators
    ------------
    position / feature : n_active_pairs  (all matched pairs)
    rotation           : n_rot_pairs     (excludes yaw_w=0 classes, e.g. indoor_plant)

    Rotation (coarse + fine)
    ------------------------
    Coarse: symmetry-aware cross-entropy over N_YAW_BINS bins.
    Fine:   L1 on intra-bin residual delta for the GT bin
            (delta_weight scales the fine term relative to CE).
    """
    import torch.nn.functional as F

    pred_pos        = preds['pred_pos']         # (N, 3)
    pred_feat       = preds['pred_feat']        # (N, F)
    pred_yaw_deltas = preds['pred_yaw_deltas']  # (N, 8) per-bin intra-bin delta
    pred_cls_logits = preds['pred_cls_logits']  # (N, n_obj)
    gt_boxes        = batch['gt']['boxes']       # (G, 7)
    gt_rot          = batch['gt'].get('rot')     # (G, 3) or None
    gt_objs         = batch['gt']['objs']        # (G,) int64 GT class ids

    loss_pos  = pred_pos.new_tensor(0.)
    loss_rot  = pred_pos.new_tensor(0.)
    loss_feat = pred_pos.new_tensor(0.)
    loss_cls  = pred_pos.new_tensor(0.)
    n_active_pairs = 0   # denominator for pos + feat
    n_rot_pairs    = 0   # denominator for rot (excludes weight-0 pairs)
    n_cls_pairs    = 0   # denominator for cls (only dist <= cls_dist_thresh)
    dev = pred_pos.device

    for matches in batch['active_matches']:
        if not matches:
            continue
        enc_idx     = torch.tensor([m[0] for m in matches], dtype=torch.long,  device=dev)
        gt_idx      = torch.tensor([m[1] for m in matches], dtype=torch.long,  device=dev)
        match_dists = torch.tensor([m[2] for m in matches], dtype=torch.float, device=dev)
        M = len(matches)

        # ── position ──────────────────────────────────────────────────────────
        gt_pos  = gt_boxes[gt_idx, 3:6]
        pos_err = F.l1_loss(pred_pos[enc_idx], gt_pos, reduction='none').mean(dim=1)
        loss_pos       += pos_err.sum()
        n_active_pairs += M

        # ── rotation setup ────────────────────────────────────────────────────
        gt_yaw = (gt_rot[gt_idx, 0].to(dev) if gt_rot is not None
                  else gt_boxes[gt_idx, 6].to(dev))

        obj_ids = batch['encoder']['objs'][enc_idx]   # (M,)
        yaw_w   = torch.ones(M, device=dev)
        yaw_w[obj_ids == 10] = 0.0   # indoor_plant (10): fully symmetric, GT yaw undefined
        yaw_w[obj_ids ==  1] = 0.0   # beanbag (1): near-circular, GT yaw has no semantic meaning

        yaw_init     = torch.zeros(M, device=dev)               # predict absolute yaw (no prior)
        gt_yaw_delta = _wrap_angle(gt_yaw - yaw_init)          # residual from zero = absolute yaw
        bins  = YAW_BIN_RAD.to(dev)                            # (8,)
        diffs = _wrap_angle(gt_yaw_delta.unsqueeze(1)
                            - bins.unsqueeze(0)).abs()          # (M, 8) angular distance

        # best_bin: the geometrically closest bin to gt_yaw_delta, accounting for
        # rotational symmetry.  Used as the anchor for the delta regression target.
        gt_bin   = diffs.argmin(dim=1)        # (M,) ignoring symmetry
        best_bin = gt_bin.clone()

        mask2 = (obj_ids == 14) | (obj_ids == 17)   # table(14), wall_cabinet(17): 2-fold (0° ≡ 180°)
        if mask2.any():
            gb2, gb2_alt = gt_bin[mask2], (gt_bin[mask2] + 4) % 8
            pick_alt = (diffs[mask2].gather(1, gb2_alt.unsqueeze(1)).squeeze(1) <
                        diffs[mask2].gather(1, gb2.unsqueeze(1)).squeeze(1))
            best_bin[mask2] = torch.where(pick_alt, gb2_alt, gb2)

        mask4 = (obj_ids == 1)    # beanbag: 4-fold (90° rotational symmetry)
        if mask4.any():
            opts      = torch.stack([(gt_bin[mask4] + k) % 8 for k in [0, 2, 4, 6]], 1)
            opt_dists = diffs[mask4].gather(1, opts)
            best_bin[mask4] = opts.gather(
                1, opt_dists.argmin(dim=1, keepdim=True)).squeeze(1)

        # ── coarse CE (symmetry-aware min-of-K over equivalent bins) ─────────
        yaw_logits = preds['pred_yaw_logits'][enc_idx]    # (M, 8)
        ce = F.cross_entropy(yaw_logits, best_bin, reduction='none')   # (M,)

        if mask2.any():
            gb2 = best_bin[mask2]
            ce[mask2] = torch.stack([
                F.cross_entropy(yaw_logits[mask2], gb2,       reduction='none'),
                F.cross_entropy(yaw_logits[mask2], (gb2+4)%8, reduction='none'),
            ], dim=1).min(dim=1).values

        if mask4.any():
            gb4 = best_bin[mask4]
            ce[mask4] = torch.stack([
                F.cross_entropy(yaw_logits[mask4], gb4,       reduction='none'),
                F.cross_entropy(yaw_logits[mask4], (gb4+2)%8, reduction='none'),
                F.cross_entropy(yaw_logits[mask4], (gb4+4)%8, reduction='none'),
                F.cross_entropy(yaw_logits[mask4], (gb4+6)%8, reduction='none'),
            ], dim=1).min(dim=1).values

        # ── fine delta L1 ─────────────────────────────────────────────────────
        # GT: small intra-bin residual in (-π/N_BINS, +π/N_BINS).
        # Supervise the GT bin's predicted delta regardless of which bin is predicted
        # (standard coarse-to-fine: both heads learn from every sample).
        gt_delta         = _wrap_angle(gt_yaw_delta - bins[best_bin])   # (M,)
        pred_delta_at_gt = pred_yaw_deltas[enc_idx].gather(
            1, best_bin.unsqueeze(1)).squeeze(1)                         # (M,)
        l1_delta = F.l1_loss(pred_delta_at_gt, gt_delta, reduction='none')  # (M,)

        loss_rot    += ((ce + delta_weight * l1_delta) * yaw_w).sum()
        n_rot_pairs += int(yaw_w.gt(0).sum().item())

        # ── shape feature ─────────────────────────────────────────────────────
        if 'feats' in batch['gt']:
            gt_feat = batch['gt']['feats'][gt_idx].to(dev)
            p   = F.normalize(pred_feat[enc_idx], dim=-1)
            sim = (p * gt_feat).sum(dim=-1)
            loss_feat += (1.0 - sim).sum()

        # ── label correction: CE only for well-matched pairs (dist ≤ threshold) ─
        # Filters out false-positive nodes forced onto distant GT slots by Hungarian,
        # which would give harmful supervision (e.g. beanbag trained as wall_cabinet).
        cls_mask = match_dists <= cls_dist_thresh
        if cls_mask.any():
            loss_cls += F.cross_entropy(
                pred_cls_logits[enc_idx[cls_mask]],
                gt_objs[gt_idx[cls_mask]].to(dev),
                reduction='sum',
            )
            n_cls_pairs += int(cls_mask.sum().item())

    if n_active_pairs > 0:
        loss_pos  /= n_active_pairs
        loss_feat /= n_active_pairs
    if n_rot_pairs > 0:
        loss_rot /= n_rot_pairs
    if n_cls_pairs > 0:
        loss_cls /= n_cls_pairs

    loss = loss_pos + loss_rot + feat_weight * loss_feat + cls_weight * loss_cls
    return loss, {'pos': loss_pos, 'rot': loss_rot, 'feat': loss_feat, 'cls': loss_cls}
