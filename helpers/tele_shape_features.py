"""
Shape feature utilities for the Tele dataset.

The 3D model library stores OpenShape (768-dim, L2-normalised) features
for each .glb model.  This module provides:

  TeleShapeLibrary  – loads the library, exposes per-class average features
                      and nearest-neighbour retrieval at evaluation time.

Usage (training)
----------------
  lib = TeleShapeLibrary('/media/gao/gao_ssd/3D objects/features.pkl')
  gt_feat = lib.gt_features_for_models(['frl_apartment_chair_01'])

Usage (evaluation / retrieval)
-------------------------------
  glb_path, sim = lib.retrieve(pred_feat_768d)   # nearest neighbour
"""
import json
import os
import pickle
from difflib import SequenceMatcher
from collections import defaultdict

import numpy as np
import torch

# Tele dataset label → library label (handle naming mismatches)
_TELE_TO_LIB = {
    'door':             'doors',
    'kitchen_cupboard': 'kitchen_cupboards',
    'tvstand':          'tv_stand',
}
# Classes with no 3D model counterpart (background / structural)
_NO_MODEL = {'_scene_', 'floor'}


class TeleShapeLibrary:
    """
    Wraps the 3D model library for training supervision and eval retrieval.

    Parameters
    ----------
    features_pkl : str
        Path to features.pkl produced for the Tele 3D object library.
    feat_dim : int
        Expected feature dimension (default 768 for OpenShape).
    """

    def __init__(self, features_pkl: str, feat_dim: int = 768):
        with open(features_pkl, 'rb') as f:
            raw = pickle.load(f)   # {rel_path: {'openshape':(768,), 'glb_path':str, 'label':str}}

        self.feat_dim = feat_dim

        # Store all (feature, glb_path) pairs
        self._all_feats   = []   # list of np.ndarray (768,)
        self._all_paths   = []   # list of str
        self._all_labels  = []   # list of str

        by_label = defaultdict(list)
        self._label_to_indices = defaultdict(list)
        for _, v in raw.items():
            feat = v['openshape'].astype(np.float32)
            feat = feat / (np.linalg.norm(feat) + 1e-8)
            lb   = v['label']
            self._all_feats.append(feat)
            self._all_paths.append(v['glb_path'])
            self._all_labels.append(lb)
            self._label_to_indices[lb].append(len(self._all_feats) - 1)
            by_label[lb].append(feat)

        # Per-class average feature (L2-normalised)
        self._class_feat = {}
        for lb, vecs in by_label.items():
            avg = np.stack(vecs).mean(axis=0)
            avg = avg / (np.linalg.norm(avg) + 1e-8)
            self._class_feat[lb] = torch.from_numpy(avg)   # (768,)

        # Fast model-name lookups used for per-instance GT supervision.
        self._name_to_idx = {}
        self._norm_name_to_idx = {}
        for i, p in enumerate(self._all_paths):
            base = os.path.splitext(os.path.basename(p))[0]
            self._name_to_idx[base] = i
            self._norm_name_to_idx[self._model_key(base)] = i

        self._label_norm_to_label = {
            self._model_key(lb): lb for lb in self._class_feat
        }

        # Stack all features for fast NN retrieval
        self._feat_matrix = torch.from_numpy(
            np.stack(self._all_feats))   # (N_models, 768)

        # Load [L, H, W] sizes from companion model_sizes.json if available.
        # None entries (compound URDF objects) indicate fall-back to enc size.
        sizes_json = os.path.join(os.path.dirname(features_pkl), 'model_sizes.json')
        self._all_sizes = self._load_sizes_from_json(sizes_json, self._all_paths)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _model_key(s):
        """Normalise model/category names for robust lookup."""
        base = os.path.splitext(os.path.basename(str(s)))[0]
        return ''.join(ch.lower() for ch in base if ch.isalnum())

    def _infer_label_from_model_name(self, model_name):
        """Best-effort class inference from a model filename."""
        key = self._model_key(model_name)
        if not key:
            return None

        aliases = {
            'chestofdrawers': 'chest_of_drawers',
            'kitchencupboard': 'kitchen_cupboards',
            'kitchencupboards': 'kitchen_cupboards',
            'tvstand': 'tv_stand',
            'door': 'doors',
        }
        for alias, label in aliases.items():
            if alias in key and label in self._class_feat:
                return label

        for label_key, label in self._label_norm_to_label.items():
            singular = label_key[:-1] if label_key.endswith('s') else label_key
            if label_key in key or (singular and singular in key):
                return label
        return None

    def _candidate_indices(self, tele_label=None, model_name=''):
        """Return label-filtered candidate indices when possible."""
        label = self.map_label(tele_label) if tele_label else None
        if label is None:
            label = self._infer_label_from_model_name(model_name)
        if label in self._label_to_indices:
            return self._label_to_indices[label]
        return list(range(len(self._all_paths)))

    def _find_model_idx(self, model_name, tele_label=None):
        """Resolve a GT model name to a feature-library row."""
        idx = self._name_to_idx.get(model_name)
        if idx is not None:
            return idx

        key = self._model_key(model_name)
        idx = self._norm_name_to_idx.get(key)
        if idx is not None:
            return idx

        candidates = self._candidate_indices(tele_label, model_name)
        if not key:
            return None

        cand_keys = [(i, self._model_key(self._all_paths[i])) for i in candidates]

        contains = [(i, ck) for i, ck in cand_keys if key in ck or ck in key]
        if len(contains) == 1:
            return contains[0][0]
        if len(contains) > 1:
            # A bare class name such as "chair" should not arbitrarily select
            # one of several chairs; mark it unresolved.
            exact_len = [i for i, ck in contains if len(key) == len(ck)]
            return exact_len[0] if len(exact_len) == 1 else None

        scored = sorted(
            ((SequenceMatcher(None, key, ck).ratio(), i) for i, ck in cand_keys),
            reverse=True)
        if not scored:
            return None
        best_score, best_idx = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= 0.72 and (best_score - second_score >= 0.08 or len(candidates) == 1):
            return best_idx
        return None

    @staticmethod
    def _load_sizes_from_json(sizes_json, paths):
        """Load [L, H, W] sizes from a pre-computed JSON file.
        Returns a list where each entry is [L, H, W] or None (fall back to enc).
        None means the model is compound/URDF and has no reliable single-mesh size.
        """
        if not os.path.exists(sizes_json):
            return [None] * len(paths)
        with open(sizes_json) as f:
            size_map = json.load(f)
        result = []
        for gp in paths:
            v = size_map.get(gp)
            if v is not None and min(v) > 0.01:
                result.append(v)       # valid [L, H, W] in metres
            else:
                result.append(None)    # invalid / compound object
        return result

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def map_label(self, tele_label: str):
        """Map Tele dataset label to library label (None if no model)."""
        if tele_label in _NO_MODEL:
            return None
        return _TELE_TO_LIB.get(tele_label, tele_label)

    def class_feature(self, tele_label: str):
        """
        Return the (768,) L2-normalised class-average feature tensor.
        Returns None for background classes or unknown labels.
        """
        lib_lb = self.map_label(tele_label)
        if lib_lb is None:
            return None
        return self._class_feat.get(lib_lb)

    def gt_features_for_batch(self, gt_labels_tensor, device='cpu'):
        """
        Build GT feature matrix for a batch of GT nodes.

        gt_labels_tensor : (G,) int64 – class indices
        label_idx_to_name : dict[int, str]

        Returns (G, 768) float tensor; rows are 0 for unknown classes.
        """
        raise NotImplementedError('call gt_features_for_objs instead')

    def gt_features_for_objs(self, tele_labels, device='cpu'):
        """
        tele_labels : list[str]  (GT class names, already Tele-normalised)
        Returns (N, 768) float tensor; rows for unknown classes are zero.
        """
        out = torch.zeros(len(tele_labels), self.feat_dim)
        for i, lb in enumerate(tele_labels):
            f = self.class_feature(lb)
            if f is not None:
                out[i] = f
        return out.to(device)

    def gt_features_for_models(self, model_names, tele_labels=None, device='cpu'):
        """
        Look up the per-instance OpenShape feature for each GT object by its
        exact model name (the 'model' field from the GT scene graph JSON).

        model_names : list[str]  e.g. ['frl_apartment_chair_01', 'frl_apartment_sofa', ...]
        tele_labels : optional list[str] of GT class names, used only to narrow
                      fuzzy matching to same-class library candidates.
        Returns (N, 768) float tensor.
        Raises KeyError if a GT model cannot be resolved to an instance feature.
        """
        out = torch.zeros(len(model_names), self.feat_dim)
        for i, name in enumerate(model_names):
            label = tele_labels[i] if tele_labels is not None and i < len(tele_labels) else None
            idx = self._find_model_idx(name, tele_label=label)
            if idx is not None:
                feat = torch.from_numpy(self._all_feats[idx])
                out[i] = feat
                continue

            candidates = self._candidate_indices(label, name)
            candidate_names = [
                os.path.splitext(os.path.basename(self._all_paths[j]))[0]
                for j in candidates[:8]
            ]
            raise KeyError(
                f'No OpenShape feature found for GT model "{name}"'
                f' (label="{label}"). Candidate models: {candidate_names}')
        return out.to(device)

    # ------------------------------------------------------------------
    # Evaluation / retrieval
    # ------------------------------------------------------------------

    def retrieve(self, pred_feat, top_k=1, label_filter=None):
        """
        Retrieve nearest library model(s) for a predicted feature vector.

        pred_feat    : (768,) tensor or ndarray – will be L2-normalised
        top_k        : how many results to return
        label_filter : str or None – restrict search to this library label

        Returns
        -------
        results : list of (glb_path, cosine_similarity) tuples
        """
        if isinstance(pred_feat, np.ndarray):
            pred_feat = torch.from_numpy(pred_feat.astype(np.float32))
        pred_feat = pred_feat.float()
        pred_feat = pred_feat / (pred_feat.norm() + 1e-8)

        matrix = self._feat_matrix.to(pred_feat.device)
        if label_filter is not None:
            mask = [i for i, lb in enumerate(self._all_labels)
                    if lb == label_filter]
            if not mask:
                label_filter = None
            else:
                matrix = matrix[mask]

        sims = (matrix @ pred_feat)   # (N,)
        top_k = min(top_k, sims.size(0))
        vals, idxs = sims.topk(top_k)

        if label_filter is not None:
            orig_idxs = [mask[i] for i in idxs.tolist()]
        else:
            orig_idxs = idxs.tolist()

        return [(self._all_paths[i], float(vals[j]))
                for j, i in enumerate(orig_idxs)]

    def retrieve_sizes_batch(self, pred_feats, tele_labels=None, fallback=None):
        """
        For each predicted feature, retrieve the nearest library model and
        return its [L, H, W] bounding-box size as a (N, 3) float tensor.

        pred_feats  : (N, feat_dim) tensor
        tele_labels : list[str] of length N – restricts NN search to that class
        fallback    : (N, 3) tensor – used when the retrieved model has no valid
                      size (compound URDF objects); defaults to ones if None
        """
        pred_feats = pred_feats.float()
        norms = pred_feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        pred_feats = pred_feats / norms
        N = pred_feats.size(0)

        if fallback is None:
            fallback = torch.ones(N, 3)
        fallback = fallback.float()

        out = torch.zeros(N, 3)
        matrix = self._feat_matrix   # (N_models, feat_dim)

        for i in range(N):
            lb = self.map_label(tele_labels[i]) if tele_labels is not None else None

            if lb is not None:
                mask = [j for j, l in enumerate(self._all_labels) if l == lb]
            else:
                mask = None

            if mask:
                sims = matrix[mask] @ pred_feats[i]
                best = mask[int(sims.argmax())]
            else:
                sims = matrix @ pred_feats[i]
                best = int(sims.argmax())

            sz = self._all_sizes[best]
            if sz is not None:
                out[i] = torch.tensor(sz, dtype=torch.float32)
            else:
                out[i] = fallback[i]   # compound/URDF: fall back to enc size

        return out   # (N, 3)  [L, H, W] in metres

    def retrieve_batch(self, pred_feats, tele_labels=None):
        """
        Retrieve best match for each row of pred_feats.

        pred_feats  : (N, 768) tensor
        tele_labels : list[str] of length N (optional, restricts per-row search)

        Returns list[str] of glb_paths.
        """
        pred_feats = pred_feats.float()
        norms = pred_feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        pred_feats = pred_feats / norms

        paths = []
        for i in range(pred_feats.size(0)):
            lb = None
            if tele_labels is not None:
                lb = self.map_label(tele_labels[i])
            results = self.retrieve(pred_feats[i], top_k=1,
                                    label_filter=lb)
            paths.append(results[0][0] if results else '')
        return paths
