"""
Analyse active-match quality in TeleDatasetSceneGraphIncremental.

For every active (enc, gt) match pair across ALL sequences and steps, record:
  - label agreement (enc_label == gt_label)
  - L2 position distance between matched enc centroid and GT centroid
  - per-class breakdown

Run:
    python scripts/analyse_match_quality.py [--split val] [--workers 0]
"""
import sys, os, argparse, json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader

from dataset.Tele_dataset_incremental import (
    TeleDatasetSceneGraphIncremental, LengthGroupedCurriculumSampler)

parser = argparse.ArgumentParser()
parser.add_argument('--robot_root', default='/media/gao/gao_ssd/3D scene graph')
parser.add_argument('--gt_root',    default='/media/gao/gao_ssd/3D Scene')
parser.add_argument('--split',      default='val')
parser.add_argument('--workers',    type=int, default=0)
args = parser.parse_args()

dataset = TeleDatasetSceneGraphIncremental(
    robot_sg_root = args.robot_root,
    gt_scene_root = args.gt_root,
    split         = args.split,
    eval          = True,
    seed          = True,
)
loader = DataLoader(
    dataset,
    batch_sampler = LengthGroupedCurriculumSampler(dataset, batch_size=1),
    collate_fn    = dataset.collate_fn_inc,
    num_workers   = args.workers,
)

classes   = TeleDatasetSceneGraphIncremental.TELE_CLASSES   # sorted list

# Global counters
n_label_match  = 0   # enc_label == gt_label
n_label_miss   = 0   # enc_label != gt_label
n_active_total = 0

dist_all   = []   # L2 distances of all matched pairs
dist_match = []   # L2 when labels agree
dist_miss  = []   # L2 when labels disagree

# Per-class: how often enc_label is confirmed by GT
per_class = defaultdict(lambda: {'match': 0, 'miss': 0, 'dist': []})

# Label confusion: enc_label → gt_label counts (only for mismatches)
confusion = defaultdict(lambda: defaultdict(int))   # [enc][gt] → count

print(f'\nAnalysing split={args.split}, {len(dataset)} sequences ...\n')

for batch in loader:
    enc    = batch['encoder']
    gt     = batch['gt']
    classes_list = classes

    for flat_sid, matches in enumerate(batch['active_matches']):
        if not matches:
            continue

        for enc_idx, gt_idx, *_ in matches:
            # enc label
            enc_cls_id  = int(enc['objs'][enc_idx].item())
            enc_label   = classes_list[enc_cls_id]

            # GT label
            gt_cls_id   = int(gt['objs'][gt_idx].item())
            gt_label    = classes_list[gt_cls_id]

            # L2 distance between enc centroid and GT centroid (raw world coords)
            enc_pos = enc['boxes'][enc_idx, 3:6].numpy()
            gt_pos  = gt['boxes'][gt_idx,  3:6].numpy()
            dist    = float(np.linalg.norm(enc_pos - gt_pos))

            n_active_total += 1
            dist_all.append(dist)

            if enc_label == gt_label:
                n_label_match += 1
                dist_match.append(dist)
                per_class[enc_label]['match'] += 1
            else:
                n_label_miss  += 1
                dist_miss.append(dist)
                per_class[enc_label]['miss']  += 1
                confusion[enc_label][gt_label] += 1

            per_class[enc_label]['dist'].append(dist)
            per_class[gt_label]          # ensure gt_label is in dict

# ── Print results ──────────────────────────────────────────────────────────────

print('=' * 64)
print(f'Total active matched pairs : {n_active_total}')
print(f'Label agreement            : {n_label_match} / {n_active_total} = '
      f'{100*n_label_match/max(n_active_total,1):.1f}%')
print(f'Label mismatch             : {n_label_miss} / {n_active_total} = '
      f'{100*n_label_miss/max(n_active_total,1):.1f}%')
print()

def _fmt_dist(arr):
    if not arr: return '—'
    a = np.array(arr)
    return f'mean={a.mean():.3f}m  p50={np.median(a):.3f}m  p90={np.percentile(a,90):.3f}m'

print(f'Enc→GT L2 distance (all)     : {_fmt_dist(dist_all)}')
print(f'Enc→GT L2 distance (agree)   : {_fmt_dist(dist_match)}')
print(f'Enc→GT L2 distance (disagree): {_fmt_dist(dist_miss)}')

print()
print('── Per-class label accuracy (sorted by mismatch rate) ──')
print(f'  {"class":25s}  {"n_match":>8}  {"n_miss":>7}  {"acc%":>6}  {"avg_dist":>9}')
print('  ' + '-'*60)
rows = sorted(per_class.items(),
              key=lambda kv: kv[1]['miss'] / max(kv[1]['match']+kv[1]['miss'], 1),
              reverse=True)
for lbl, acc in rows:
    n_m  = acc['match']
    n_mm = acc['miss']
    n    = n_m + n_mm
    if n == 0:
        continue
    pct  = 100 * n_m / n
    avg_d = np.mean(acc['dist']) if acc['dist'] else 0.0
    print(f'  {lbl:25s}  {n_m:8d}  {n_mm:7d}  {pct:6.1f}%  {avg_d:9.3f}m')

print()
print('── Cross-label confusions (enc_label → gt_label, top mismatches) ──')
conf_flat = []
for enc_lb, gt_dict in confusion.items():
    for gt_lb, cnt in gt_dict.items():
        conf_flat.append((cnt, enc_lb, gt_lb))
conf_flat.sort(reverse=True)
for cnt, enc_lb, gt_lb in conf_flat[:20]:
    print(f'  {enc_lb:25s} → {gt_lb:25s} : {cnt}')

# distance threshold analysis
print()
print('── Matched pairs by distance threshold ──')
dist_arr = np.array(dist_all)
for thr in [0.3, 0.5, 0.75, 1.0, 1.5]:
    frac = (dist_arr < thr).mean() * 100
    print(f'  < {thr:.2f}m : {frac:.1f}%')
