"""
Compute per-side normalization statistics for the Tele dataset.

Encoder (robot SG) and GT use separate stats because their size distributions
differ (E2 occlusion error makes robot bbox smaller than GT).  Positions stay
in the original world/map coordinate frame; no scene centering is applied.

Outputs
-------
  <gt_root>/boxes_stats_tele_enc_trainval.txt   — encoder stats
  <gt_root>/boxes_stats_tele_gt_trainval.txt    — GT stats

File format (2 rows × 7 cols):
  row 0: mean  [l, h, w, cx, cy, cz, yaw]
  row 1: std   [l, h, w, cx, cy, cz, yaw]

Note: yaw is restored to its raw-radian value after normalization in
_build_dict, so the yaw column in the stats file has no effect on training.

Usage
-----
  python scripts/compute_tele_box_stats.py \
      --robot_root "/media/gao/gao_ssd/3D scene graph" \
      --gt_root    "/media/gao/gao_ssd/3D Scene"
"""
import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dataset.Tele_dataset_incremental import TeleDatasetSceneGraphIncremental


def collect_boxes(robot_root, gt_root, splits_dir=None):
    """
    Collect box parameters for stats computation, one physical object = one sample.

    Encoder (robot SG) is cumulative: same object appears in every subsequent
    step.  Fix: take only the LAST step, which has the complete accumulated SG
    with each object counted exactly once.

    GT is per-frame visibility: the same physical object re-appears in every
    frame where it is visible, but its world-frame centroid is identical each
    time (GT geometry is static).  Fix: collect GT from ALL steps, deduplicate
    by (class_id, centroid rounded to 1 mm) so each physical object is counted
    exactly once per sequence regardless of how many frames it appears in.
    """
    enc_all, gt_all = [], []
    for split in ('train', 'val'):
        ds = TeleDatasetSceneGraphIncremental(
            robot_sg_root=robot_root,
            gt_scene_root=gt_root,
            split=split,
            splits_dir=splits_dir)
        for i in range(len(ds)):
            steps = ds[i]['steps']
            if not steps:
                continue

            # ── encoder: last step = complete scene, no repetition ──────────
            enc_all.append(steps[-1]['encoder']['boxes'].numpy())

            # ── GT: union across all steps, deduplicated ─────────────────────
            seen    = set()
            gt_dedup = []
            for step in steps:
                boxes = step['gt']['boxes'].numpy()   # (N, 7)
                objs  = step['gt']['objs'].numpy()    # (N,)  class ids
                for j in range(len(boxes)):
                    # key: class + centroid at 1 mm precision
                    key = (int(objs[j]),) + tuple(boxes[j, 3:6].round(3))
                    if key not in seen:
                        seen.add(key)
                        gt_dedup.append(boxes[j])

            if gt_dedup:
                gt_all.append(np.stack(gt_dedup))   # (M_unique, 7)

    return (np.concatenate(enc_all, axis=0),
            np.concatenate(gt_all,  axis=0))


def compute_and_save(boxes, out_path, name, fix_yaw_std=False):
    mean = boxes.mean(axis=0)
    std  = boxes.std(axis=0)
    if fix_yaw_std:
        std[6] = 1.0    # yaw is restored after normalisation; keep std stable
    else:
        std[std < 1e-6] = 1.0

    names = ['l', 'h', 'w', 'cx', 'cy', 'cz', 'yaw']
    print(f'\n=== {name} ({boxes.shape[0]} nodes) ===')
    print(f'{"":>4}  {"mean":>9}  {"std":>8}')
    for i, n in enumerate(names):
        print(f'{n:>4}  {mean[i]:9.4f}  {std[i]:8.4f}')

    np.savetxt(out_path, np.stack([mean, std]))
    print(f'Saved → {out_path}')
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_root',
                        default='/media/gao/gao_ssd/3D scene graph')
    parser.add_argument('--gt_root',
                        default='/media/gao/gao_ssd/3D Scene')
    parser.add_argument('--splits_dir', default=None,
                        help='directory with train/val/test_sequences.txt; '
                             'if omitted uses auto room-based split')
    args = parser.parse_args()

    print('Collecting boxes from train + val splits …')
    enc_boxes, gt_boxes = collect_boxes(args.robot_root, args.gt_root,
                                        splits_dir=args.splits_dir)

    enc_out = os.path.join(args.gt_root,
                           'boxes_stats_tele_enc_trainval.txt')
    gt_out  = os.path.join(args.gt_root,
                           'boxes_stats_tele_gt_trainval.txt')

    compute_and_save(enc_boxes, enc_out, 'Encoder (robot SG)', fix_yaw_std=True)
    compute_and_save(gt_boxes,  gt_out,  'GT',                 fix_yaw_std=False)

    print('\nDone.')
    print(f'  Encoder stats : {enc_out}')
    print(f'  GT stats      : {gt_out}')


if __name__ == '__main__':
    main()
