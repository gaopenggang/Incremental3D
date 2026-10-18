"""
Training script for Sg2ScAEModelTeleIncremental.

Usage:
  python scripts/train_tele_incremental.py \
      --robot_root /media/gao/gao_ssd/3D\ scene\ graph \
      --gt_root    /media/gao/gao_ssd/3D\ Scene \
      --exp        /home/gao/data_weight/tele_incremental
"""
from __future__ import print_function
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse, random, json, time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataset.Tele_dataset_incremental import (
    TeleDatasetSceneGraphIncremental, LengthGroupedCurriculumSampler)
from model.AE_tele_incremental import (
    Sg2ScAEModelTeleIncremental, compute_active_loss)
from helpers.util import bool_flag


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--robot_root', default='/media/gao/gao_ssd/3D scene graph')
parser.add_argument('--gt_root',    default='/media/gao/gao_ssd/3D Scene')
parser.add_argument('--exp',        default='/home/gao/data_weight/tele_incremental')
parser.add_argument('--outf',       default='checkpoint')
parser.add_argument('--logf',       default='logs')

parser.add_argument('--batchSize',  type=int,   default=4)
parser.add_argument('--nepoch',     type=int,   default=200)
parser.add_argument('--lr',         type=float, default=1e-4)
parser.add_argument('--workers',    type=int,   default=4)

parser.add_argument('--node_dim',      type=int, default=128)
parser.add_argument('--num_layers',    type=int, default=5)
parser.add_argument('--shape_feat_dim',type=int, default=768)
parser.add_argument('--residual',   type=bool_flag, default=True)
parser.add_argument('--pooling',    type=str,   default='avg')

parser.add_argument('--feat_weight',   type=float, default=0.1,
                    help='weight for shape feature cosine loss')
parser.add_argument('--delta_weight',  type=float, default=1.0,
                    help='weight for intra-bin delta L1 (coarse-to-fine yaw)')
parser.add_argument('--cls_weight',      type=float, default=1.0,
                    help='weight for label correction cross-entropy loss')
parser.add_argument('--cls_dist_thresh', type=float, default=0.75,
                    help='max 3D match distance (m) to include a pair in cls loss')
parser.add_argument('--shape_features_pkl', type=str,
                    default='/media/gao/gao_ssd/3D objects/features.pkl',
                    help='path to 3D library features.pkl')
parser.add_argument('--grad_clip',   type=float, default=5.0)
parser.add_argument('--save_every',  type=int,   default=10)
parser.add_argument('--log_every',   type=int,   default=50)

parser.add_argument('--loadepoch',    type=int, default=-1,
                    help='resume from this epoch checkpoint (-1 = start fresh)')
parser.add_argument('--enc_stats', type=str, default='',
                    help='encoder (robot SG) normalization stats file')
parser.add_argument('--gt_stats',  type=str, default='',
                    help='GT normalization stats file')
parser.add_argument('--splits_dir', type=str, default='',
                    help='directory with train/val/test_sequences.txt')
args = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def to_cuda(x, device='cuda'):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_cuda(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [to_cuda(v, device) for v in x]
    return x


def require_stats(path, arg_name):
    if not path:
        raise ValueError(
            f'{arg_name} is required. Run scripts/compute_tele_box_stats.py '
            'and pass the generated uncentered stats file.')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{arg_name} not found: {path}')
    return path


def save_checkpoint(model, optimizer, epoch, counter, exp):
    ckpt_dir = os.path.join(exp, args.outf)
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f'model_{epoch}.pth')
    torch.save({
        'epoch':     epoch,
        'counter':   counter,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path)
    print(f'[ckpt] saved {path}')


def load_checkpoint(model, optimizer, epoch, exp):
    path = os.path.join(exp, args.outf, f'model_{epoch}.pth')
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['epoch'], ckpt['counter']


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────
def train():
    seed = random.randint(1, 10000)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Seed: {seed}  |  PyTorch {torch.__version__}  |  device={device}')
    enc_stats = require_stats(args.enc_stats, '--enc_stats')
    gt_stats  = require_stats(args.gt_stats,  '--gt_stats')

    dataset = TeleDatasetSceneGraphIncremental(
        robot_sg_root        = args.robot_root,
        gt_scene_root        = args.gt_root,
        split                = 'train',
        shuffle_objs         = True,
        seed                 = False,
        enc_normalized_stats = enc_stats,
        gt_normalized_stats  = gt_stats,
        shape_features_pkl   = args.shape_features_pkl or None,
        splits_dir           = args.splits_dir or None,
    )
    loader = DataLoader(
        dataset,
        batch_sampler = LengthGroupedCurriculumSampler(dataset, args.batchSize),
        collate_fn    = dataset.collate_fn_inc,
        num_workers   = args.workers,
        pin_memory    = (device.type == 'cuda'),
    )

    vocab = {
        'object_idx_to_name': {i: c for i, c in
                                enumerate(TeleDatasetSceneGraphIncremental.TELE_CLASSES)},
        'pred_idx_to_name':   {i: r for i, r in
                                enumerate(TeleDatasetSceneGraphIncremental.TELE_RELATIONS)},
    }

    model = Sg2ScAEModelTeleIncremental(
        vocab           = vocab,
        node_embedding_dim = args.node_dim,
        gconv_num_layers   = args.num_layers,
        shape_feat_dim     = args.shape_feat_dim,
        residual           = args.residual,
        gconv_pooling      = args.pooling,
    )
    model = model.to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.nepoch, eta_min=args.lr * 0.05)

    start_epoch, counter = 0, 0
    if args.loadepoch >= 0:
        start_epoch, counter = load_checkpoint(model, optimizer, args.loadepoch, args.exp)
        start_epoch += 1
        print(f'[resume] epoch {args.loadepoch}, counter {counter}')

    os.makedirs(args.exp, exist_ok=True)
    with open(os.path.join(args.exp, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    writer = SummaryWriter(os.path.join(args.exp, args.logf))

    print(f'Dataset: {len(dataset)} train sequences')
    print('---- Starting training loop ----')

    # Pre-load size stats for AABB rotation loss (denormalisation)
    enc_size_mu = enc_size_sg = gt_size_mu = gt_size_sg = None
    _es = np.loadtxt(enc_stats)
    enc_size_mu = torch.tensor(_es[0, :3], dtype=torch.float32)
    enc_size_sg = torch.tensor(_es[1, :3], dtype=torch.float32)
    _gs = np.loadtxt(gt_stats)
    gt_size_mu = torch.tensor(_gs[0, :3], dtype=torch.float32)
    gt_size_sg = torch.tensor(_gs[1, :3], dtype=torch.float32)

    for epoch in range(start_epoch, args.nepoch):
        model.train()
        epoch_loss = 0.0
        epoch_pos  = 0.0
        epoch_rot  = 0.0
        epoch_feat = 0.0
        epoch_cls  = 0.0
        epoch_pairs = 0
        n_batches  = 0

        for batch in loader:
            batch = to_cuda(batch, device)
            active_pairs = sum(len(m) for m in batch['active_matches'])
            if active_pairs == 0:
                continue

            preds = model(batch, reset_states=True)
            loss, ld = compute_active_loss(
                preds, batch,
                feat_weight=args.feat_weight,
                delta_weight=args.delta_weight,
                cls_weight=args.cls_weight,
                cls_dist_thresh=args.cls_dist_thresh,
                enc_size_mu=enc_size_mu, enc_size_sg=enc_size_sg,
                gt_size_mu=gt_size_mu,   gt_size_sg=gt_size_sg,
            )

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item() * active_pairs
            epoch_pos  += ld['pos'].item() * active_pairs
            epoch_rot  += ld['rot'].item() * active_pairs
            epoch_feat += ld['feat'].item() * active_pairs
            epoch_cls  += ld['cls'].item() * active_pairs
            epoch_pairs += active_pairs
            n_batches  += 1
            counter    += 1

            writer.add_scalar('Train/loss', loss.item(), counter)
            writer.add_scalar('Train/pos',  ld['pos'].item(), counter)
            writer.add_scalar('Train/rot',  ld['rot'].item(), counter)
            writer.add_scalar('Train/feat', ld['feat'].item(), counter)
            writer.add_scalar('Train/cls',  ld['cls'].item(), counter)

        scheduler.step()

        avg = epoch_loss / max(epoch_pairs, 1)
        print(f'Epoch {epoch:03d} — avg loss={avg:.4f}  '
              f'pos={epoch_pos/max(epoch_pairs,1):.4f}  '
              f'rot={epoch_rot/max(epoch_pairs,1):.4f}  '
              f'feat={epoch_feat/max(epoch_pairs,1):.4f}  '
              f'cls={epoch_cls/max(epoch_pairs,1):.4f}  '
              f'pairs={epoch_pairs}  '
              f'lr={scheduler.get_last_lr()[0]:.2e}')
        writer.add_scalar('Epoch/loss', avg, epoch)
        writer.add_scalar('Epoch/pos',  epoch_pos / max(epoch_pairs, 1), epoch)
        writer.add_scalar('Epoch/rot',  epoch_rot / max(epoch_pairs, 1), epoch)
        writer.add_scalar('Epoch/feat', epoch_feat / max(epoch_pairs, 1), epoch)
        writer.add_scalar('Epoch/cls',  epoch_cls  / max(epoch_pairs, 1), epoch)
        writer.add_scalar('Epoch/pairs', epoch_pairs, epoch)
        writer.add_scalar('Epoch/lr',   scheduler.get_last_lr()[0], epoch)

        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, counter, args.exp)

    save_checkpoint(model, optimizer, args.nepoch - 1, counter, args.exp)
    writer.close()
    print('Training complete.')


if __name__ == '__main__':
    train()
