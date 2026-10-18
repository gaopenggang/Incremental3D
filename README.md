# Increment3D-ext

Given a noisy, incrementally-built robot scene graph (positions, orientations, and object labels from onboard perception), the model predicts poses and shape features for each object as the robot explores the scene.

---

## Environment Setup

Tested on Ubuntu 20.04, CUDA 11.7.

```bash
conda create -n increment3d python=3.9
conda activate increment3d
conda install -y -c conda-forge cudatoolkit=11.7 cudatoolkit-dev=11.7
pip install --index-url https://download.pytorch.org/whl/cu117 \
    torch==2.0.1+cu117 torchvision==0.15.2+cu117
pip install open3d==0.19.0 pyglet==2.1.2 PyOpenGL==3.1.0
pip install pytorch3d==0.7.4
pip install pyquaternion==0.9.9 pyrender==0.1.45 imageio==2.37.0
pip install -r requirements.txt
```

Build the Chamfer Distance extension (required for shape evaluation):

```bash
cd extension
python setup.py install
cd ..
```

---

## Dataset

The dataset consists of two aligned data sources per sequence:

| Source | Path | Description |
|---|---|---|
| Robot SG | `<robot_root>/<seq>/frame_XXXXXX.json` | Cumulative scene graph from onboard SLAM + YOLO detection |
| Ground Truth | `<gt_root>/<seq>/sg_frame_XXXXX.json` | Per-frame GT 3D annotation (position, orientation, size, shape model) |


### Normalization Statistics

Compute per-split normalization statistics before training (only needed once, or when the dataset changes):

```bash
python scripts/compute_tele_box_stats.py \
    --robot_root "/path/to/robot_sg_root" \
    --gt_root    "/path/to/gt_root" \
    --splits_dir "/path/to/splits"
```

This outputs:
- `boxes_stats_tele_enc_trainval.txt` — robot SG normalization parameters
- `boxes_stats_tele_gt_trainval.txt` — GT normalization parameters

Statistics are computed from train + val only and applied consistently to the test set.

---

## Pretrained Models

Model weights will be released upon paper acceptance.

---

## Training

```bash
python scripts/train_tele_incremental.py \
    --robot_root "/path/to/robot_sg_root" \
    --gt_root    "/path/to/gt_root" \
    --exp        "/path/to/output_experiment" \
    --enc_stats  "/path/to/boxes_stats_tele_enc_trainval.txt" \
    --gt_stats   "/path/to/boxes_stats_tele_gt_trainval.txt" \
    --splits_dir "/path/to/splits" \
    --shape_features_pkl "/path/to/3d_objects/features.pkl" \
    --batchSize 5 \
    --nepoch 200
```

Key training arguments:

| Argument | Default | Description |
|---|---|---|
| `--node_dim` | 128 | GCN node embedding dimension |
| `--num_layers` | 5 | Number of GCN layers |
| `--shape_feat_dim` | 768 | Shape feature dimension (OpenShape) |
| `--feat_weight` | 0.1 | Weight for shape feature cosine loss |
| `--cls_weight` | 1.0 | Weight for label correction CE loss |
| `--lr` | 1e-4 | Initial learning rate (cosine annealing) |
| `--grad_clip` | 5.0 | Gradient clipping norm |

Training logs are written to TensorBoard under `<exp>/logs/`.

---

## Evaluation

```bash
python scripts/eval_tele_incremental.py \
    --exp        "/path/to/experiment" \
    --epoch      199 \
    --split      val \
    --enc_stats  "/path/to/boxes_stats_tele_enc_trainval.txt" \
    --gt_stats   "/path/to/boxes_stats_tele_gt_trainval.txt" \
    --splits_dir "/path/to/splits"
```
python scripts/eval_tele_incremental.py     --exp   /home/gao/data_weight/tele_incremental_12    --epoch 199     --split val     --enc_stats "/media/gao/gao_ssd/3D Scene/boxes_stats_tele_enc_trainval.txt"     --gt_stats  "/media/gao/gao_ssd/3D Scene/boxes_stats_tele_gt_trainval.txt" --splits_dir "/media/gao/gao_ssd/3D Scene/splits" 

python scripts/eval_tele_incremental.py \
    --exp        /home/gao/data_weight/tele_incremental_12 \
    --epoch      199 \
    --split      val \
    --enc_stats  "/media/gao/gao_ssd/3D Scene/boxes_stats_tele_enc_trainval.txt" \
    --gt_stats   "/media/gao/gao_ssd/3D Scene/boxes_stats_tele_gt_trainval.txt" \
    --splits_dir "/media/gao/gao_ssd/3D Scene/splits" \
    --seq        27_3 \
    --visualize  True \
    --shape_features_pkl "/media/gao/gao_ssd/3D objects/features.pkl" \
    --obj_dir    "/media/gao/gao_ssd/3D objects"

python scripts/ros_telescene_inference.py     --exp      /home/gao/data_weight/tele_incremental_12     --epoch    199     --enc_stats "/media/gao/gao_ssd/3D Scene/boxes_stats_tele_enc_trainval.txt"     --gt_stats  "/media/gao/gao_ssd/3D Scene/boxes_stats_tele_gt_trainval.txt"     --obj_dir   "/media/gao/gao_ssd/3D objects"     --shape_features_pkl "/media/gao/gao_ssd/3D objects/features.pkl" --pose_deadband_m 0.1

conda create -n increment3d python=3.8
conda activate increment3d
source /opt/ros/foxy/setup.bash          # 让 rclpy/scene_msgs 可见（cp38 匹配）
source ~/ros2_ws/install/setup.bash      # scene_msgs 在这个工作空间里
# README 里的依赖：torch 2.0.1+cu117 不支持 3.8 的话用 torch 2.0.1 的 cp38 wheel（PyPI/cu117 index 有 cp38）
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install open3d trimesh scipy tensorboardX numpy   # open3d 0.19 需要 Python ≥3.8，可用

The evaluator reports:

- **Position error** (L1, metres) — overall and per-class
- **Yaw error** (degrees)
- **Shape retrieval** — Chamfer Distance, F-score
- **Full vs. partial observation** — metrics split by enc/GT footprint ratio ≥ 0.7

---

## Visualization

### SLAM-style Incremental Visualization

To visualize the model's step-by-step inference on a specific sequence:

```bash
python scripts/eval_tele_incremental.py \
    --exp        "/path/to/experiment" \
    --epoch      199 \
    --split      val \
    --enc_stats  "/path/to/boxes_stats_tele_enc_trainval.txt" \
    --gt_stats   "/path/to/boxes_stats_tele_gt_trainval.txt" \
    --splits_dir "/path/to/splits" \
    --seq        36_2 \
    --visualize
```

**Ground-truth 3D scene graph:**

```bash
python scripts/visualize_gt_3d.py \
    --sg_dir "/path/to/gt_root/<seq>" \
    --obj_dir "/path/to/3d_objects"
```

**Robot-side (operator) scene graph:**

```bash
python scripts/visualize_robot_3d.py \
    --sg_dir  "/path/to/robot_sg_root/<seq>" \
    --obj_dir "/path/to/3d_objects"
```


