import numpy as np
import torch


def corners_from_box(box):
    """Return (8,3) corners of an OBB given [l,h,w,x,y,z,theta_deg]."""
    l, h, w, x, y, z, theta = box.tolist()
    xt = torch.tensor([ w/2, w/2,-w/2,-w/2, w/2, w/2,-w/2,-w/2])
    yt = torch.tensor([ h/2, h/2, h/2, h/2,-h/2,-h/2,-h/2,-h/2])
    zt = torch.tensor([ l/2,-l/2,-l/2, l/2, l/2,-l/2,-l/2, l/2])
    R  = torch.tensor([[ np.cos(np.deg2rad(theta)), 0, np.sin(np.deg2rad(theta))],
                        [ 0,                         1, 0                        ],
                        [-np.sin(np.deg2rad(theta)), 0, np.cos(np.deg2rad(theta))]])
    pts = torch.stack([xt, yt, zt], 1).double() @ R.double().T \
          + torch.tensor([x, y, z], dtype=torch.float64)
    return pts   # (8,3)


def box_volume_from_corners(c):
    a = (c[0]-c[1]).norm(); b = (c[1]-c[2]).norm(); c_ = (c[0]-c[4]).norm()
    return (a * b * c_).item()


# ── pure-numpy 2D convex polygon intersection (Sutherland-Hodgman) ─────────────

def _sh_inside(p, a, b):
    """True if point p is on the left side (or on) edge a→b."""
    return (b[0]-a[0])*(p[1]-a[1]) >= (b[1]-a[1])*(p[0]-a[0])


def _sh_intersect(p1, p2, p3, p4):
    """Intersection of line p1-p2 with line p3-p4."""
    d1 = (p2[0]-p1[0], p2[1]-p1[1])
    d2 = (p4[0]-p3[0], p4[1]-p3[1])
    cross = d1[0]*d2[1] - d1[1]*d2[0]
    if abs(cross) < 1e-10:
        return p1
    t = ((p3[0]-p1[0])*d2[1] - (p3[1]-p1[1])*d2[0]) / cross
    return (p1[0]+t*d1[0], p1[1]+t*d1[1])


def _clip_polygon(subject, clip):
    """Sutherland-Hodgman: clip convex subject polygon against convex clip polygon."""
    output = list(subject)
    n = len(clip)
    for i in range(n):
        if not output:
            return []
        a, b = clip[i], clip[(i+1) % n]
        inp = output
        output = []
        for j in range(len(inp)):
            curr = inp[j]
            prev = inp[j-1]
            if _sh_inside(curr, a, b):
                if not _sh_inside(prev, a, b):
                    output.append(_sh_intersect(prev, curr, a, b))
                output.append(curr)
            elif _sh_inside(prev, a, b):
                output.append(_sh_intersect(prev, curr, a, b))
    return output


def _polygon_area(pts):
    """Shoelace formula for polygon area."""
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i+1) % n
        area += pts[i][0]*pts[j][1] - pts[j][0]*pts[i][1]
    return abs(area) / 2.0


def intersection_volume(c1: np.ndarray, c2: np.ndarray) -> float:
    """3D intersection volume of two OBBs via XZ-plane polygon clipping."""
    # corners_from_box returns CW winding in XZ; reverse to CCW for SH algorithm
    pts1 = [(float(c1[i, 0]), float(c1[i, 2])) for i in range(3, -1, -1)]
    pts2 = [(float(c2[i, 0]), float(c2[i, 2])) for i in range(3, -1, -1)]
    clipped = _clip_polygon(pts1, pts2)
    area = _polygon_area(clipped)
    if area <= 0.0:
        return 0.0
    y_min1, y_max1 = float(c1[:, 1].min()), float(c1[:, 1].max())
    y_min2, y_max2 = float(c2[:, 1].min()), float(c2[:, 1].max())
    h_int = max(0.0, min(y_max1, y_max2) - max(y_min1, y_min2))
    return area * h_int


def obb_overlap_gt(box_pred, box_gt):
    c_pred = corners_from_box(box_pred).numpy()
    c_gt   = corners_from_box(box_gt).numpy()
    vol_gt = box_volume_from_corners(torch.from_numpy(c_gt))
    v_int  = intersection_volume(c_pred, c_gt)
    return v_int / (vol_gt + 1e-6)


def validate_box_accuracy(boxes_pred: torch.Tensor,
                          gt_boxes:   torch.Tensor) -> dict:
    """
    Compute position / size / orientation / overlap metrics for a set of
    matched (pred, gt) box pairs.

    boxes_pred, gt_boxes : (N, 7)  [l, h, w, cx, cy, cz, yaw_deg]
    Returns a dict of scalar metrics.
    """
    size_p,   center_p, ang_p = boxes_pred[:, :3], boxes_pred[:, 3:6], boxes_pred[:, 6]
    size_g,   center_g, ang_g = gt_boxes[:,  :3],  gt_boxes[:,  3:6],  gt_boxes[:,  6]

    pos_err  = (center_p - center_g).square().sum(1).sqrt()
    size_err = (size_p   - size_g  ).square().sum(1).sqrt()
    ang_diff = (ang_p - ang_g).abs() % 360.0
    ang_err  = torch.minimum(ang_diff, 360.0 - ang_diff)

    size_norm  = size_g.square().sum(1).sqrt().clamp(min=1e-6)
    size_err_p = size_err / size_norm
    ang_err_p  = ang_err / 360.0

    overlaps = torch.tensor(
        [obb_overlap_gt(bp.cpu(), bg.cpu()) for bp, bg in zip(boxes_pred, gt_boxes)],
        device=boxes_pred.device,
    )

    return {
        'num_boxes'          : boxes_pred.shape[0],
        'position'           : pos_err.mean().item(),
        'size'               : size_err.mean().item(),
        'orientation'        : ang_err.mean().item(),
        'Overlap_GT(%)'      : overlaps.mean().item() * 100.,
        'size_p(%)'          : size_err_p.mean().item() * 100.,
        'orientation_p(%)'   : ang_err_p.mean().item() * 100.,
    }


# ── Shape evaluation ──────────────────────────────────────────────────────────

def _sample_points(mesh, n: int) -> np.ndarray:
    """Sample n surface points from an open3d or trimesh mesh."""
    try:
        import open3d as o3d
        import trimesh
    except ImportError:
        raise ImportError("open3d and trimesh are required for shape evaluation")

    if isinstance(mesh, o3d.geometry.TriangleMesh):
        return np.asarray(mesh.sample_points_uniformly(n).points)
    elif isinstance(mesh, trimesh.Trimesh):
        pts, _ = trimesh.sample.sample_surface(mesh, n)
        return pts
    else:
        raise TypeError(f"unsupported mesh type: {type(mesh)}")


def _to_o3d(mesh):
    """Convert trimesh → open3d TriangleMesh, pass through open3d meshes."""
    import open3d as o3d
    import trimesh

    if isinstance(mesh, o3d.geometry.TriangleMesh):
        return mesh
    elif isinstance(mesh, trimesh.Trimesh):
        m = o3d.geometry.TriangleMesh()
        m.vertices  = o3d.utility.Vector3dVector(mesh.vertices)
        m.triangles = o3d.utility.Vector3iVector(mesh.faces)
        return m
    else:
        raise TypeError(f"unsupported mesh type: {type(mesh)}")



def _get_chamfer():
    """Lazy-load the CUDA chamfer extension; fall back to pure-PyTorch on CPU."""
    try:
        import sys as _sys
        _sys.path.insert(0, '/media/gao/gao_ssd/Increment3D')
        from extension.dist_chamfer import chamferDist
        return chamferDist(), True   # (module, is_cuda)
    except Exception:
        return None, False


def validate_shape_accuracy(shapes_mesh_pred, shapes_mesh_gt,
                            n_sample: int   = 2048,
                            f_thresh: float = 0.1,
                            voxel_size: float = 0.25) -> dict:
    """
    Compute shape quality metrics for a list of (pred, gt) mesh pairs.

    Args:
        shapes_mesh_pred / shapes_mesh_gt : lists of open3d.TriangleMesh or trimesh.Trimesh
        n_sample   : surface points sampled per mesh for CD and F-score
        f_thresh   : distance threshold (metres) for F-score
        voxel_size : voxel edge length (metres) for volumetric IoU

    Returns dict with keys: num_shapes, CD, F-score, Vox_IoU
    """
    import open3d as o3d
    chamfer_fn, use_cuda = _get_chamfer()

    cd_vals, f_vals, vox_vals = [], [], []

    for mesh_p, mesh_g in zip(shapes_mesh_pred, shapes_mesh_gt):
        pts_p = _sample_points(mesh_p, n_sample)   # (N, 3)
        pts_g = _sample_points(mesh_g, n_sample)

        # ── Chamfer Distance ─────────────────────────────────────────────────
        if use_cuda and torch.cuda.is_available():
            # CUDA extension: input (1, N, 3), returns squared distances (1, N)
            p = torch.from_numpy(pts_p).float().unsqueeze(0).cuda()
            g = torch.from_numpy(pts_g).float().unsqueeze(0).cuda()
            dist1, dist2 = chamfer_fn(p, g)        # (1,N), (1,M) – squared
            d1 = dist1.squeeze(0).sqrt()            # (N,)  pred → gt
            d2 = dist2.squeeze(0).sqrt()            # (M,)  gt → pred
        else:
            p = torch.from_numpy(pts_p).float()
            g = torch.from_numpy(pts_g).float()
            sq = (p.unsqueeze(1) - g.unsqueeze(0)).pow(2).sum(-1)   # (N, M)
            d1 = sq.min(1).values.sqrt()
            d2 = sq.min(0).values.sqrt()
        cd = (d1.mean() + d2.mean()).item()

        # ── F-score ──────────────────────────────────────────────────────────
        prec = (d1 < f_thresh).float().mean()
        reca = (d2 < f_thresh).float().mean()
        denom = prec + reca
        fscore = (2 * prec * reca / denom).item() if denom > 0 else 0.0

        # ── Voxel IoU ────────────────────────────────────────────────────────
        vox_p = o3d.geometry.VoxelGrid.create_from_triangle_mesh(_to_o3d(mesh_p), voxel_size)
        vox_g = o3d.geometry.VoxelGrid.create_from_triangle_mesh(_to_o3d(mesh_g), voxel_size)
        idx_p = {tuple(v.grid_index) for v in vox_p.get_voxels()}
        idx_g = {tuple(v.grid_index) for v in vox_g.get_voxels()}
        inter = len(idx_p & idx_g)
        union = len(idx_p | idx_g)
        vox_iou = inter / (union + 1e-6)

        cd_vals.append(cd)
        f_vals.append(fscore)
        vox_vals.append(vox_iou)

    return {
        'num_shapes': len(cd_vals),
        'CD'        : float(np.mean(cd_vals)),
        'F-score'   : float(np.mean(f_vals)),
        'Vox_IoU'   : float(np.mean(vox_vals)),
    }
