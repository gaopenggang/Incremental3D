import numpy as np


def normalize_box_params(box_params, file=None, scale=3):
    """Normalize box parameters [l,h,w,cx,cy,cz,yaw] for stable learning."""
    if file is None:
        mean = np.array([1.3827214, 1.309359, 0.9488993,
                         -0.12464812, 0.6188591, -0.54847, 0.73127955])
        std  = np.array([1.7797655, 1.657638, 0.8501885,
                          1.9160025, 2.0038228, 0.70099753, 0.50347435])
    else:
        stats = np.loadtxt(file)
        mean, std = stats[0], stats[1]
    return scale * ((box_params - mean) / std)


def denormalize_box_params(box_params, file=None, scale=3, params=7):
    """Inverse of normalize_box_params."""
    if file is None:
        if params == 6:
            mean = np.array([1.3827214, 1.309359, 0.9488993,
                             -0.12464812, 0.6188591, -0.54847])
            std  = np.array([1.7797655, 1.657638, 0.8501885,
                              1.9160025, 2.0038228, 0.70099753])
        else:
            mean = np.array([1.3827214, 1.309359, 0.9488993,
                             -0.12464812, 0.6188591, -0.54847, 0.73127955])
            std  = np.array([1.7797655, 1.657638, 0.8501885,
                              1.9160025, 2.0038228, 0.70099753, 0.50347435])
    else:
        stats = np.loadtxt(file)
        if params == 6:
            mean, std = stats[0][:6], stats[1][:6]
        else:
            mean, std = stats[0], stats[1]
    return (box_params * std) / scale + mean


def bool_flag(s):
    """argparse helper: accept True/False as bool."""
    if s in ('1', 'True'):
        return True
    if s in ('0', 'False'):
        return False
    raise ValueError('Invalid bool flag: %s  (use 0/False or 1/True)' % s)
