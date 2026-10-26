"""Ground-truth loop labels for keyframes of a TUM RGB-D sequence.

Label codes for a pair (query i, database j):
  POS       revisit: close in position AND viewing direction
  NEG       clearly a different place
  IGNORE    ambiguous band between POS and NEG: never counted as TP or FP
  EXCLUDED  j is not older than i by t_excl (odometry's job, not loop detection)
  UNKNOWN   i or j has no ground-truth pose (mocap gap)
"""
import numpy as np

POS, NEG, IGNORE, EXCLUDED, UNKNOWN = 1, 0, -1, -2, -3


def quat_to_R(qx, qy, qz, qw):
    """Unit quaternion (Hamilton, x-y-z-w order as in TUM files) -> 3x3 rotation."""
    return np.array([
        [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw),     2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw),     1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw),     2 * (qy*qz + qx*qw),     1 - 2 * (qx*qx + qy*qy)]])


def load_tum_trajectory(path):
    """'timestamp tx ty tz qx qy qz qw' per line -> stamps (N,), T_w_c (N, 4, 4)."""
    data = np.loadtxt(path, comments="#")
    T = np.tile(np.eye(4), (len(data), 1, 1))
    for k, (_, tx, ty, tz, qx, qy, qz, qw) in enumerate(data):
        T[k, :3, :3] = quat_to_R(qx, qy, qz, qw)
        T[k, :3, 3] = (tx, ty, tz)
    return data[:, 0], T


def poses_at(stamps, gt_stamps, gt_T, max_dt=0.02):
    """Nearest GT pose for each keyframe stamp; valid[i] False if the gap is > max_dt."""
    idx = np.clip(np.searchsorted(gt_stamps, stamps), 1, len(gt_stamps) - 1)
    left_closer = np.abs(gt_stamps[idx - 1] - stamps) < np.abs(gt_stamps[idx] - stamps)
    idx = np.where(left_closer, idx - 1, idx)
    valid = np.abs(gt_stamps[idx] - stamps) <= max_dt
    return gt_T[idx], valid


def label_matrix(stamps, T, valid, d_pos=1.0, ang_pos=30.0, d_neg=2.0, ang_neg=60.0,
                 t_excl=20.0):
    """L[i, j] for every keyframe pair. Only j with stamps[i] - stamps[j] >= t_excl is labelled."""
    p = T[:, :3, 3]                      # camera centres in the world frame [m]
    z = T[:, :3, 2]                      # optical axes (3rd column of R_w_c)
    dist = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    ang = np.degrees(np.arccos(np.clip(z @ z.T, -1.0, 1.0)))

    older = (stamps[:, None] - stamps[None, :]) >= t_excl
    known = valid[:, None] & valid[None, :]

    L = np.full(dist.shape, EXCLUDED, dtype=np.int8)
    L[older & ~known] = UNKNOWN
    m = older & known
    L[m] = IGNORE
    L[m & ((dist > d_neg) | (ang > ang_neg))] = NEG
    L[m & (dist <= d_pos) & (ang <= ang_pos)] = POS
    return L