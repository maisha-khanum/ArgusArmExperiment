#!/usr/bin/env python3
"""Display a 3D pose trajectory from an npz file, coloured by time.

Usage:
    python3 plot_npz.py NPZ_PATH [--out FIG.png] [--frames N] [--no-frames]

Handles both npz layouts produced in this project:
  * c3d rigid-body pose  (keys: R, t, quat, centroid, valid, frame_rate, ...)
  * VIO odometry         (keys: t, t_rel, position, orientation_wxyz, ...)

The path is drawn as a line coloured from start (dark) to end (bright) by
time, with orientation triads (body X/Y/Z axes) sampled along it.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    """(N,4) Hamilton quaternion (w,x,y,z) -> (N,3,3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def load(npz_path: Path):
    """Return (pos Nx3, time N, R Nx3x3|None, units, kind)."""
    d = np.load(npz_path)
    keys = set(d.files)

    if "position" in keys:                       # VIO odometry
        pos = np.asarray(d["position"], float)
        time = np.asarray(d["t_rel"] if "t_rel" in keys else d["t"], float)
        time = time - time[0]
        R = quat_wxyz_to_R(np.asarray(d["orientation_wxyz"], float)) \
            if "orientation_wxyz" in keys else None
        units, kind = "m", "VIO odometry"

    elif "centroid" in keys or "R" in keys:      # c3d rigid-body pose
        pos = np.asarray(d["centroid" if "centroid" in keys else "t"], float)
        n = pos.shape[0]
        fr = float(d["frame_rate"]) if "frame_rate" in keys else 0.0
        time = np.arange(n) / fr if fr > 0 else np.arange(n, dtype=float)
        R = np.asarray(d["R"], float) if "R" in keys else None
        units, kind = "mm", "c3d pose"

    else:
        raise ValueError(f"Unrecognised npz layout; keys = {sorted(keys)}")

    # Drop frames with no valid position.
    good = np.isfinite(pos).all(axis=1)
    if "valid" in keys and d["valid"].shape[0] == pos.shape[0]:
        good &= np.asarray(d["valid"], bool)
    pos, time = pos[good], time[good]
    if R is not None:
        R = R[good]
    if len(pos) == 0:
        raise ValueError("No valid pose samples to plot.")
    return pos, time, R, units, kind


def colored_path(ax, pos, time, cmap="viridis"):
    """Draw pos (N,3) as a line whose colour encodes time."""
    pts = pos.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = plt.Normalize(time.min(), time.max())
    lc = Line3DCollection(segs, cmap=cmap, norm=norm, linewidth=2)
    lc.set_array(time[:-1])
    ax.add_collection3d(lc)
    return lc


def draw_triads(ax, pos, R, n_frames, scale):
    """Draw n_frames body-axis triads (X=red, Y=green, Z=blue) along the path."""
    if R is None or n_frames <= 0:
        return
    idx = np.linspace(0, len(pos) - 1, min(n_frames, len(pos))).astype(int)
    for axis, c in enumerate(("r", "g", "b")):
        for i in idx:
            o, v = pos[i], R[i][:, axis] * scale
            ax.plot([o[0], o[0] + v[0]], [o[1], o[1] + v[1]],
                    [o[2], o[2] + v[2]], color=c, linewidth=1.5)


def set_equal_aspect(ax, pos):
    mins, maxs = pos.min(0), pos.max(0)
    center = (mins + maxs) / 2
    r = (maxs - mins).max() / 2 or 1.0
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    return r


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path, help="path to a *_pose.npz or *_odometry.npz")
    ap.add_argument("--out", type=Path, help="save figure here instead of showing")
    ap.add_argument("--frames", type=int, default=12, help="number of orientation triads")
    ap.add_argument("--no-frames", action="store_true", help="don't draw orientation triads")
    ap.add_argument("--cmap", default="viridis")
    args = ap.parse_args()

    pos, time, R, units, kind = load(args.npz)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    lc = colored_path(ax, pos, time, args.cmap)
    r = set_equal_aspect(ax, pos)
    if not args.no_frames:
        draw_triads(ax, pos, R, args.frames, scale=0.12 * r)

    ax.scatter(*pos[0], color="k", s=40, marker="o", label="start")
    ax.scatter(*pos[-1], color="k", s=60, marker="*", label="end")

    ax.set_xlabel(f"x [{units}]")
    ax.set_ylabel(f"y [{units}]")
    ax.set_zlabel(f"z [{units}]")
    ax.set_title(f"{kind}: {args.npz.name}\n"
                 f"{len(pos)} samples, {time[-1] - time[0]:.1f}s")
    ax.legend(loc="upper left")

    cb = fig.colorbar(lc, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label("time [s]")

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
