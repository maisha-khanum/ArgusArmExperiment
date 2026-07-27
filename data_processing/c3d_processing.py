#!/usr/bin/env python3
"""Inspect a C3D file and compute the rigid-body 6-DOF pose.

Usage:
    python3 c3d_processing.py TRIAL_NAME

Prints the data structure and computes/saves the per-frame 6-DOF pose.

Expects the file at: data/TRIAL_NAME/TRIAL_NAME.c3d (relative to this script).
An explicit path to a .c3d file may also be passed instead of a trial name.

Pose is computed by a per-frame Umeyama (Kabsch) rigid fit of the labelled
markers to their configuration in the reference frame (frame 0 by default).
The resulting transform T = (R, t) satisfies  p_now = R @ p_ref + t, i.e. the
object's pose relative to its frame-0 pose, expressed in the Vicon world frame.
"""

import sys
from pathlib import Path

import numpy as np
import ezc3d

# Markers that make up the rigid body (labelled in Nexus).
RIGID_MARKERS = ("argus1", "argus2", "argus3", "argus4")


def resolve_path(arg: str) -> Path:
    """Resolve a trial name (or explicit path) to a .c3d file path."""
    p = Path(arg)
    if p.suffix == ".c3d" and p.is_file():
        return p

    data_dir = Path(__file__).resolve().parent / "data"
    candidate = data_dir / arg / f"{arg}.c3d"
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Could not find a C3D file for '{arg}'. Looked for:\n"
        f"  - {p}\n"
        f"  - {candidate}"
    )


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def inspect(path: Path) -> None:
    print(f"File: {path}")
    print(f"Size: {path.stat().st_size / 1024:.1f} KiB")

    c = ezc3d.c3d(str(path))

    # ---- Header ----------------------------------------------------------
    header = c["header"]
    print_section("HEADER")
    pts = header["points"]
    ana = header["analogs"]
    print(f"Point frames:   {pts['first_frame']} .. {pts['last_frame']} "
          f"(rate {pts['frame_rate']} Hz)")
    print(f"Analog frames:  {ana['first_frame']} .. {ana['last_frame']} "
          f"(rate {ana['frame_rate']} Hz)")

    # ---- Points / markers ------------------------------------------------
    params = c["parameters"]
    point_data = c["data"]["points"]        # shape: (4, n_markers, n_frames)
    print_section("POINTS / MARKERS")
    labels = params["POINT"]["LABELS"]["value"]
    units = params["POINT"].get("UNITS", {}).get("value", ["?"])
    print(f"Number of markers: {len(labels)}")
    print(f"Units:             {units[0] if units else '?'}")
    print(f"Data array shape:  {point_data.shape}  (xyz1, markers, frames)")
    print("Markers:")
    for i, name in enumerate(labels):
        print(f"  [{i:3d}] {name}")

    # ---- Analog channels -------------------------------------------------
    analog_data = c["data"]["analogs"]      # shape: (1, n_channels, n_frames)
    print_section("ANALOG CHANNELS")
    a_labels = params["ANALOG"]["LABELS"]["value"] if "ANALOG" in params else []
    print(f"Number of channels: {len(a_labels)}")
    print(f"Data array shape:   {analog_data.shape}  (1, channels, frames)")
    for i, name in enumerate(a_labels):
        print(f"  [{i:3d}] {name}")

    # ---- Parameter groups ------------------------------------------------
    print_section("PARAMETER GROUPS")
    for group_name, group in params.items():
        keys = [k for k in group.keys() if not k.startswith("__")]
        print(f"  {group_name}: {', '.join(keys)}")


# ---------------------------------------------------------------------------
# Pose computation
# ---------------------------------------------------------------------------
def rigid_fit(ref: np.ndarray, cur: np.ndarray):
    """Umeyama/Kabsch rigid fit (no scale).

    Given matched point sets ref, cur of shape (N, 3), find rotation R (3x3)
    and translation t (3,) minimising || cur_i - (R @ ref_i + t) ||.
    Returns (R, t, rmse).
    """
    mu_ref = ref.mean(axis=0)
    mu_cur = cur.mean(axis=0)
    ref_c = ref - mu_ref
    cur_c = cur - mu_cur

    H = ref_c.T @ cur_c                     # 3x3 cross-covariance
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))  # reflection guard
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = mu_cur - R @ mu_ref

    resid = cur - (ref @ R.T + t)
    rmse = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
    return R, t, rmse


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def rotmat_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> intrinsic XYZ Euler angles (radians)."""
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    ry = np.arcsin(sy)
    if abs(sy) < 1.0 - 1e-9:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        rx = np.arctan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return np.array([rx, ry, rz])


def compute_pose(path: Path, ref_frame: int = 0):
    """Compute per-frame 6-DOF pose of the rigid body.

    Returns a dict of arrays keyed by frame (length = n_frames):
      R          (n, 3, 3)  rotation, p_now = R @ p_ref + t
      t          (n, 3)     translation (mm)
      quat       (n, 4)     orientation quaternion (w, x, y, z)
      euler_xyz  (n, 3)     intrinsic XYZ Euler angles (deg)
      centroid   (n, 3)     centroid of visible markers (mm), absolute
      rmse       (n,)       fit residual (mm)
      n_used     (n,)       markers used in the fit
      valid      (n,)       bool, True if a pose was solved
    """
    c = ezc3d.c3d(str(path))
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    idx = {name: labels.index(name) for name in RIGID_MARKERS if name in labels}
    missing = [m for m in RIGID_MARKERS if m not in idx]
    if missing:
        raise ValueError(f"Missing labelled markers in c3d: {missing}")

    xyz = c["data"]["points"][:3]           # (3, n_markers, n_frames)
    marker_ids = [idx[m] for m in RIGID_MARKERS]
    P = xyz[:, marker_ids, :].transpose(2, 1, 0)   # (n_frames, n_markers, 3)
    n_frames = P.shape[0]

    ref_pts = P[ref_frame]                  # (n_markers, 3)
    ref_valid = ~np.isnan(ref_pts).any(axis=1)
    if ref_valid.sum() < 3:
        raise ValueError(
            f"Reference frame {ref_frame} has <3 visible markers; "
            f"choose a different reference frame."
        )

    R = np.full((n_frames, 3, 3), np.nan)
    t = np.full((n_frames, 3), np.nan)
    quat = np.full((n_frames, 4), np.nan)
    euler = np.full((n_frames, 3), np.nan)
    centroid = np.full((n_frames, 3), np.nan)
    rmse = np.full(n_frames, np.nan)
    n_used = np.zeros(n_frames, dtype=int)
    valid = np.zeros(n_frames, dtype=bool)

    for f in range(n_frames):
        cur = P[f]
        cur_valid = ~np.isnan(cur).any(axis=1)
        common = ref_valid & cur_valid
        n = int(common.sum())
        n_used[f] = n
        if n < 3:
            continue
        Rf, tf, ef = rigid_fit(ref_pts[common], cur[common])
        R[f] = Rf
        t[f] = tf
        quat[f] = rotmat_to_quat(Rf)
        euler[f] = np.degrees(rotmat_to_euler_xyz(Rf))
        centroid[f] = cur[cur_valid].mean(axis=0)
        rmse[f] = ef
        valid[f] = True

    return {
        "R": R, "t": t, "quat": quat, "euler_xyz": euler,
        "centroid": centroid, "rmse": rmse, "n_used": n_used, "valid": valid,
        "ref_frame": ref_frame, "markers": list(RIGID_MARKERS),
        "frame_rate": c["header"]["points"]["frame_rate"],
    }


def save_pose(path: Path, pose: dict) -> None:
    """Save pose to <trial>/<trial>_pose.npz and a flat .csv next to the c3d."""
    out_npz = path.with_name(path.stem + "_pose.npz")
    np.savez_compressed(out_npz, **pose)

    out_csv = path.with_name(path.stem + "_pose.csv")
    header = ("frame,valid,n_used,rmse_mm,"
              "tx,ty,tz,qw,qx,qy,qz,ex_deg,ey_deg,ez_deg,"
              "cx,cy,cz")
    n = len(pose["valid"])
    rows = np.column_stack([
        np.arange(n), pose["valid"].astype(int), pose["n_used"], pose["rmse"],
        pose["t"], pose["quat"], pose["euler_xyz"], pose["centroid"],
    ])
    np.savetxt(out_csv, rows, delimiter=",", header=header, comments="",
               fmt=["%d", "%d", "%d"] + ["%.6f"] * 14)

    # ---- summary ----
    v = pose["valid"]
    print_section("POSE (Umeyama rigid fit)")
    print(f"Reference frame:   {pose['ref_frame']}")
    print(f"Markers used:      {', '.join(pose['markers'])}")
    print(f"Frames:            {n}  (valid: {v.sum()}, unsolved: {(~v).sum()})")
    if v.any():
        print(f"Fit RMSE (mm):     mean {np.nanmean(pose['rmse']):.3f}  "
              f"max {np.nanmax(pose['rmse']):.3f}")
        tv = pose["t"][v]
        print(f"Translation range: "
              f"x[{tv[:,0].min():.1f},{tv[:,0].max():.1f}] "
              f"y[{tv[:,1].min():.1f},{tv[:,1].max():.1f}] "
              f"z[{tv[:,2].min():.1f},{tv[:,2].max():.1f}] mm")
    print(f"\nSaved: {out_npz}")
    print(f"Saved: {out_csv}")


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    path = resolve_path(sys.argv[1])
    inspect(path)
    pose = compute_pose(path, ref_frame=0)
    save_pose(path, pose)


if __name__ == "__main__":
    main()
