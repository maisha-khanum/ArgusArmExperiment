#!/usr/bin/env python3
"""Run Argus_LightVIO on a trial's ROS2 bag and save the odometry as npz.

Usage:
    python3 argus_processing.py TRIAL_NAME

Layout (relative to this script):
    data/TRIAL_NAME/TRIAL_NAME/              rosbag2 dir (.db3 + metadata.yaml)
    argus_repo/ws/                           colcon workspace (built once)
    data/TRIAL_NAME/TRIAL_NAME_odometry.npz  <- output

What it does:
    1. Starts the argus_vio node (subscribes /cam0/image_raw, /imu0).
    2. Records /vio/odometry (nav_msgs/Odometry) in-process via rclpy.
    3. Plays the bag start-to-finish (real time; startup standstill lets the
       filter align gravity + gyro bias, per the repo README).
    4. Saves per-message pose/twist/covariance to a compressed .npz.

The script re-execs itself under a ROS 2 Humble + workspace shell if needed,
so it can be launched with a plain `python3 argus_processing.py TRIAL_NAME`.
"""

import os
import sys
import time
import signal
import shutil
import tempfile
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE / "argus_repo" / "ws"
CONFIG = WS / "install" / "argus_vio" / "share" / "argus_vio" / "config" / "calibration.yaml"

IMAGE_TOPIC = "/cam0/image_raw"
IMU_TOPIC = "/imu0"
ODOM_TOPIC = "/vio/odometry"

# Seconds to keep recording after the bag finishes, to catch trailing odometry.
GRACE_S = 2.0
# Seconds to wait after starting the node before playing the bag.
NODE_WARMUP_S = 2.5


def reexec_with_ros():
    """Re-exec this script inside a bash shell that sources ROS + workspace."""
    ros_setup = "/opt/ros/humble/setup.bash"
    ws_setup = WS / "install" / "setup.bash"
    if not Path(ros_setup).is_file():
        sys.exit(f"ROS 2 Humble not found at {ros_setup}")
    if not ws_setup.is_file():
        sys.exit(f"Workspace not built. Build it with:\n"
                 f"  cd {WS} && colcon build --symlink-install --packages-select argus_vio")
    args = " ".join(f"'{a}'" for a in sys.argv)
    cmd = f"source '{ros_setup}' && source '{ws_setup}' && exec python3 {args}"
    os.environ["ARGUS_ROS_SOURCED"] = "1"
    os.execvp("bash", ["bash", "-lc", cmd])


def resolve_bag(trial: str) -> Path:
    """Return the rosbag2 directory for a trial name (or an explicit path)."""
    p = Path(trial)
    if (p / "metadata.yaml").is_file():
        return p
    bag = HERE / "data" / trial / trial
    if (bag / "metadata.yaml").is_file():
        return bag
    raise FileNotFoundError(
        f"No rosbag2 dir found for '{trial}'. Looked for:\n"
        f"  - {p}/metadata.yaml\n  - {bag}/metadata.yaml"
    )


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    # Ensure we run under a ROS-sourced environment.
    if not os.environ.get("ARGUS_ROS_SOURCED"):
        reexec_with_ros()

    trial = sys.argv[1]
    bag_dir = resolve_bag(trial)
    trial_name = bag_dir.name
    out_dir = bag_dir.parent
    out_npz = out_dir / f"{trial_name}_odometry.npz"

    # Imports that require the ROS environment.
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from nav_msgs.msg import Odometry

    print(f"Bag:     {bag_dir}")
    print(f"Config:  {CONFIG}")
    print(f"Output:  {out_npz}\n")

    # ---- Odometry recorder node ----------------------------------------
    class OdomRecorder(Node):
        def __init__(self):
            super().__init__("argus_odom_recorder")
            self.rows = []
            qos = QoSProfile(depth=200,
                             reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST)
            self.create_subscription(Odometry, ODOM_TOPIC, self._cb, qos)

        def _cb(self, m: Odometry):
            p = m.pose.pose.position
            q = m.pose.pose.orientation
            lv = m.twist.twist.linear
            av = m.twist.twist.angular
            self.rows.append((
                m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                p.x, p.y, p.z,
                q.w, q.x, q.y, q.z,
                lv.x, lv.y, lv.z,
                av.x, av.y, av.z,
                list(m.pose.covariance),
                list(m.twist.covariance),
            ))

    # ---- Start the VIO node -------------------------------------------
    # Logs go to a temp dir; deleted on success, kept only if the run fails.
    log_dir = Path(tempfile.mkdtemp(prefix=f"argus_vio_{trial_name}_"))
    node_log = open(log_dir / "vio_node.log", "w")
    node_proc = subprocess.Popen(
        ["ros2", "run", "argus_vio", "argus_vio_node", "--ros-args",
         "-p", f"image_topic:={IMAGE_TOPIC}",
         "-p", f"imu_topic:={IMU_TOPIC}",
         "-p", f"odom_topic:={ODOM_TOPIC}",
         "-p", f"config_path:={CONFIG}"],
        stdout=node_log, stderr=subprocess.STDOUT,
        start_new_session=True,   # own process group, so we can kill the child node
    )
    print(f"[node] argus_vio_node started (pid {node_proc.pid}); warming up...")
    time.sleep(NODE_WARMUP_S)
    if node_proc.poll() is not None:
        node_log.close()
        sys.exit(f"argus_vio_node exited early; logs kept at {log_dir}")

    # ---- Recorder + bag playback --------------------------------------
    rclpy.init()
    rec = OdomRecorder()
    play_log = open(log_dir / "bag_play.log", "w")
    play_proc = subprocess.Popen(
        ["ros2", "bag", "play", str(bag_dir)],
        stdout=play_log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"[play] ros2 bag play started (pid {play_proc.pid}); recording {ODOM_TOPIC}...")

    t_start = time.monotonic()
    last_report = 0.0
    try:
        # Spin until playback finishes, then a short grace period.
        while play_proc.poll() is None:
            rclpy.spin_once(rec, timeout_sec=0.1)
            el = time.monotonic() - t_start
            if el - last_report >= 10.0:
                print(f"  [{el:6.1f}s] odometry messages: {len(rec.rows)}")
                last_report = el
        grace_end = time.monotonic() + GRACE_S
        while time.monotonic() < grace_end:
            rclpy.spin_once(rec, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        rows = rec.rows
        rec.destroy_node()
        rclpy.shutdown()
        # SIGINT the whole process group of each subprocess (ros2 run/play
        # spawn the real executable as a child), then escalate to SIGKILL.
        def stop(pr, sig):
            if pr.poll() is None:
                try:
                    os.killpg(os.getpgid(pr.pid), sig)
                except ProcessLookupError:
                    pass
        for pr in (play_proc, node_proc):
            stop(pr, signal.SIGINT)
        for pr in (play_proc, node_proc):
            try:
                pr.wait(timeout=10)
            except subprocess.TimeoutExpired:
                stop(pr, signal.SIGKILL)
        node_log.close()
        play_log.close()

    # ---- Save ----------------------------------------------------------
    n = len(rows)
    if n == 0:
        sys.exit(f"No odometry messages recorded; logs kept at {log_dir}")

    # Run succeeded — drop the temp logs.
    shutil.rmtree(log_dir, ignore_errors=True)

    t = np.array([r[0] for r in rows])
    position = np.array([r[1:4] for r in rows])
    orientation_wxyz = np.array([r[4:8] for r in rows])
    linear_velocity = np.array([r[8:11] for r in rows])
    angular_velocity = np.array([r[11:14] for r in rows])
    pose_cov = np.array([r[14] for r in rows]).reshape(n, 6, 6)
    twist_cov = np.array([r[15] for r in rows]).reshape(n, 6, 6)

    np.savez_compressed(
        out_npz,
        t=t, t_rel=t - t[0],
        position=position,
        orientation_wxyz=orientation_wxyz,
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        pose_covariance=pose_cov,
        twist_covariance=twist_cov,
        frame_id="global", child_frame_id="imu",
        odom_topic=ODOM_TOPIC,
    )

    dur = t[-1] - t[0]
    print("\n" + "=" * 60)
    print("VIO ODOMETRY")
    print("=" * 60)
    print(f"Messages:   {n}  over {dur:.1f}s  ({n / dur:.1f} Hz)")
    print(f"Start pos:  {position[0].round(3)} m")
    print(f"End pos:    {position[-1].round(3)} m")
    print(f"Path range: x[{position[:,0].min():.2f},{position[:,0].max():.2f}] "
          f"y[{position[:,1].min():.2f},{position[:,1].max():.2f}] "
          f"z[{position[:,2].min():.2f},{position[:,2].max():.2f}] m")
    print(f"\nSaved: {out_npz}")


if __name__ == "__main__":
    main()
