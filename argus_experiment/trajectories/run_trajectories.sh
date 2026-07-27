#!/usr/bin/env bash
#
# Run a queue of EE trajectories on the YAM arm.
#
# First argument is the trial name, used to title the output data folder
# (recordings/<trial_name>_<timestamp>/).
#
# Edit the --traj lines below to set up the queue. Each is:
#     --traj WAVE:MOTION:AXIS:SPEED
#   WAVE   : sinusoidal | sawtooth
#   MOTION : linear  (AXIS x/y/z,        SPEED m/s)
#            angular (AXIS roll/pitch/yaw, SPEED rad/s)
#   SPEED  : peak for sinusoidal, constant for sawtooth
#
# Extra flags (--sim, --periods N, --dt S) are forwarded via "$@", e.g.:
#     bash argus_experiment/trajectories/run_trajectories.sh TRIAL_NAME --sim --periods 3 --dt 0.01
#
# Before EVERY new trajectory sequence, test in sim
#     bash argus_experiment/trajectories/run_trajectories.sh TRIAL_NAME --sim
#
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 TRIAL_NAME [--sim] [--periods N] [--dt S] ..." >&2
    exit 1
fi
TRIAL_NAME="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# demo movements, will eventually load full sequence
# python "$SCRIPT_DIR/run_trajectories.py" "$TRIAL_NAME" \
#     --traj sawtooth:linear:x:0.1 \
#     --traj sawtooth:linear:z:0.1 \
#     "$@"


python "$SCRIPT_DIR/run_trajectories.py" "$TRIAL_NAME" \
    --traj sawtooth:angular:roll:0.2 \
    --traj sawtooth:angular:pitch:0.2 \
    --traj sawtooth:angular:yaw:0.2 \
    "$@"

# python "$SCRIPT_DIR/run_trajectories.py" "$TRIAL_NAME" \
#     --traj sinusoidal:angular:roll:0.2 \
#     --traj sinusoidal:angular:pitch:0.2 \
#     --traj sinusoidal:angular:yaw:0.2 \
#     --traj sinusoidal:linear:x:0.1 \
#     --traj sinusoidal:linear:z:0.1 \
#     "$@"