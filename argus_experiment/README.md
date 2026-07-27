Instructions for running the robot arm trajectories.

## Setup
```bash
conda activate yam_arms
```

Make sure the arm is powered on (power brick light is lit and E-stop is open)!

Bring up the CAN interface before running any arm commands:

```bash
sudo ip link set can0 up type can bitrate 1000000
```

To verify:

```bash
ip link show can0
# Should show state UP
```

If the adapter is unresponsive, reset it:

```bash
sudo bash robots_realtime/dependencies/i2rt/scripts/reset_all_can.sh
```

Now, ensure the ARGUS and MoCap data collections are running.

## Run trial
```bash
bash argus_experiment/trajectories/run_trajectories.sh TRIAL_NAME
```

Trial ends automatically. End the ARGUS and MoCap data collection.

Trial name, parameters and arm information should be saved under ~/ArgusArmExperiment/argus_experiment/trajectories/recordings