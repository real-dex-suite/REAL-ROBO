# Real-Robo Teleoperation (RR-T)

Collecting tele-operation data with Real-Robo!

## Dependencies

- Ubuntu 20.04 / 22.04
- CUDA 11.8
- ROS noetic (necessary)
  - [Installing ROS 1 in Ubuntu 22.04 (Chinese)](https://www.bilibili.com/opus/890840405512290392)
- ROS humble (optional)

## Supported Devices

- Robot Arms
  - [x] JAKA
  - [x] Flexiv
  - [x] Franka (tested)
- Robot Hands
  - hand
    - [x] Leaphand
    - [x] Paxini
  - gripper
    - [x] Panda Gripper (tested)
- Tele-operation Devices
  - dexterous hand
    - [x] Mediapipe
  - arm + dexterous hand
    - [x] Leapmotion
    - [x] Oculus VR
    - [x] HAMER
  - arm + gripper
    - [x] PICO 4 (tested)
    - [ ] Keyboard
    - [ ] Spacemouse
    - [ ] HAMER
    - [ ] Meta Quest 3
    
## Installation

```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8

git clone --recurse-submodules git@github.com:real-dex-suite/REAL-ROBO.git -b hwfan-dev-genesis
conda create -n real-robo python=3.8
conda activate real-robo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e dependencies/curobo --no-build-isolation --verbose
pip install -e .
ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 $CONDA_PREFIX/lib/libffi.so.7
```

### Franka Control

```bash
unset ROS_DISTRO && source /opt/ros/noetic/local_setup.bash
pip install -e dependencies/frankapy
cd dependencies/frankapy && ./bash_scripts/make_catkin.sh
```

## Tele-operation

### Step 1: Run VR Streamer

#### Run on Docker (Recommended)

```bash
pushd vr/pico_streamer
# de-comment if no container exists
bash start_streaming_docker.sh # --init
popd
```

#### Run on Workstation (ROS 2 Installed)

- Dependencies

```bash
bash install.sh
```

- Run

```bash
pushd vr/pico_streamer
bash start_streaming_local.sh
popd
```

### Step 2: Run VR Publisher

```bash
bash pipelines/vr_bridge.sh 
```

### Step 3: Run Tele-operation

#### Simulation (Genesis)

```bash
conda activate real-robo
bash pipelines/teleop_sim.sh
```

#### Real (Franka)

```bash
conda activate real-robo
# Step 1: start franka daemon processes
bash pipelines/start_franka.sh
# Step 2: start teleop process
bash pipelines/teleop_real.sh
```

#### Real Data Recording

```bash
bash pipelines/franka_record.sh
```

#### Camera Calibration

Please refer to [Camera Calibration](https://github.com/kingchou007/camera-calibration) for more details.

## Acknowledgement

Real-Robo mainly borrows [Holo-dex](https://github.com/SridharPandian/Holo-Dex) framework. Thanks for their wonderful job!

Maintained by Jinzhou Li ([@kingchou007](https://github.com/kingchou007)) and Hongwei Fan ([@hwfan](https://github.com/hwfan)).