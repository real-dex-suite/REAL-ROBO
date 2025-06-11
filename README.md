# Real-Robo

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
  - [x] Franka Research 3 (tested)
- Robot Hands
  - hand
    - [x] Leaphand
    - [x] paxini
  - gripper
    - [x] Franka Gripper (tested)
- Tele-operation Devices
  - hand
    - [x] Mediapipe
    - [x] Leapmotion
    - [x] Oculus VR
    - [x] HAMER
  - gripper
    - [x] Spacemouse
    - [x] Keyboard
    - [x] PICO 4 (tested)
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
# de-comment if this is the first time of running
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
bash pipelines/teleop_sim.sh
```

#### Real (Franka)

```bash
# Step 1: start franka daemon processes
bash pipelines/start_franka.sh
# Step 2: start teleop process
bash pipelines/teleop_real.sh
```

#### Real Data Recording

```bash
bash pipelines/franka_record.sh
```

## Acknowledgement

Real-Robo mainly borrows [Holo-dex](https://github.com/SridharPandian/Holo-Dex). Thanks for their wonderful job!

Maintained by Jinzhou Li ([@kingchou007](https://github.com/kingchou007)) and Hongwei Fan ([@hwfan](https://github.com/hwfan)).