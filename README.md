# Real-Robo Teleoperation (RR-T)

Collecting tele-operation data with Real-Robo!

## Dependencies

- Ubuntu 20.04
- CUDA 11.8
- ROS noetic

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
    - [x] Ctek Gripper (tested)
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

git clone --recurse-submodules git@github.com:real-dex-suite/REAL-ROBO.git
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
cd dependencies/frankapy && ./bash_scripts/make_catkin.sh && cd ../..
```

## Tele-operation

### Step 1: Run VR Streamer

```bash
pushd vr/pico_streamer
# de-comment if no container exists
bash start_streaming_docker.sh # --init
popd
```

### Step 2: Run VR Publisher

```bash
bash pipelines/vr_bridge.sh 
```

### Step 3: Run Tele-operation

#### Visualize VR Control

```bash
conda activate real-robo
bash pipelines/vis_vr.sh
```

#### Simulation (Genesis)

```bash
conda activate real-robo
bash pipelines/teleop_sim.sh
```

#### Real (Franka)

!!! NOTE: Please change the settings in `dependencies/frankapy/bash_scripts/start_control_pc.sh` to your own.

!!! NOTE: Run `bash pipelines/start_franka.sh` again if you change the controlling yaml.

```bash
conda activate real-robo
# Step 1: start franka daemon processes
bash pipelines/start_franka.sh
# Step 2: start teleop process
bash pipelines/teleop_real.sh
```

#### Real Data Recording

```bash
bash pipelines/auto_record.sh
```

#### Camera Calibration

Please refer to [Camera Calibration](https://github.com/kingchou007/camera-calibration) for more details.

## Acknowledgement

Real-Robo mainly borrows [Holo-dex](https://github.com/SridharPandian/Holo-Dex) framework. For the dexterous hand retargeting, we refer to and modify [AnyTeleop](https://github.com/dexsuite/dex-retargeting). Please cite their work if you use this code in your research. Thanks for their wonderful job!

Maintained by Jinzhou Li ([@kingchou007](https://github.com/kingchou007)), Hongwei Fan ([@hwfan](https://github.com/hwfan)), Tianhao Wu ([@tianhaowu](https://github.com/tianhaowuhz)) and Jiyao Zhang ([@jiyao06](https://github.com/Jiyao06)).