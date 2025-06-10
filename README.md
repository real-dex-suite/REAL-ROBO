# Real-Robo

Collecting tele-operation data with Real-Robo!

## Dependencies

- Ubuntu 22.04
- CUDA 11.8
- ROS 1 and ROS 2 ([Installing ROS 1 in Ubuntu 22.04 (Chinese)](https://www.bilibili.com/opus/890840405512290392))
- Teleoperation device (optional)

## Supported Devices

- Robot Arms
  - [x] JAKA
  - [x] Kinova
  - [x] Flexiv
  - [x] Franka (tested)
- Robot Hands
  - hand
    - [x] Allegro
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
    - [x] Spacemouse (tested)
    - [x] Keyboard (tested)
    - [x] PICO VR (tested)
    - [x] HAMER (tested)
    
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
# ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 $CONDA_PREFIX/lib/libffi.so.7
```

For controlling real robot, please follow [Frankapy](https://iamlab-cmu.github.io/frankapy/install.html) documents for installing frankapy.

## Tele-operation

Follow [pico_streamer](vr/pico_streamer) first if use PICO VR.

```bash
unset ROS_DISTRO && source /opt/ros/noetic/local_setup.bash
python teleop.py
```

## Real Data Recording

```bash
bash franka_record.sh
```

## Acknowledgement

Real-Robo mainly borrows [Holo-dex](https://github.com/SridharPandian/Holo-Dex). Thanks for their wonderful job!

Maintained by Jinzhou Li ([@kingchou007](https://github.com/kingchou007)) and Hongwei Fan ([@hwfan](https://github.com/hwfan)).