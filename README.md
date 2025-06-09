# Real-Robo

Collecting tele-operation data with Real-Robo!

## Supported Devices

- Robot Arms
  - [x] JAKA
  - [x] Kinova
  - [x] Flexiv
  - [x] Franka (verified)
- Robot Hands
  - hand
    - [x] Allegro
    - [x] Leaphand
  - gripper
    - [x] Franka Gripper (verified)
- Tele-operation Devices
  - hand
    - [x] Mediapipe
    - [x] Leapmotion
    - [x] Oculus VR
  - gripper
    - [x] Spacemouse (verified)
    - [x] Keyboard (verified)
    - [x] PICO VR (verified)

## Installation

```bash
conda create -n real-robo python=3.10
conda activate real-robo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install genesis-world
pip install -r requirements.txt
cd $HOME
git clone https://github.com/NVlabs/curobo.git
cd curobo && pip install -e . --no-build-isolation && cd ..
ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 $CONDA_PREFIX/lib/libffi.so.7
```

## Tele-operation

### Real-world

```bash
python teleop.py
```

### Simulation (Genesis)
```bash
python teleop_sim.py
```

## Data Collection

```bash
bash franka_record.sh
```

## Acknowledgement

Real-Robo mainly borrows [Holo-dex](https://github.com/SridharPandian/Holo-Dex). Thanks for their wonderful job!

Maintained by Jinzhou Li ([@kingchou007](https://github.com/kingchou007)) and Hongwei Fan ([@hwfan](https://github.com/hwfan)).