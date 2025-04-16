# REAL-ROBOT

# Real Robot Control

This repository contains code for controlling real robot arms, including the Flexiv Rizon4 robot.

## Features

- Direct control of Flexiv Rizon4 / Franka robot arm
- Joint state and end-effector pose publishing
- Cartesian motion control with force limits
- Thread-safe movement queue system
- ROS integration for robot state monitoring
- Teleop control

Tested on Ubuntu 20.04


## Installation

```bash
conda activate frankapy
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
cd $HOME
git clone https://github.com/NVlabs/curobo.git
cd curobo && pip install -e . --no-build-isolation && cd ..
ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 $CONDA_PREFIX/lib/libffi.so.7
```