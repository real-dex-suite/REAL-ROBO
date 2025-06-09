# REAL-ROBOT

This repository contains code for controlling real robot arms / hands.

## Supported Devices

- [ ]

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

## Teleop

```bash
python teleop_sim.py
```