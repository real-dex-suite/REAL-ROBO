# Real-Robo Teleoperation (RR-T)

Collecting tele-operation data with Real-Robo!

![RR-T](./docs/data/example.gif)

## Supported Devices
- Robot Arms
  - [x] Flexiv (tested)
  - [x] Franka (tested)
- Robot Hands
  - hand
    - [x] Leaphand (tested)
    - [ ] Insipred Hand
    - [ ] Zhiyuan Hand
  - gripper
    - [x] Panda Gripper (tested)
    - [ ] Robotiq Gripper
    - [x] Ctek Gripper (tested)
- Tele-operation Devices
    - [x] Meta Oculus VR
    - [x] Realsense with hand tracking
    - [ ] Glove
    - [ ] Motion Capture
  - arm + gripper
    - [x] PICO 4 (tested)
    - [ ] Keyboard
    - [ ] Spacemouse
    - [ ] HAMER
    - [ ] Meta Quest 3
    
## Setup

- Please refer to [Setup](./docs/setup.md) for more details.
- Please refer to [Usage](./docs/usage.md) for more details.
- Please refer to [Data Collection](./docs/data_collection.md) for more details.
- For camera calibration, please see the [Camera Calibration Guide](https://github.com/real-dex-suite/REAL-ROBO/blob/main/tools/calibration/CAMERA_CALI.md). You may need to calibrate the camera if you plan to use point cloud data.

## TODO

- [ ] Add more devices (e.g. motion capture, skeleton, etc.)
- [ ] Add better documentation
- [ ] Add more examples
- [ ] Add more tutorials

## Citation

If you find this code useful, please cite our paper:

```bibtex
```

## Acknowledgement

Real-Robo mainly borrows [Holo-dex](https://github.com/SridharPandian/Holo-Dex) framework. For the dexterous hand retargeting, we refer to and modify [AnyTeleop](https://github.com/dexsuite/dex-retargeting). Please cite their work if you use this code in your research. Thanks for their wonderful job!

Maintained by:
- Jinzhou Li ([kingchou007](https://github.com/kingchou007))
- Hongwei Fan ([hwfan](https://github.com/hwfan))
- Tianhao Wu ([tianhaowu](https://github.com/tianhaowuhz))
- Jiyao Zhang ([jiyao06](https://github.com/Jiyao06))

We thank them for their contributions.
