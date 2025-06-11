import argparse
from frankapy import FrankaArm

DEFAULT_MOBILE_FRANKA = [-0.00950185, -0.11585174, -0.09009125, -2.25216384, -0.0030539, 2.14814552, 1.41996225]

if __name__ == '__main__':
    fa = FrankaArm(with_gripper=False)
    fa.goto_joints(DEFAULT_MOBILE_FRANKA)