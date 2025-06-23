import argparse
from frankapy import FrankaArm

# isaac sim
# DEFAULT_MOBILE_FRANKA = [-0.00950185, -0.11585174, -0.09009125, -2.25216384, -0.0030539, 2.14814552, 1.41996225]

# bread
DEFAULT_MOBILE_FRANKA = [-1.02085034, -0.15071283, 1.12373157, -2.49985287, 1.3068513, 3.24757076, 0.41839715]

if __name__ == '__main__':
    fa = FrankaArm(with_gripper=False)
    fa.goto_joints(DEFAULT_MOBILE_FRANKA)