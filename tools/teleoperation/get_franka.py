import argparse
from frankapy import FrankaArm

if __name__ == '__main__':
    fa = FrankaArm(with_gripper=False)
    print(fa.get_joints())