import argparse
from frankapy import FrankaArm

if __name__ == '__main__':
    print('Starting robot')
    fa = FrankaArm()
    print("start ok")
    fa.goto_joints([0.0405, -0.0053, -0.1853, -2.2179, -0.0057, 2.2234, 0.6904])