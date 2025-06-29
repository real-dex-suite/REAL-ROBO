import argparse
from frankapy import FrankaArm

DEFAULT_FR3 = [0.040505826194643726, -0.00536819607860028,         -0.18538284760651613,     -2.217976190474148,   -0.005728349209802673,                          2.223454317248339,         0.6904711141123535,  ]

if __name__ == '__main__':
    fa = FrankaArm()
    fa.goto_joints(DEFAULT_FR3)
    fa.open_gripper()