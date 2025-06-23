import argparse
from frankapy import FrankaArm

DEFAULT_FR3 = [0.040505826194643726, -0.00536819607860028,         -0.18538284760651613,     -2.217976190474148,   -0.005728349209802673,                          2.223454317248339,         0.6904711141123535,  ]
MILK_FR3 = [-0.481852,0.85785228,-0.19883735,-1.72379066,1.09372448,1.15488531,-1.89647886]
STACKING_FR3 = [-0.21798363194007797, 0.5657058928603688, -0.16164596636425474, -2.0364824735370632, 0.11219769010436825, 2.6176562296111725, 0.3161291930273689]
STANDING_OREO_FR3 = [-1.0775517259122251, 0.9176085787135911, 0.496674303078926, -1.7921937051848211, 0.7358905435749714, 1.5004150911871992, -1.6959823819401603]
LYING_OREO_FR3 = [-1.2798323565617922, 0.772613130402114, 0.7738525342811828, -2.0967133381855505, -0.9179348109834208, 2.502731935276255, -0.6262284250197506]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="milk",
    )
    args = parser.parse_args()

    fa = FrankaArm()
    fa.goto_joints(DEFAULT_FR3)
    if args.task == "milk":
        fa.goto_joints(MILK_FR3)
        fa.close_gripper()
    elif args.task == "stacking":
        fa.goto_joints(STACKING_FR3)
        fa.open_gripper()
    elif args.task == "standing_oreo":
        fa.goto_joints(STANDING_OREO_FR3)
        fa.close_gripper()
    elif args.task == "lying_oreo":
        fa.goto_joints(LYING_OREO_FR3)
        fa.close_gripper()
    else:
        pass