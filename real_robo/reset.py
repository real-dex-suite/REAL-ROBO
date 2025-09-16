#!/usr/bin/env python

"""Moving robot joint positions to initial pose for starting new experiments."""
import argparse
from pathlib import Path
import numpy as np
from pathlib import Path

import time
import sys
from pathlib import Path

sys.path.append("dependencies/deoxys_control_research3/deoxys")


from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from real_robo_logger import get_real_robo_logger

logger = get_real_robo_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument(
        "--controller-cfg", type=str, default="joint-position-controller.yml"
    )
    parser.add_argument(
        "--folder", type=Path, default="data_collection_example/example_data"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    robot_interface = FrankaInterface(
        config_root + f"/{args.interface_cfg}", use_visualizer=True
    )
    controller_cfg = YamlConfig(config_root + f"/{args.controller_cfg}").as_easydict()

    controller_type = "JOINT_POSITION"

    # Golden resetting joints
  
    time.sleep(3)
    # x = robot_interface.last_eef_pose
    # b = robot_interface.last_eef_quat_and_pos
    state = robot_interface._state_buffer[-1]
    # print(x)
    # print(b)
    # print(d)
    # print(f.name for f in state.DESCRIPTOR.fields)


    # O_T_EE [0.9996502965139658, 0.008449335558703863, 0.02505711834220238, 0.0, 0.008402197919113286, -0.9999627118320544, 0.0019858970441198383, 0.0, 0.02507296437688722, -0.001774667762151065, -0.9996840485931989, 0.0, 0.45654878942025484, 0.030733401139585705, 0.26551665659101, 1.0]
    # O_T_EE_d [0.9996505567868075, 0.008450568023620764, 0.025046316842259133, 0.0, 0.008403440500042436, -0.9999627006417776, 0.001986274004995802, 0.0, 0.025062168632149384, -0.001775104742455333, -0.999684318525907, 0.0, 0.45654426849761276, 0.03073313881985854, 0.2655137489446315, 1.0]
    # F_T_EE [0.7071067690849304, -0.7071067690849304, 0.0, 0.0, 0.7071067690849304, 0.7071067690849304, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.10339999943971634, 1.0]
    # F_T_NE [0.7071067690849304, -0.7071067690849304, 0.0, 0.0, 0.7071067690849304, 0.7071067690849304, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.10339999943971634, 1.0]
    # NE_T_EE [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    # EE_T_K [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    # m_ee 0.7300000190734863
    # I_ee [0.0010000000474974513, 0.0, 0.0, 0.0, 0.0024999999441206455, 0.0, 0.0, 0.0, 0.0017000000225380063]
    # F_x_Cee [-0.009999999776482582, 0.0, 0.029999999329447746]
    # I_load [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # F_x_Cload [0.0, 0.0, 0.0]
    # m_total 0.7300000190734863
    # I_total [0.0010000000474974513, 0.0, 0.0, 0.0, 0.0024999999441206455, 0.0, 0.0, 0.0, 0.0017000000225380063]
    # F_x_Ctotal [-0.009999999776482582, 0.0, 0.029999999329447746]
    # elbow [-0.022568422402161014, -1.0]
    # elbow_d [-0.022569448404257297, -1.0]
    # elbow_c [0.0, 0.0]
    # delbow_c [0.0, 0.0]
    # ddelbow_c [0.0, 0.0]
    # tau_J [-0.06153404712677002, -22.41485023498535, -0.20317772030830383, 21.037290573120117, 0.8386210799217224, 2.075812578201294, 0.12339316308498383]
    # tau_J_d [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # dtau_J [-29.177114486694336, -3.6768569946289062, 35.49861145019531, -69.8550033569336, 15.674836158752441, -9.532405853271484, -57.969993591308594]
    # q [0.09097236858695062, -0.20009279594686075, -0.022568422402161014, -2.474346402288704, -0.010692679240323128, 2.299169635253515, 0.8527876874142343]
    # q_d [0.09097340334051673, -0.2000965549043825, -0.022569448404257297, -2.474355838297549, -0.010692606915410114, 2.2991645140314927, 0.852786471830017]
    # dq [-0.00020635227995610027, 0.0010415339271161084, 0.0009387333670054788, 0.0004951875282955764, 0.0006189058704365216, -0.0008853995478074071, 0.0016028933746694172]
    # dq_d [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # ddq_d [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # joint_contact [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # cartesian_contact [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # joint_collision [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # cartesian_collision [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # tau_ext_hat_filtered [-0.050295471011176006, -0.4869234298364684, -0.13227754665094327, -0.1438721002387308, 0.19537230375775275, -0.12532321783763753, 0.1340741485184134]
    # O_F_ext_hat_K [-1.8742887318734651, -0.024900905887675216, 0.9503810421157509, 0.19391256745386237, -1.0960225577925735, -0.07808235965994069]
    # K_F_ext_hat_K [-1.8500299357610015, 0.01103919177593049, -0.9970305516598347, 0.15353252930702127, 0.16554769814536638, 0.1285336024287196]
    # O_dP_EE_d [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # O_T_EE_c [0.9996505567868075, 0.008450568023620764, 0.025046316842259133, 0.0, 0.008403440500042436, -0.9999627006417776, 0.001986274004995802, 0.0, 0.025062168632149384, -0.001775104742455333, -0.999684318525907, 0.0, 0.45654426849761276, 0.03073313881985854, 0.26551374894463153, 1.0]
    # O_dP_EE_c [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # O_ddP_EE_c [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # theta [0.09096826612949371, -0.2017057090997696, -0.022584673017263412, -2.472770929336548, -0.01059405505657196, 2.2994134426116943, 0.8528022766113281]
    # dtheta [0.0, 0.0, 0.0, -0.025956861674785614, 0.03898753225803375, 0.0, 0.0]
    # current_errors 
    # last_motion_errors 
    # robot_mode 1
    # time toSec: 7941.403
    # toMSec: 7941403

    for field, value in state.ListFields():
        print(field.name, value)

    
    from ipdb import set_trace; set_trace()
    
    reset_joint_positions = [
        0.09162008114028396,
        -0.19826458111314524,
        -0.01990020486871322,
        -2.4732269941140346,
        -0.01307073642274261,
        2.30396583422025,
        0.8480939705504309,
    ]

    # This is for varying initialization of joints a little bit to
    # increase data variation.
    reset_joint_positions = [
        e + np.clip(np.random.randn() * 0.005, -0.005, 0.005)
        for e in reset_joint_positions
    ]
    action = reset_joint_positions + [-1.0]

    while True:
        if len(robot_interface._state_buffer) > 0:
            logger.info(f"Current Robot joint: {np.round(robot_interface.last_q, 3)}")
            logger.info(f"Desired Robot joint: {np.round(robot_interface.last_q_d, 3)}")

            if (
                np.max(
                    np.abs(
                        np.array(robot_interface._state_buffer[-1].q)
                        - np.array(reset_joint_positions)
                    )
                )
                < 1e-3
            ):
                break
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )
    robot_interface.close()


if __name__ == "__main__":
    main()
