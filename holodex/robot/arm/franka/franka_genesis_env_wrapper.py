#!/usr/bin/env python
import time
import numpy as np
# from holodex.robot.arm.franka.kinematics_solver import FrankaSolver
from kinematics_solver import FrankaSolver
from scipy.spatial.transform import Rotation as R
import os
os.environ['MUJOCO_GL'] = 'glx'
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import genesis as gs
import torch

def design_scene():
    genesis_scene_path_dict = {
        "robot": "/home/jinzhou/Lab/REAL-ROBO/holodex/robot/arm/franka/fr3/fr3.urdf",
    }
    
    scene = gs.Scene(
        viewer_options = gs.options.ViewerOptions(
            camera_pos    = (0, -3.5, 2.5),
            camera_lookat = (0.0, 0.0, 0.5),
            camera_fov    = 30,
            max_FPS       = 60,
        ),
        sim_options = gs.options.SimOptions(
            dt = 0.01,
        ),
        show_viewer = True,
    )
    
    franka = scene.add_entity(
        gs.morphs.URDF(
            file  = genesis_scene_path_dict["robot"],
            pos   = (0, 0, 0),
            quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
            scale = 1.0,
            merge_fixed_links=False,
            fixed=True,
            
        ),
        material=gs.materials.Rigid(gravity_compensation=1.0),
    )
    genesis_scene_dict = {
        "robot": franka,
    }
    return scene, genesis_scene_dict


class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha 
        self.prev_value = None
    
    def __call__(self, new_value):
        if self.prev_value is None:
            self.prev_value = new_value
        else:
            self.prev_value = self.alpha * new_value + (1 - self.alpha) * self.prev_value
        return self.prev_value
    
class FrankaGenesisEnvWrapper:
    def __init__(self):
        gs.init(backend=gs.gpu, logging_level = 'warning')
        scene, genesis_scene_dict = design_scene()
        self.arm = genesis_scene_dict["robot"]
        self.ik_solver = FrankaSolver()
        self.low_pass_filter = LowPassFilter(0.2) # higher meaning more smoothing
        scene.build(n_envs=1)
        jnt_names = [
            'fr3_joint1',
            'fr3_joint2',
            'fr3_joint3',
            'fr3_joint4',
            'fr3_joint5',
            'fr3_joint6',
            'fr3_joint7',
            "fr3_finger_joint1",
            "fr3_finger_joint2",
        ]
        dofs_idx = [self.arm.get_joint(name).dof_idx_local for name in jnt_names]
        self.jnt_names = jnt_names
        self.dofs_idx = dofs_idx
        self.scene = scene
        self._initialize_state()
        self._initialize_joint_control_config()
        
    def _initialize_state(self):
        # Initialize the current joint state and end-effector pose
        self.current_joint_state = self.arm.get_dofs_position(dofs_idx_local=self.dofs_idx)[0]
        self.joint_state = self.current_joint_state[:]
        ee_link = self.arm.get_link(name="fr3_link8")
        self.current_ee_pose = torch.cat([ee_link.get_pos(), ee_link.get_quat()], 1)[0]
        self.ee_pose = self.current_ee_pose

    def _initialize_joint_control_config(self):
        # This configuration is calibrated between the real robot and simulation parameters
        self.arm.control_dofs_position(
            self.current_joint_state.unsqueeze(0)
        )

    def get_arm_position(self):
        # Get the current joint positions of the arm
        self.current_joint_state = self.arm.get_dofs_position(dofs_idx_local=self.dofs_idx)[0]
        return self.current_joint_state

    def ee2joint(self, ee_pose):
        # Convert end-effector pose to joint positions using inverse kinematics
        np.set_printoptions(precision=4, suppress=True)
        print("ee_pose", ee_pose)
        print("cur_ee_pose", self.get_tcp_position())
        ik_res = self.ik_solver.solve_ik(ee_pose[:3], ee_pose[3:])
        return ik_res

    def open_gripper(self):
        # Open the robot's gripper
        self.arm.control_dofs_position(
            [0.4, 0.4],
            dofs_idx_local=self.dofs_idx[7:9],
        )
        
    def close_gripper(self):
        # Close the robot's gripper
        self.arm.control_dofs_position(
            [0.0, 0.0],
            dofs_idx_local=self.dofs_idx[7:9],
        )
        
    def move(self, target_pose):
        target_joint = self.ee2joint(target_pose)
        current_joint = self.get_arm_position()
        max_step = 1

        delta = target_joint - current_joint

        # Clip all joints (not just 1 & 2)
        delta_clipped = delta.copy()  # Create a copy to modify
        for i in range(7):  # For all 7 joints
            delta_clipped[i] = np.clip(delta[i], -max_step, max_step)

        safe_joint = current_joint + delta_clipped

        # Apply filtering (if needed)
        # Initialize filters if not done already
        if not hasattr(self, 'joint_filters'):
            self.joint_filters = [LowPassFilter(alpha=0.5) for _ in range(7)]

        filtered_joint = np.zeros_like(safe_joint)
        for i in range(7):
            filtered_joint[i] = self.joint_filters[i](safe_joint[i])  # Apply filter

        print("target_joint:", target_joint)
        print("filtered_joint:", filtered_joint)
        self.move_joint(filtered_joint, self.dofs_idx[:7])
        time.sleep(0.1)

    def move_joint(self, target_joint, dof_idxs): #! double check the type of target_joint
        # Move the robot's joints to the target positions
        self.arm.control_dofs_position(
            target_joint,
            dofs_idx_local=dof_idxs,
        )

    def get_tcp_position(self):
        """
        Get the TCP position of the robot
        return:
            Translation: [x, y, z]
            Quaternion: [w, x, y, z]
        """
        # Retrieve the current end-effector pose and return it as a concatenated array
        ee_link = self.arm.get_link(name="fr3_link8")
        ee_pose = torch.cat([ee_link.get_pos(), ee_link.get_quat()], 1)[0]
        return ee_pose

    def run(self):
        while True:
            self.scene.step()
        # Main loop to move the robot and open the gripper
        # rate = rospy.Rate(10)  # test in 10 Hz
        # for i in range(11):
        #     if rospy.is_shutdown():
        #         break
        #     x = self.get_arm_position()
        #     x[6] += 0.05
        #     x[5] += 0.05
        #     x[4] += 0.05
        #     x[3] += 0.05
        #     x[2] += 0.05
        #     x[1] += 0.05
        #     x[0] += 0.05
        #     self.move_joint(x)
        #     print(self.current_joint_state)
        #     rate.sleep()
        # self.open_gripper()
        pass

    def shutdown(self):
        # Placeholder for shutdown procedures
        pass

if __name__ == '__main__':
    controller = FrankaGenesisEnvWrapper()
    controller.run()
        
    # try:
    #     controller = FrankaEnvWrapper()
    #     controller.run()

    #     # a = controller.get_tcp_position()
    #     # b = controller.ee2joint(a)
    #     # c = controller.get_arm_position()

    #     # from termcolor import cprint
    #     # cprint(f'original_pose: {a}', 'red')
    #     # cprint(f'solved_joint: {b}', 'blue')
    #     # cprint(f'actual_joint: {c}', 'green')

    #     # for i in range(100):
    #     #     np.set_printoptions(precision=4, suppress=True)
    #     #     print(f"tcp pose: {controller.get_tcp_position()}")



    # except rospy.ROSInterruptException:
    #     pass
    # finally:
    #     controller.shutdown()
