#!/usr/bin/env python
import numpy as np
try:
    from holodex.robot.arm.franka.kinematics_solver import FrankaSolver
except:
    from kinematics_solver import FrankaSolver
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64MultiArray, Bool
import rospy

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
    def __init__(self, control_mode="joint", teleop=False, gripper="panda"):
        assert gripper in ["panda"] or gripper is None, f"Gripper {gripper} is not supported for FrankaGenesisWrapper."
        rospy.init_node('genesis_tele', anonymous=True)
        rospy.sleep(1.0)

        self.ik_solver = FrankaSolver(ik_type="motion_gen", ik_sim=True, simulator="genesis")
        self.dof = 7
        self.with_gripper = gripper is not None
        if self.with_gripper:
            self.dof += 1
        self.current_joint_state = None
        self.current_ee_state = None
        self.joint_control_pub = rospy.Publisher(
            "/genesis/joint_control",
            Float64MultiArray,
            queue_size=1,
        )
        self.gripper_control_pub = rospy.Publisher(
            "/genesis/gripper_control",
            Bool,
            queue_size=1,
        )
        self.joint_state_sub = rospy.Subscriber(
            "/genesis/joint_states",
            Float64MultiArray,
            self._callback_current_joint_state,
            queue_size=1,
        )
        self.ee_state_sub = rospy.Subscriber(
            "/genesis/ee_states",
            Float64MultiArray,
            self._callback_current_ee_state,
            queue_size=1,
        )

        self._initialize_state()
        
    def _initialize_state(self):
        if self.current_joint_state is None:
            msg = rospy.wait_for_message('/genesis/joint_states', Float64MultiArray, timeout=5.0)
            self.current_joint_state = msg.data
        if self.current_ee_state is None:
            msg = rospy.wait_for_message('/genesis/ee_states', Float64MultiArray, timeout=5.0)
            self.current_ee_state = msg.data
        
    def _callback_current_joint_state(self, msg):
        self.current_joint_state = msg.data

    def _callback_current_ee_state(self, msg):
        self.current_ee_state = msg.data

    def eulerZYX2quat(self, euler, degree=False):
        if degree:
            euler = np.radians(euler)

        tmp_quat = R.from_euler("xyz", euler).as_quat().tolist()
        quat = [tmp_quat[3], tmp_quat[0], tmp_quat[1], tmp_quat[2]]
        return quat

    def get_arm_position(self):
        # Get the current joint positions of the arm
        return np.array(self.current_joint_state)

    def get_tcp_position(self):
        """
        Get the TCP position of the robot
        return:
            Translation: [x, y, z]
            Quaternion: [w, x, y, z]
        """
        # Retrieve the current end-effector pose and return it as a concatenated array
        return np.array(self.current_ee_state)
    
    def get_gripper_position(self):
        if self.with_gripper:
            return self.current_joint_state[7]
        else:
            raise RuntimeError("No gripper equipped in Franka. get_gripper_position should not work.")
    
    def home_robot(self):
        pass
    
    def open_gripper(self):
        if self.with_gripper:
            # Open the robot's gripper
            gripper_msg = Bool(data=True)
            self.gripper_control_pub.publish(gripper_msg)
        else:
            raise RuntimeError("No gripper equipped in Franka. open_gripper should not work.")
        
    def close_gripper(self):
        if self.with_gripper:
            # Open the robot's gripper
            gripper_msg = Bool(data=False)
            self.gripper_control_pub.publish(gripper_msg)
        else:
            raise RuntimeError("No gripper equipped in Franka. close_gripper should not work.")
        
    def move_gripper(self, gripper_cmd):
        """
        Control gripper for teleoperation with binary open/close command.
        Includes debouncing to avoid too frequent control commands.
        
        Args:
            gripper_cmd (float or int): Binary command for gripper
                - Values <= 0.05: Close the gripper
                - Values > 0.05: Open the gripper
                
        """
        if self.with_gripper:
            # Initialize state tracking if not already set
            if not hasattr(self, '_gripper_state'):
                self._gripper_state = None
                
            # Debounce logic - only send commands when state actually changes
            if float(gripper_cmd) > 0.05:
                self.open_gripper()
                # Open gripper command
                if self._gripper_state != 'open':
                    self._gripper_state = 'open'
            else:
                self.close_gripper()
                # Close gripper command
                if self._gripper_state != 'closed':
                    self._gripper_state = 'closed'
        else:
            raise RuntimeError("No gripper equipped in Franka. move_gripper should not work.")
          
        
    def reset(self):
        """
        This function is used to reset the robot to the home position from the frankapy.
        """
        pass
            

    def solve_ik(self, ee_pose: list) -> list:
        """
        Solve inverse kinematics.

        Args:
            ee_pose (list): The end-effector pose in the form [x, y, z, qx, qy, qz, qw].

        Returns:
            list: The joint positions that achieve the desired end-effector pose.

        Raises:
            ValueError: If no IK solution found
        """
        
        ik_res = self.ik_solver.solve_ik_by_motion_gen(
            self.get_arm_position(), ee_pose[:3], ee_pose[3:]
        )
        if ik_res is None:
            return None
        ik_res = np.array(ik_res[-1])
        return ik_res

    def move_joint(self, target_joint): #! double check the type of target_joint
        if target_joint is not None:
            joint_pos_msg = Float64MultiArray(data=target_joint)
            self.joint_control_pub.publish(joint_pos_msg)
            
    def move_joint_ik(self, target_ee: list):
        """
        Move joints to target positions.

        Args:
            target_joint (list): The target joint position for the robot.

        Returns:
            None
        """
        target_joint = self.solve_ik(target_ee)
        self.move_joint(target_joint)
        
    def move(self, target_cmd):
        self.move_joint_ik(target_cmd[:7])
        if self.with_gripper:
            self.move_gripper(target_cmd[7])
        
    def run(self):
        pass

    def shutdown(self):
        # Placeholder for shutdown procedures
        pass

if __name__ == '__main__':
    controller = FrankaGenesisEnvWrapper()
    controller.run()