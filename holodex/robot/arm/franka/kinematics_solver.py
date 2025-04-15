import time
import torch
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from termcolor import cprint
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState


# Enable PyTorch performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Constants #TODO: add it to config 
JS_NAMES = ["panda_joint1","panda_joint2","panda_joint3","panda_joint4", "panda_joint5",
    "panda_joint6","panda_joint7"]

class FrankaSolver:
    def __init__(self, ik_type="ik_solver"):
        """Initialize the Franka IK Solver."""
        self.tensor_args = TensorDeviceType()
        config_file = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))

        urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
        base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
        ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]

        robot_cfg = RobotConfig.from_basic(
            urdf_file, base_link, ee_link, self.tensor_args
        )
        

        self.kin_model = CudaRobotModel(robot_cfg.kinematics)

        # IK solver config
        if ik_type == "ik_solver":
            self.ik_config = IKSolverConfig.load_from_robot_config(
                robot_cfg,
                None,
                rotation_threshold=0.05,
                position_threshold=0.005,
                num_seeds=100,
                self_collision_check=False,
                self_collision_opt=False,
                tensor_args=self.tensor_args,
                use_cuda_graph=True,
            )
            self.ik_solver = IKSolver(self.ik_config)
        elif ik_type == "motion_gen":
            self.plan_config = MotionGenConfig.load_from_robot_config(
                robot_cfg,
                None,
                tensor_args=self.tensor_args,
                interpolation_dt=0.02,
                ee_link_name="ee_link",
            )
            self.motion_gen = MotionGen(self.plan_config)
            cprint("warming up motion gen solver", "green")

            # If constraints are needed, uncomment the following lines
            # pose_cost_metric = PoseCostMetric(
            #     hold_partial_pose=True,
            #     hold_vec_weight=self.tensor_args.to_device([1, 1, 1, 0, 1, 0]),
            # )
            # plan_config.pose_cost_metric = pose_cost_metric

            self.motion_gen.warmup(warmup_js_trajopt=False)
            self.plan_config_temp = MotionGenPlanConfig(
                enable_graph=False,
                enable_graph_attempt=4,
                max_attempts=2,
                enable_finetune_trajopt=True,
                time_dilation_factor=0.5,
            )
        else:
            raise ValueError(f"Unsupported IK type: {ik_type}")


    def solve_ik_by_motion_gen(self, curr_joint_state, target_trans, target_quat):
         # motion generation:

        cu_js = JointState(
            position=self.tensor_args.to_device(curr_joint_state),
            velocity=self.tensor_args.to_device(curr_joint_state) * 0.0,
            acceleration=self.tensor_args .to_device(curr_joint_state) * 0.0,
            jerk=self.tensor_args .to_device(curr_joint_state) * 0.0,
            joint_names=JS_NAMES,
        )
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        # compute curobo solution:
        ik_goal = Pose(
            position=self.tensor_args.to_device(target_trans),
            quaternion=self.tensor_args.to_device(target_quat),
        )
        result = self.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, self.plan_config_temp)
        succ = result.success.item()  
        if succ:
            print("success")
            motion_plan = result.get_interpolated_plan()
            motion_plan = self.motion_gen.get_full_js(motion_plan)
            motion_plan = motion_plan.get_ordered_joint_state(JS_NAMES)

            return motion_plan.position.cpu().numpy().tolist()

        else:
            print("failed")
            return None

    def solve_ik(self, target_position, target_quaternion):
        """
        Solve a single IK problem for the Franka robot.

        Args:
            target_position (list): Target position [x, y, z].
            target_quaternion (list): Target quaternion [w, x, y, z].

        Returns:
            list or None: Solution (7 joint angles) if successful, otherwise None.
        """
        # Pre-allocate tensors on GPU to avoid repeated allocations
        if not hasattr(self, 'pos_tensor_buffer'):
            self.pos_tensor_buffer = torch.zeros(3, device=self.tensor_args.device, dtype=torch.float32)
            self.quat_tensor_buffer = torch.zeros(4, device=self.tensor_args.device, dtype=torch.float32)
        
        # Copy data to pre-allocated tensors instead of creating new ones
        self.pos_tensor_buffer.copy_(torch.tensor(target_position, dtype=torch.float32))
        self.quat_tensor_buffer.copy_(torch.tensor(target_quaternion, dtype=torch.float32))
        
        # Create pose using the buffer tensors
        goal = Pose(self.pos_tensor_buffer, self.quat_tensor_buffer)

        # TODO: add interpolation for the goal pose
        
        # Solve IK
        result = self.ik_solver.solve(goal)
        
        # Only synchronize if needed
        if result.success.item():
            # Access solution directly without copying to CPU if possible
            solution = result.solution.cpu().numpy().flatten()
            return solution
        return None
    

    def compute_fk(self, joint_angles):
        """
        Compute the forward kinematics for the Franka robot, returning only position and quaternion.

        Args:
            joint_angles: Tensor of joint angles [q1, q2, ..., q7]

        Returns:
            position: End-effector position [x, y, z]
            quaternion: End-effector quaternion [w, x, y, z]
        """
        joint_angles = torch.tensor(joint_angles, device=self.tensor_args.device, dtype=torch.float32)
        out = self.kin_model.get_state(joint_angles)
        position = out.ee_position.cpu().numpy().flatten()
        quaternion = out.ee_quaternion.cpu().numpy().flatten()
        return position, quaternion

if __name__ == "__main__":
    target_position = [0.5219848708, -0.0764696627, 0.2757705941]
    target_quaternion = [0.0019550403, 0.9997118148, -0.0235102575, 0.0044412223]
    fanka_solver = FrankaSolver()

    for _ in range(10):
        start_time = time.time()
        joint_solution = fanka_solver.solve_ik(target_position, target_quaternion)
        end_time = time.time()
        print(f"Time taken for this solve: {end_time - start_time:.6f} seconds")

    actual_joints = [
        0.0399413690,
        -0.0059193035,
        -0.1844682830,
        -2.2195858055,
        -0.0049009859,
        2.2230507255,
        0.6908514895,
    ]
    print("actual_joints:", actual_joints)
    print("solved_joints:", joint_solution)

    old_pose = torch.cat((torch.tensor(target_position), torch.tensor(target_quaternion)))

    print("original pose", old_pose)
    trans, quat = fanka_solver.compute_fk(joint_solution)
    
    print("solved pose", trans, quat)
