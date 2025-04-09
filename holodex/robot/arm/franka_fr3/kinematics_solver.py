import time
import torch
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

# Enable PyTorch performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class FrankaSolver:
    def __init__(self):
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
        ik_config = IKSolverConfig.load_from_robot_config(
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
        self.ik_solver = IKSolver(ik_config)

    def solve_ik(self, target_position, target_quaternion):
        """
        Solve a single IK problem for the Franka robot.

        Args:
            target_position (list): Target position [x, y, z].
            target_quaternion (list): Target quaternion [w, x, y, z].

        Returns:
            tuple: Solution (7 joint angles) and solve time in seconds, or (None, solve time) if failed.
        """
        pos_tensor = torch.tensor(
            target_position, device=self.tensor_args.device, dtype=torch.float32
        )
        quat_tensor = torch.tensor(
            target_quaternion, device=self.tensor_args.device, dtype=torch.float32
        )
        goal = Pose(pos_tensor, quat_tensor)

        start_time = time.time()
        result = self.ik_solver.solve(goal)
        torch.cuda.synchronize()
        solve_time = time.time() - start_time

        if result.success.item():
            solution = result.solution.to("cpu").numpy().flatten()
            return solution, solve_time
        return None, solve_time

    def compute_fk(self, joint_angles):
        """
        Compute the forward kinematics for the Franka robot, returning only position and quaternion.

        Args:
            joint_angles: Tensor of joint angles [q1, q2, ..., q7]

        Returns:
            position: End-effector position [x, y, z]
            quaternion: End-effector quaternion [w, x, y, z]
        """
        joint_angles = torch.tensor(
            joint_angles, device=self.tensor_args.device, dtype=torch.float32
        )
        out = self.kin_model.get_state(joint_angles)
        position = out.ee_position
        quaternion = out.ee_quaternion
        return position, quaternion


if __name__ == "__main__":
    target_position = [0.5219848708, -0.0764696627, 0.2757705941]
    target_quaternion = [0.0019550403, 0.9997118148, -0.0235102575, 0.0044412223]
    fanka_solver = FrankaSolver()

    for _ in range(10):
        start_time = time.time()
        joint_solution, _ = fanka_solver.solve_ik(target_position, target_quaternion)
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

    old_pose = torch.cat(
        (torch.tensor(target_position), torch.tensor(target_quaternion))
    )

    print("original pose", old_pose)
    trans, quat = fanka_solver.compute_fk(joint_solution)

    print("solved pose", trans, quat)
