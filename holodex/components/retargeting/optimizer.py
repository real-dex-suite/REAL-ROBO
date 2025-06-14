from abc import abstractmethod
from typing import List

import nlopt
import numpy as np
import sapien.core as sapien
import torch


class Optimizer:
    retargeting_type = "BASE"

    def __init__(
        self,
        robot: sapien.Articulation,
        wrist_link_name: str,
        target_joint_names: List[str],
        target_link_human_indices: np.ndarray,
    ):
        self.robot = robot
        self.robot_dof = robot.dof
        self.model = robot.create_pinocchio_model()
        self.wrist_link_name = wrist_link_name

        joint_names = [joint.get_name() for joint in robot.get_active_joints()]
        target_joint_index = []
        for target_joint_name in target_joint_names:
            if target_joint_name not in joint_names:
                raise ValueError(
                    f"Joint {target_joint_name} given does not appear to be in robot XML."
                )
            target_joint_index.append(joint_names.index(target_joint_name))
        self.target_joint_names = target_joint_names
        self.target_joint_indices = np.array(target_joint_index)
        self.fixed_joint_indices = np.array(
            [i for i in range(robot.dof) if i not in target_joint_index], dtype=int
        )
        self.opt = nlopt.opt(nlopt.LD_SLSQP, len(target_joint_index))
        self.dof = len(target_joint_index)

        # Target
        self.target_link_human_indices = target_link_human_indices

        # Free joint
        link_names = [link.get_name() for link in self.robot.get_links()]
        self.has_free_joint = len([name for name in link_names if "dummy" in name]) >= 6

    def set_joint_limit(self, joint_limits: np.ndarray):
        if joint_limits.shape != (self.dof, 2):
            raise ValueError(
                f"Expect joint limits have shape: {(self.dof, 2)}, but get {joint_limits.shape}"
            )
        self.opt.set_lower_bounds(joint_limits[:, 0].tolist())
        self.opt.set_upper_bounds(joint_limits[:, 1].tolist())

    def get_last_result(self):
        return self.opt.last_optimize_result()

    def get_link_names(self):
        return [link.get_name() for link in self.robot.get_links()]

    def get_link_indices(self, target_link_names):
        target_link_index = []
        for target_link_name in target_link_names:
            if target_link_name not in self.get_link_names():
                raise ValueError(
                    f"Body {target_link_name} given does not appear to be in robot XML."
                )
            target_link_index.append(self.get_link_names().index(target_link_name))
        return target_link_index

    @abstractmethod
    def retarget(self, ref_value, fixed_qpos, last_qpos=None):
        pass

    def optimize(self, objective_fn, last_qpos):
        self.opt.set_min_objective(objective_fn)
        try:
            qpos = self.opt.optimize(last_qpos)
        except RuntimeError as e:
            print(e)
            return np.array(last_qpos)
        return qpos


class PositionOptimizer(Optimizer):
    retargeting_type = "position"

    def __init__(
        self,
        robot: sapien.Articulation,
        wrist_link_name: str,
        target_joint_names: List[str],
        target_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
    ):
        super().__init__(
            robot, wrist_link_name, target_joint_names, target_link_human_indices
        )
        self.body_names = target_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta)
        self.norm_delta = norm_delta

        # Sanity check and cache link indices
        self.target_link_indices = self.get_link_indices(target_link_names)

        # Use local jacobian if target link name <= 2, otherwise first cache all jacobian and then get all
        # This is only for the speed but will not affect the performance
        if len(target_link_names) <= 40:
            self.use_sparse_jacobian = True
        else:
            self.use_sparse_jacobian = False
        self.opt.set_ftol_abs(1e-5)

    def _get_objective_function(
        self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.robot_dof)
        qpos[self.fixed_joint_indices] = fixed_qpos
        torch_target_pos = torch.as_tensor(target_pos)
        torch_target_pos.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.target_joint_indices] = x
            self.model.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.model.get_link_pose(index) for index in self.target_link_indices
            ]
            body_pos = np.array([pose.p for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Loss term for kinematics retargeting based on 3D position error
            huber_distance = self.huber_loss(torch_body_pos, torch_target_pos)
            # huber_distance = torch.norm(torch_body_pos - torch_target_pos, dim=1).mean()
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                if self.use_sparse_jacobian:
                    jacobians = []
                    for i, index in enumerate(self.target_link_indices):
                        link_spatial_jacobian = (
                            self.model.compute_single_link_local_jacobian(qpos, index)[
                                :3, self.target_joint_indices
                            ]
                        )
                        link_rot = self.model.get_link_pose(
                            index
                        ).to_transformation_matrix()[:3, :3]
                        link_kinematics_jacobian = link_rot @ link_spatial_jacobian
                        jacobians.append(link_kinematics_jacobian)
                    jacobians = np.stack(jacobians, axis=0)
                else:
                    self.model.compute_full_jacobian(qpos)
                    jacobians = [
                        self.model.get_link_jacobian(index, local=True)[
                            :3, self.target_joint_indices
                        ]
                        for index in self.target_link_indices
                    ]

                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]
                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)

                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective

    def retarget(self, ref_value, fixed_qpos, last_qpos=None):
        if len(fixed_qpos) != len(self.fixed_joint_indices):
            raise ValueError(
                f"Optimizer has {len(self.fixed_joint_indices)} joints but non_target_qpos {fixed_qpos} is given"
            )
        if last_qpos is None:
            last_qpos = np.zeros(self.dof)
        if isinstance(last_qpos, np.ndarray):
            last_qpos = last_qpos.astype(np.float32)
        last_qpos = list(last_qpos)
        objective_fn = self._get_objective_function(
            ref_value, fixed_qpos, np.array(last_qpos).astype(np.float32)
        )
        return self.optimize(objective_fn, last_qpos)


class VectorOptimizer(Optimizer):
    retargeting_type = "VECTOR"

    def __init__(
        self,
        robot: sapien.Articulation,
        wrist_link_name: str,
        target_joint_names: List[str],
        target_origin_link_names: List[str],
        target_task_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
        scaling=List[float],
    ):
        super().__init__(
            robot, wrist_link_name, target_joint_names, target_link_human_indices
        )
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="mean")
        self.norm_delta = norm_delta
        self.scaling = torch.as_tensor(scaling).unsqueeze(1)

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_task_link_names]
        )

        # Sanity check and cache link indices
        self.robot_link_indices = self.get_link_indices(self.computed_link_names)

        # Use local jacobian if target link name <= 2, otherwise first cache all jacobian and then get all
        # This is only for the speed but will not affect the performance
        if len(self.computed_link_names) <= 40:
            self.use_sparse_jacobian = True
        else:
            self.use_sparse_jacobian = False
        self.opt.set_ftol_abs(1e-6)

    def _get_objective_function(
        self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.robot_dof)
        qpos[self.fixed_joint_indices] = fixed_qpos
        torch_target_vec = torch.as_tensor(target_vector) * self.scaling
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.target_joint_indices] = x
            self.model.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.model.get_link_pose(index) for index in self.robot_link_indices
            ]
            body_pos = np.array([pose.p for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            huber_distance = self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                if self.use_sparse_jacobian:
                    jacobians = []
                    for i, index in enumerate(self.robot_link_indices):
                        link_spatial_jacobian = (
                            self.model.compute_single_link_local_jacobian(qpos, index)[
                                :3, self.target_joint_indices
                            ]
                        )
                        link_rot = self.model.get_link_pose(
                            index
                        ).to_transformation_matrix()[:3, :3]
                        link_kinematics_jacobian = link_rot @ link_spatial_jacobian
                        jacobians.append(link_kinematics_jacobian)
                    jacobians = np.stack(jacobians, axis=0)
                else:
                    self.model.compute_full_jacobian(qpos)
                    jacobians = [
                        self.model.get_link_jacobian(index, local=True)[
                            :3, self.target_joint_indices
                        ]
                        for index in self.robot_link_indices
                    ]

                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]
                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)

                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective

    def retarget(self, ref_value, fixed_qpos, last_qpos=None):
        if len(fixed_qpos) != len(self.fixed_joint_indices):
            raise ValueError(
                f"Optimizer has {len(self.fixed_joint_indices)} joints but non_target_qpos {fixed_qpos} is given"
            )
        if last_qpos is None:
            last_qpos = np.zeros(self.dof)
        last_qpos = list(last_qpos)
        objective_fn = self._get_objective_function(
            ref_value, fixed_qpos, np.array(last_qpos).astype(np.float32)
        )
        return self.optimize(objective_fn, last_qpos)


class DexPilotAllegroOptimizer(Optimizer):
    """Retargeting optimizer using the method proposed in DexPilot

    This is a broader adaptation of the original optimizer delineated in the DexPilot paper.
    While the initial DexPilot study focused solely on the four-fingered Allegro Hand, this version of the optimizer
    embraces the same principles for both four-fingered and five-fingered hands. It projects the distance between the
    thumb and the other fingers to facilitate more stable grasping.
    Reference: https://arxiv.org/abs/1910.03135

    Args:
        robot:
        target_joint_names:
        finger_tip_link_names:
        wrist_link_name:
        gamma:
        project_dist:
        escape_dist:
        eta1:
        eta2:
        scaling:
    """

    retargeting_type = "DEXPILOT"

    def __init__(
        self,
        robot: sapien.Articulation,
        target_joint_names: List[str],
        finger_tip_link_names: List[str],
        finger_dip_link_names: List[str],
        wrist_link_name: str,
        huber_delta=0.03,
        norm_delta=4e-3,
        # DexPilot parameters
        # gamma=2.5e-3,
        project_dist=0.03,
        escape_dist=0.05,
        eta1=1e-4,
        eta2=3e-2,
        scaling=1.0,
        target_tip_link_human_indices=np.array([0, 1, 2, 3]),
        target_dip_link_human_indices=np.array([]),
    ):
        if len(finger_tip_link_names) < 4 or len(finger_tip_link_names) > 5:
            raise ValueError(
                f"DexPilot optimizer can only be applied to hands with four or five fingers"
            )
        is_four_finger = len(finger_tip_link_names) == 4
        if is_four_finger:
            origin_link_index = [2, 3, 4, 3, 4, 4, 0, 0, 0, 0]
            task_link_index = [1, 1, 1, 2, 2, 3, 1, 2, 3, 4]
            if len(finger_dip_link_names) > 0:
                origin_link_index += [5, 6, 7, 8]
                task_link_index += [1, 2, 3, 4]
                self.num_dip_fingers = len(finger_dip_link_names)
            else:
                self.num_dip_fingers = 0
            self.num_fingers = 4
        else:
            origin_link_index = [2, 3, 4, 5, 3, 4, 5, 4, 5, 5, 0, 0, 0, 0, 0]
            task_link_index = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 1, 2, 3, 4, 5]
            self.num_fingers = 5

        human_thumb_idx = target_tip_link_human_indices[0]
        human_index_idx = target_tip_link_human_indices[1]
        human_middle_idx = target_tip_link_human_indices[2]
        human_ring_idx = target_tip_link_human_indices[3]
        human_origin_link_index = [
            human_index_idx,
            human_middle_idx,
            human_ring_idx,
            human_middle_idx,
            human_ring_idx,
            human_ring_idx,
            0,
            0,
            0,
            0,
        ]
        human_task_link_index = [
            human_thumb_idx,
            human_thumb_idx,
            human_thumb_idx,
            human_index_idx,
            human_index_idx,
            human_middle_idx,
            human_thumb_idx,
            human_index_idx,
            human_middle_idx,
            human_ring_idx,
        ]
        if len(finger_dip_link_names) > 0:
            human_origin_link_index += target_dip_link_human_indices
            human_task_link_index += [
                human_thumb_idx,
                human_index_idx,
                human_middle_idx,
                human_ring_idx,
            ]

        target_link_human_indices = (
            np.stack([human_origin_link_index, human_task_link_index], axis=0)
        ).astype(int)

        link_names = [wrist_link_name] + finger_tip_link_names + finger_dip_link_names
        target_origin_link_names = [link_names[index] for index in origin_link_index]
        target_task_link_names = [link_names[index] for index in task_link_index]

        super().__init__(
            robot, wrist_link_name, target_joint_names, target_link_human_indices
        )
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.scaling = scaling
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="none")
        self.norm_delta = norm_delta

        # DexPilot parameters
        self.project_dist = project_dist
        self.escape_dist = escape_dist
        self.eta1 = eta1
        self.eta2 = eta2

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_task_link_names]
        )

        # Sanity check and cache link indices
        self.robot_link_indices = self.get_link_indices(self.computed_link_names)

        # Use local jacobian if target link name <= 2, otherwise first cache all jacobian and then get all
        # This is only for the speed but will not affect the performance
        if len(self.computed_link_names) <= 40:
            self.use_sparse_jacobian = True
        else:
            self.use_sparse_jacobian = False
        self.opt.set_ftol_abs(1e-6)

        # DexPilot cache
        if is_four_finger:
            self.projected = np.zeros(6, dtype=bool)
            self.s2_project_index_origin = np.array([1, 2, 2], dtype=int)
            self.s2_project_index_task = np.array([0, 0, 1], dtype=int)
            self.projected_dist = np.array([eta1] * 3 + [eta2] * 3)
        else:
            self.projected = np.zeros(10, dtype=bool)
            self.s2_project_index_origin = np.array([1, 2, 3, 2, 3, 3], dtype=int)
            self.s2_project_index_task = np.array([0, 0, 0, 1, 1, 2], dtype=int)
            self.projected_dist = np.array([eta1] * 4 + [eta2] * 6)

    def _get_objective_function_four_finger(
        self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        target_vector = target_vector.astype(np.float32)
        qpos = np.zeros(self.robot_dof)
        qpos[self.fixed_joint_indices] = fixed_qpos

        len_proj = len(self.projected)
        len_s2 = len(self.s2_project_index_task)
        len_s1 = len_proj - len_s2

        # Update projection indicator
        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
        self.projected[:len_s1][target_vec_dist[0:len_s1] < self.project_dist] = True
        self.projected[:len_s1][target_vec_dist[0:len_s1] > self.escape_dist] = False
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[:len_s1][self.s2_project_index_origin],
            self.projected[:len_s1][self.s2_project_index_task],
        )
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[len_s1:len_proj], target_vec_dist[len_s1:len_proj] <= 0.03
        )

        # Update weight vector
        normal_weight = np.ones(len_proj, dtype=np.float32) * 1
        high_weight = np.array([200] * len_s1 + [400] * len_s2, dtype=np.float32)
        weight = np.where(self.projected, high_weight, normal_weight) * 0

        # We change the weight to 10 instead of 1 here, for vector originate from wrist to fingertips
        # This ensures better intuitive mapping due wrong pose detection
        weight = torch.from_numpy(
            np.concatenate(
                [
                    weight,
                    np.ones(self.num_fingers, dtype=np.float32) * len_proj
                    + self.num_fingers,
                    np.ones(self.num_dip_fingers, dtype=np.float32) * 10,
                ]
            )
        )

        # Compute reference distance vector
        normal_vec = target_vector * self.scaling  # (10, 3)
        dir_vec = target_vector[:len_proj] / (target_vec_dist[:, None] + 1e-6)  # (6, 3)
        projected_vec = dir_vec * self.projected_dist[:, None]  # (6, 3)

        # Compute final reference vector
        reference_vec = np.where(
            self.projected[:, None], projected_vec, normal_vec[:len_proj]
        )  # (6, 3)
        reference_vec = np.concatenate(
            [reference_vec, normal_vec[len_proj:]], axis=0
        )  # (10, 3)
        torch_target_vec = torch.as_tensor(reference_vec, dtype=torch.float32)
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.target_joint_indices] = x
            self.model.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.model.get_link_pose(index) for index in self.robot_link_indices
            ]
            body_pos = np.array([pose.p for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            # Different from the original DexPilot, we use huber loss here instead of the squared dist
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            huber_distance = (
                self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
                * weight
                / (robot_vec.shape[0])
            ).sum()
            huber_distance = huber_distance.sum()
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                if self.use_sparse_jacobian:
                    jacobians = []
                    for i, index in enumerate(self.robot_link_indices):
                        link_spatial_jacobian = (
                            self.model.compute_single_link_local_jacobian(qpos, index)[
                                :3, self.target_joint_indices
                            ]
                        )
                        link_rot = self.model.get_link_pose(
                            index
                        ).to_transformation_matrix()[:3, :3]
                        link_kinematics_jacobian = link_rot @ link_spatial_jacobian
                        jacobians.append(link_kinematics_jacobian)
                    jacobians = np.stack(jacobians, axis=0)
                else:
                    self.model.compute_full_jacobian(qpos)
                    jacobians = [
                        self.model.get_link_jacobian(index, local=True)[
                            :3, self.target_joint_indices
                        ]
                        for index in self.robot_link_indices
                    ]

                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]
                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)

                # In the original DexPilot, γ = 2.5 × 10−3 is a weight on regularizing the Allegro angles to zero
                # which is equivalent to fully opened the hand
                # In our implementation, we regularize the joint angles to the previous joint angles
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective

    def retarget(self, ref_value, fixed_qpos, last_qpos=None):
        if len(fixed_qpos) != len(self.fixed_joint_indices):
            raise ValueError(
                f"Optimizer has {len(self.fixed_joint_indices)} joints but non_target_qpos {fixed_qpos} is given"
            )
        if last_qpos is None:
            last_qpos = np.zeros(self.dof)
        objective_fn = self._get_objective_function_four_finger(
            ref_value, fixed_qpos, np.array(last_qpos).astype(np.float32)
        )
        return self.optimize(objective_fn, last_qpos)
