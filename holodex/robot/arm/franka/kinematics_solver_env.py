
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.util_file import get_robot_path




def solve(joint_state, ee_translation_goal, ee_orientation_goal, args):
    js_names = ["panda_joint1","panda_joint2","panda_joint3","panda_joint4", "panda_joint5",
      "panda_joint6","panda_joint7"]

    #collision_checker_type = CollisionCheckerType.BLOX

    tensor_args = TensorDeviceType()

    config_file = load_yaml(join_path(get_robot_path(), args.robot))#need to change
    urdf_file = config_file["robot_cfg"]["kinematics"][
        "urdf_path"
    ]  
    base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
    ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]

    robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)
    kin_model = CudaRobotModel(robot_cfg.kinematics)
    # compute forward kinematics:
    q = torch.tensor(joint_state).to(device="cuda:0")
    out = kin_model.get_state(q)
    print("fk_ee_position: ", out.ee_position)
    print("fk_ee_quaternion: ", out.ee_quaternion)
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        None,
        tensor_args,
        interpolation_dt=0.02,
        ee_link_name="ee_link",
    )
    motion_gen = MotionGen(motion_gen_config)
    print("warming up..")
    motion_gen.warmup(warmup_js_trajopt=False)

    tensor_args = TensorDeviceType()

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=4,
        max_attempts=2,
        enable_finetune_trajopt=True,
        time_dilation_factor=0.5,
    )

    #print("Constrained: Holding ")
    pose_cost_metric = PoseCostMetric(
        #hold_partial_pose=True,
        hold_vec_weight=motion_gen.tensor_args.to_device([1, 1, 1, 0, 1, 0]),
    )

    plan_config.pose_cost_metric = pose_cost_metric

    # motion generation:
    cu_js = JointState(
        position=tensor_args.to_device(joint_state),
        velocity=tensor_args.to_device(joint_state) * 0.0,
        acceleration=tensor_args.to_device(joint_state) * 0.0,
        jerk=tensor_args.to_device(joint_state) * 0.0,
        joint_names=js_names,
    )
    cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

   

    # compute curobo solution:
    ik_goal = Pose(
        position=tensor_args.to_device(ee_translation_goal),
        quaternion=tensor_args.to_device(ee_orientation_goal),
    )
    result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)

    succ = result.success.item()  
    if succ:
        print("success")
        motion_plan = result.get_interpolated_plan()
        motion_plan = motion_gen.get_full_js(motion_plan)
        motion_plan = motion_plan.get_ordered_joint_state(js_names)

        fk_out=kin_model.get_state(motion_plan.position)

        motion_plan_dict = {
            "position": motion_plan.position.cpu().numpy().tolist(),
            "ee_translation": fk_out.ee_position.cpu().numpy().tolist(),
            "joint_names": motion_plan.joint_names,
        }
        if args.debug >= 2:
            with open("motion_plan.json", "w") as json_file:
                json.dump(motion_plan_dict, json_file, indent=4)
            print("Saved motion_plan to motion_plan.json")
        
        if args.debug >= 1:
            print("visualizing")
            visualize(fk_out.ee_position.cpu().numpy())
        return motion_plan.position.cpu().numpy().tolist(), fk_out.ee_position.cpu().numpy().tolist(), motion_plan.position[-1].cpu().numpy().tolist()

    else:
        print("failed")
        return None