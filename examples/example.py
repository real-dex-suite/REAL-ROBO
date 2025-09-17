from real_robo.robots.franka_env_wrapper import FrankaEnvWrapper


if __name__ == "__main__":
    robot = FrankaEnvWrapper()
    print(robot.get_joint_positions())