
from real_robo.robots.franka_env_wrapper import FrankaEnvWrapper

#TODO: add more functionalities
class RobotController(object):
    def __init__(
        self,
    ) -> None:

 
        self.robot = FrankaEnvWrapper()


    def get_arm_position(self):
       return self.robot.get_arm_position()

    def get_gripper_state(self):
        return self.robot.get_gripper_state()
    

if __name__ == "__main__":
    robot = RobotController()
    print(robot.get_arm_position())