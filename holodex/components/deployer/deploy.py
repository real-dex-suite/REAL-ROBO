import sys
import numpy as np
from PIL import Image as PILImage
import torch
from torchvision import transforms as T
from holodex.components.deployer.wrappers import *
from holodex.utils.network import ImageSubscriber, frequency_timer
from holodex.utils import converter
from holodex.components.robot_operators.robot import RobotController
from holodex.constants import *
from scipy.spatial.transform import Rotation as R
import rospy
import threading

class DexArmDeploy(object):
    def __init__(
        self,
        deploy_configs
    ):
        torch.set_printoptions(sci_mode = False)
        self.configs = deploy_configs
        model = self.configs.model
        self.cam_num = self.configs.task.selected_view

        # Image transform
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean = torch.tensor(self.configs['task']['image_parameters']['mean_tensors'][self.cam_num - 1]),
                std = torch.tensor(self.configs['task']['image_parameters']['std_tensors'][self.cam_num - 1])
            )
        ])

        # Model initialization
        if model == 'VINN':
            print('Initializing the VINN deployment module...')
            self.model = DeployVINN(
                encoder_configs = self.configs['task']['encoder'],
                encoder_weights_path = self.configs['task']['vinn']['encoder_weights_path'],
                data_path = self.configs['task']['vinn']['data_path'],
                demo_list = self.configs['task']['vinn']['demos_list'],
                min_action_distance = self.configs['task']['vinn']["min_action_distance"],
                run_store_path = self.configs['task']['run_store_path'],
                absolute_actions = self.configs['absolute_actions'],
                selected_view = self.cam_num,
                nn_buffer_limit = self.configs['task']['vinn']['nn_buffer_limit'],
                transform = transform
            )

        elif model == 'BC':
            print('Initializing the BC deployment module...')
            self.model = DeployBC(
                encoder_configs = self.configs['task']['encoder'],
                predictor_configs = self.configs['task']['bc']['predictor'],
                model_weights_path = self.configs['task']['bc']['model_weights'],
                run_store_path = self.configs['task']['run_store_path'],
                selected_view = self.cam_num,
                transform = transform
            )
    
        # Image subscriber initialization
        self.robot_image_subscriber = ImageSubscriber(
            '/robot_camera_{}/color_image'.format(self.cam_num), 
            'robot_camera_{}_color'.format(self.cam_num)
        )

        # Robot controller initialization
        self.robot = RobotController(teleop = False, servo_mode = True, arm_control_mode = "joint")
        print('Robot controller initialized!')
        
        # move the robot to the home position
        self.robot.home_robot()
 
        if self.configs['run_loop']:
            self.frequency_timer = frequency_timer(self.configs.loop_rate)

    def _transform_image(self, image):
        image = PILImage.fromarray(image)
        image = image.crop(self.configs['task']['image_parameters']['crop_sizes'][self.cam_num - 1])
        image = image.resize((
            self.configs['task']['image_parameters']['image_size'], 
            self.configs['task']['image_parameters']['image_size']
        ))
        return np.asarray(image)

    def _get_transformed_image(self):
        return self._transform_image(self.robot_image_subscriber.get_image())
    
    def _get_hand_state(self):
        return self.robot.get_hand_position()
    
    def _get_arm_state(self):
        return self.robot.get_arm_tcp_position()
     
    def _predict_action(self, observation):
        return self.model.get_action(observation)

    def _postprocess_arm_action(self, pred_action, action_type):
        """
        Post-processes predicted arm action based on the specified type.

        Args:
            pred_action (np.array): Predicted action.
            action_type (str): Type of action ('cartesian_pose' or 'joint').

        Returns:
            np.array: Processed action.
        """
        if action_type == "cartesian_pose":
            arm_ee_pose = np.array(pred_action[:7])
            processed_action = np.zeros(6)
            arm_ee_pose[:3] /= ARM_POS_SCALE
            temp = converter.from_quaternion_to_euler_angle(arm_ee_pose[3:])
            processed_action[:3] = arm_ee_pose[:3]
            processed_action[3:] = temp
            return processed_action

        elif action_type == "joint":
            arm_joint = pred_action[:6]
            processed_action = np.array(converter.unscale_transform(
                arm_joint,
                ARM_JOINT_LOWER_LIMIT,
                ARM_JOINT_UPPER_LIMIT
            ))
            return processed_action
        else:
            raise ValueError("Invalid action type specified")

    def _postprocess_hand_action(self, pred_action, action_type):
        if action_type == "cartesian_pose": #TODO: quat, also have to consider the euler angle
            hand_joint = np.array(pred_action[7:])

        elif action_type == "joint":
            hand_joint = pred_action[6:]
        
        hand_joint = np.array(converter.unscale_transform(
            hand_joint,
            HAND_JOINT_LOWER_LIMIT,
            HAND_JOINT_UPPER_LIMIT
        ))
        return hand_joint

    def _create_robot_action(self, pred_action):
        arm_ee_pose = self._postprocess_arm_action(pred_action, 'joint')
        hand_joint = self._postprocess_hand_action(pred_action, 'joint')
        print("arm_ee_psoe", arm_ee_pose)
        print("hand_joint", hand_joint)

        return {
            'arm': arm_ee_pose,
            'hand': hand_joint
        }

    def _execute_action(self, action):
        self.robot.servo_move_test(action)
    
       
    def solve(self):
        sys.stdin = open(0)  # To get inputs while spawning multiple processes
        
        while True:
            robot_image = self.robot_image_subscriber.get_image()
            if robot_image is None:
                print('No image received!')
                continue
            
            hand_position = self.robot.get_hand_position()
            arm_position = self.robot.get_arm_tcp_position()
            if hand_position is None:
                print('No hand state received!')
                continue
            
            if arm_position is None:
                print('No arm state received!')
                continue

            print('\n***********************************************')
            
            # if not self.configs['run_loop']:
            #     register = input('\nPress a key to perform an action...')

            #     if register == 'h':
            #        print('Reseting the Robot!')
            #         self.robot.home_robot()
            #         continue
            
            # Get input - image
            transformed_image = self._get_transformed_image()
            print("Image has been received and transformed")
            
            input_dict = {
                'image': transformed_image,
            }
            
            pred_action = self._predict_action(input_dict)        
            postprocessed_action = self._create_robot_action(pred_action)
            print(postprocessed_action)
            self._execute_action(postprocessed_action)
            

            if self.configs['run_loop']:
                self.frequency_timer.sleep()

   