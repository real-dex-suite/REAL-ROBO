#!/usr/bin/env python3
import rospy
import open3d as o3d
import numpy as np
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from transforms3d.quaternions import quat2mat

def swap_y_z_axis(T):
    """
    Swap Y and Z axes in a 4x4 transformation matrix.
    
    Args:
        T (np.ndarray): 4x4 transformation matrix
    
    Returns:
        np.ndarray: New transformation matrix with Y and Z swapped
    """
    # Make a copy to avoid modifying the original
    T_new = T.copy()
    
    # Swap rotation rows (Y and Z)
    T_new[1, :], T_new[2, :] = T[2, :], T[1, :]
    
    # Swap rotation columns (Y and Z)
    T_new[:, 1], T_new[:, 2] = T_new[:, 2], T_new[:, 1].copy()
    
    return T_new

class EEPoseVisualizer:
    def __init__(self):
        rospy.init_node('ee_pose_visualizer', anonymous=True)
        
        # 订阅者
        self.pose_sub = rospy.Subscriber('vr/ee_pose', Pose, self.pose_callback)
        self.gripper_sub = rospy.Subscriber('vr/gripper', Float64, self.gripper_callback)
        
        # 当前姿态和夹爪状态
        self.current_pose = None
        self.gripper_width = 0.0
        self.pose_received = False
        self.rot = np.eye(3)
        self.pos = np.zeros((3, ))
        
        # 初始化Open3D可视化
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='End Effector Pose Visualization')
        
        # 创建坐标系和夹爪模型
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        # 添加到可视化
        self.vis.add_geometry(self.coordinate_frame)
        
        # 设置视角
        ctr = self.vis.get_view_control()
        ctr.set_front([0, -1, 0])  # 设置视角方向
        ctr.set_up([0, 0, 1])      # 设置上方向
    
    def pose_callback(self, msg):
        # 更新当前姿态
        self.current_pose = msg
        self.pose_received = True
    
    def gripper_callback(self, msg):
        self.gripper_width = msg.data
    
    def run(self):
        while not rospy.is_shutdown():
            if self.pose_received:
                # 更新坐标系位置
                pos = np.array([
                    self.current_pose.position.x,
                    self.current_pose.position.y,
                    self.current_pose.position.z
                ])
                
                # 转换四元数为旋转矩阵
                quat = [
                    self.current_pose.orientation.w,
                    self.current_pose.orientation.x,
                    self.current_pose.orientation.y,
                    self.current_pose.orientation.z
                ]
                rot = quat2mat(quat)
                transmat = np.zeros((4,4))
                transmat[:3, :3] = rot
                transmat[:3, 3] = pos
                transmat = swap_y_z_axis(transmat)
                def rfu_to_flu(T_rfu):
                    """
                    Convert a transformation matrix from RFU (Right, Front, Up) to FLU (Front, Left, Up).
                    
                    Args:
                        T_rfu (np.ndarray): 4x4 transformation matrix in RFU coordinates
                    
                    Returns:
                        np.ndarray: 4x4 transformation matrix in FLU coordinates
                    """
                    # Transformation matrix C (RFU -> FLU)
                    C = np.array([
                        [0, 1, 0, 0],
                        [-1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ])
                    
                    # Compute T_flu = C @ T_rfu @ C^{-1}
                    # Since C is orthonormal, C^{-1} = C.T
                    C_inv = C.T
                    
                    T_flu = C @ T_rfu @ C_inv
                    
                    return T_flu
                transmat = rfu_to_flu(transmat)
                rot = transmat[:3, :3]
                pos = transmat[:3, 3]
                self.coordinate_frame.translate(pos - self.pos)
                self.coordinate_frame.rotate(rot @ np.linalg.inv(self.rot), center=pos)
                
                self.pos = pos.copy()
                self.rot = rot.copy()
                
                # 更新可视化
                self.vis.update_geometry(self.coordinate_frame)
            
            # 渲染
            self.vis.poll_events()
            self.vis.update_renderer()
            
            rospy.sleep(0.01)

if __name__ == '__main__':
    try:
        visualizer = EEPoseVisualizer()
        visualizer.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        # 关闭可视化窗口
        visualizer.vis.destroy_window()