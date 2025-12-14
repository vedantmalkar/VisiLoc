#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from tf_transformations import euler_from_quaternion
import cv2 as cv
import numpy as np


class Aruco_Node(Node):
    def __init__(self):
        super().__init__("basic_aruco_localizer")
        self.image_width = 800    
        self.image_height = 600   
        self.horizontal_fov = 1.3962634 # found in bot urdf
        self.fx = (self.image_width / 2.0) / np.tan(self.horizontal_fov / 2.0)  #focal length  = (H/2) / tan(fov/2)
        self.fy = self.fx 
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0
        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.dist_coeffs = np.zeros(5, dtype=np.float64)    # Assuming ideal 
        self.aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50) 
        self.aruco_params = cv.aruco.DetectorParameters()

        self.marker_size = 0.4  # 40cm x 40cm marker
        
        self.marker_world_position = np.array([4.8, 0.0, 1.5])
        self.marker_world_yaw = np.pi  
        
        self.robot_yaw = 0
        self.odom_received = False

        self.bridge = CvBridge()  

        self.image_sub = self.create_subscription(Image,"/camera/image_raw",self.image_callback,10)

        self.odom_sub = self.create_subscription(Odometry,"/odom",self.odom_callback,10)
        
        self.position_pub = self.create_publisher(PoseStamped,"/robot_position",10)
        
        self.get_logger().info("Finder Node has been started")
    
    
    def odom_callback(self, msg):
        q = msg.pose.pose.orientation   # quaternion
        orientation_list = [q.x, q.y, q.z, q.w]
        
        roll, pitch, yaw = euler_from_quaternion(orientation_list)  #euler angles
        
        self.robot_yaw = yaw
        self.odom_received = True
    
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')   # convet to cv image (bridge)     
        
        marker_pose = self.detect_marker(cv_image)
        
        if marker_pose is not None and self.odom_received:
            robot_position = self.calculate_robot_world_position(marker_pose)

            self.print_localization_info(marker_pose, robot_position)
        
            self.publish_position(robot_position)
        elif marker_pose is not None:
            self.get_logger().warn("Yaw not found")

    def detect_marker(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        corners, ids, rejected = cv.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )
        
        if corners is None or ids is None or len(corners) == 0:
            return None

        rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
            corners,             
            self.marker_size,    
            self.camera_matrix,   
            self.dist_coeffs      # lens distortion 
        )
        
        rvec = rvecs[0][0]  
        tvec = tvecs[0][0]  
        marker_id = ids[0][0]
        
        distance = np.sqrt(tvec[0]**2 + tvec[2]**2)  # Euclidean distance
        
        return {
            'id': int(marker_id),
            'rvec': rvec,  
            'tvec': tvec,  
            'distance': distance
        }

    def calculate_robot_world_position(self, marker_pose):
        rvec = marker_pose['rvec']
        tvec = marker_pose['tvec']
        distance = marker_pose['distance']

        R_cam_to_marker, _ = cv.Rodrigues(rvec) # rotation matrix

        camera_pos_in_marker_frame = -R_cam_to_marker.T @ tvec      # - [rotation matrix] -> Transpose x transformation vector = camera position wrt marker 

        cam_x_marker = camera_pos_in_marker_frame[0]  # Horizontal offset
        cam_z_marker = camera_pos_in_marker_frame[2]  # Depth/distance
        
        cos_yaw = np.cos(self.marker_world_yaw)  # cos(180°) = -1
        sin_yaw = np.sin(self.marker_world_yaw)  # sin(180°) = 0
        
        camera_x_world = self.marker_world_position[0] + cam_z_marker * cos_yaw - cam_x_marker * sin_yaw

        camera_y_world = self.marker_world_position[1] + cam_z_marker * sin_yaw + cam_x_marker * cos_yaw
        
        robot_x = camera_x_world
        robot_y = camera_y_world
        
        return {
            'x': robot_x,
            'y': robot_y,
            'yaw': self.robot_yaw
        }
    
    def publish_position(self, robot_position):
        msg = PoseStamped()
        msg.header.frame_id = "world"
        
        msg.pose.position.x = robot_position['x']
        msg.pose.position.y = robot_position['y']
        msg.pose.position.z = 0.0
        
        self.position_pub.publish(msg)
    
    
    def print_localization_info(self, marker_pose, robot_position):
        yaw_deg = np.degrees(robot_position['yaw'])
        
        # Calculate rotation angle from rvec for educational purposes
        rotation_angle = np.linalg.norm(marker_pose['rvec'])
        rotation_angle_deg = np.degrees(rotation_angle)
        
        self.get_logger().info(f"current bot position: (x = {robot_position['x']}, y = {robot_position['y']}) | robot yaw: {robot_position['yaw']} | marker_id: {marker_pose['id']} | distance: {marker_pose['distance']:.3f}")


def main(args=None):
    rclpy.init(args=args)
    
    node = Aruco_Node()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()