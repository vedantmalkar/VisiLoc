#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from tf_transformations import euler_from_quaternion
import cv2 as cv
import numpy as np
import tkinter as tk
import threading
import math
import time

# --- ArUco and Camera Setup (Same as original) ---
image_width = 800
image_height = 600
horizontal_fov = 1.3962634
fx = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
fy = fx
cx = image_width / 2.0
cy = image_height / 2.0
dist_coef = np.zeros(5, dtype=np.float64)
cam_mat = np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float64)

marker_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
param_markers = cv.aruco.DetectorParameters()

def detect_aruco_markers(frame, marker_size=0.06, draw_markers=True):
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, rejected = cv.aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers)
    
    annotated_frame = frame.copy()
    distance = None
    pose = None
    
    if marker_corners and marker_IDs is not None:
        rVec, tVec, _ = cv.aruco.estimatePoseSingleMarkers(marker_corners, marker_size, cam_mat, dist_coef)
        rvec = rVec[0][0]
        tvec = tVec[0][0]
        corners = marker_corners[0]
        marker_id = marker_IDs[0][0]
        distance = np.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)
        pose = {
            'x': float(tvec[0]),
            'y': float(tvec[1]),
            'z': float(tvec[2]),
            'rvec': rvec,
            'tvec': tvec,
            'id': int(marker_id)
        }
        if draw_markers:
            # Drawing marker details on the CV image
            cv.polylines(annotated_frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA)
            corners_reshaped = corners.reshape(4, 2).astype(int)
            top_right = corners_reshaped[0]
            bottom_right = corners_reshaped[2]
            cv.drawFrameAxes(annotated_frame, cam_mat, dist_coef, rvec, tvec, 0.03, 3)
            cv.putText(annotated_frame, f"ID: {marker_id} Dist: {distance:.2f}m",
                      top_right, cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2, cv.LINE_AA)
            cv.putText(annotated_frame, f"x:{tvec[0]:.2f} y:{tvec[1]:.2f}",
                      bottom_right, cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2, cv.LINE_AA)
    
    return annotated_frame, distance, pose


class ArucoLocalizerGUI(Node):
    
    # --- GUI Constants ---
    CANVAS_SIZE = 500
    WORLD_MIN = -5.0
    WORLD_MAX = 5.0
    WORLD_SPAN = WORLD_MAX - WORLD_MIN
    ROBOT_RADIUS_PIXELS = 10
    
    def __init__(self):
        super().__init__("aruco_localizer_gui")
        
        # --- ROS 2 Initialization ---
        self.bridge = CvBridge()
        self.marker_size = 0.4
        self.marker_world_pos = (4.8, 0.0, 1.5) # Known marker position
        
        # Localization state variables
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0  # 0 radians (East/forward)
        self.odom_received = False
        
        # Subscribers and Publishers
        self.image_subscriber = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
        )
        self.odom_subscriber = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10
        )
        self.annotated_image_pub = self.create_publisher(
            Image,
            "/camera/image_annotated",
            10
        )
        
        self.get_logger().info('ROS 2 Node initialized. Subscribed to /camera/image_raw and /odom.')
        
        # --- Tkinter Initialization ---
        self.root = tk.Tk()
        self.root.title("Robot World Position")
        self.setup_gui()
        
    def setup_gui(self):
        """Sets up the Tkinter window and canvas."""
        self.canvas = tk.Canvas(self.root, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE, bg="#f0f0f0")
        self.canvas.pack(padx=10, pady=10)
        
        # Draw the world axes and bounds
        scale = self.CANVAS_SIZE / self.WORLD_SPAN
        center = self.CANVAS_SIZE / 2
        
        # Draw outer bounds
        self.canvas.create_rectangle(0, 0, self.CANVAS_SIZE, self.CANVAS_SIZE, outline="#222", width=2)
        
        # Draw central axes (at 0,0)
        self.canvas.create_line(center, 0, center, self.CANVAS_SIZE, fill="gray", dash=(4, 2))
        self.canvas.create_line(0, center, self.CANVAS_SIZE, center, fill="gray", dash=(4, 2))
        
        # Label axes
        self.canvas.create_text(self.CANVAS_SIZE - 20, center + 15, text="+X", fill="#cc0000") # X-axis
        self.canvas.create_text(center + 15, 20, text="+Y", fill="#006600") # Y-axis
        
        # Draw the marker world position (Static reference)
        marker_x_c, marker_y_c = self.world_to_canvas(self.marker_world_pos[0], self.marker_world_pos[1])
        self.canvas.create_oval(marker_x_c - 5, marker_y_c - 5, 
                                marker_x_c + 5, marker_y_c + 5, 
                                fill="blue", tags="marker_pos")
        self.canvas.create_text(marker_x_c, marker_y_c - 10, text=f"Marker ({self.marker_world_pos[0]}, {self.marker_world_pos[1]})", fill="blue")

        # Create placeholder for the robot
        self.robot_viz = self.canvas.create_oval(0, 0, 0, 0, fill="red", tags="robot")
        self.robot_arrow = self.canvas.create_line(0, 0, 0, 0, arrow=tk.LAST, fill="red", width=3, tags="robot")
        
        # Status Label
        self.status_text = tk.StringVar(self.root, "Waiting for data...")
        tk.Label(self.root, textvariable=self.status_text, font=('Arial', 12)).pack(pady=5)

    def world_to_canvas(self, x_world, y_world):
        """Converts world coordinates (-5 to 5) to canvas coordinates (0 to 500)."""
        scale = self.CANVAS_SIZE / self.WORLD_SPAN
        
        # X mapping: -5 -> 0, 5 -> 500
        canvas_x = (x_world - self.WORLD_MIN) * scale
        
        # Y mapping (Flipped for canvas: 5 -> 0, -5 -> 500)
        canvas_y = self.CANVAS_SIZE - ((y_world - self.WORLD_MIN) * scale)
        
        return int(canvas_x), int(canvas_y)

    def update_canvas(self):
        """Draws the robot's current position and orientation on the Tkinter canvas.
           This MUST be called only from the Tkinter main thread."""
           
        x_c, y_c = self.world_to_canvas(self.robot_x, self.robot_y)
        
        # 1. Update Robot Circle (Position)
        r = self.ROBOT_RADIUS_PIXELS
        self.canvas.coords(self.robot_viz, x_c - r, y_c - r, x_c + r, y_c + r)
        self.canvas.itemconfigure(self.robot_viz, fill="red", outline="black")
        
        # 2. Update Robot Arrow (Orientation)
        yaw = self.robot_yaw
        arrow_length = 20
        
        # Calculate arrow endpoint in canvas coordinates
        # Note: Math.cos(yaw) corresponds to X, Math.sin(yaw) corresponds to Y.
        # Since canvas Y is inverted, we use -math.sin(yaw) to make +Y point up.
        # However, the world_to_canvas transformation already handles Y inversion.
        # Let's use standard math here and trust the geometry.
        
        # ROS 2 Yaw: 0 is X-axis (Right), pi/2 is Y-axis (Up)
        # Screen: X increases right, Y increases DOWN
        
        # Calculate endpoint (relative to center)
        end_x_rel = arrow_length * math.cos(yaw)
        end_y_rel = arrow_length * math.sin(yaw)
        
        # Canvas coordinates: Y is inverted, but X is consistent.
        end_x_c = x_c + end_x_rel
        end_y_c = y_c - end_y_rel # Subtract because canvas Y goes down

        self.canvas.coords(self.robot_arrow, x_c, y_c, end_x_c, end_y_c)
        
        # 3. Update Status Label
        self.status_text.set(
            f"Robot World: X={self.robot_x:.2f}m, Y={self.robot_y:.2f}m\nYaw (rad): {self.robot_yaw:.2f}"
        )

    def odom_callback(self, msg):
        """Extract yaw angle from odometry quaternion"""
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, 
                          orientation_q.z, orientation_q.w]
        
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.robot_yaw = yaw
        self.odom_received = True
    
    def calculate_robot_position(self, pose):
        """Calculates robot world position from marker pose and known marker world position."""
        rvec = pose['rvec']
        tvec = pose['tvec']
        
        R_marker_to_cam, _ = cv.Rodrigues(rvec)
        
        # Camera position in marker's frame: -R^T * tvec
        camera_pos_in_marker_frame = -R_marker_to_cam.T @ tvec
        
        # Marker orientation in world frame (from URDF: 180 degrees, facing South)
        marker_yaw_world = np.pi  
        
        # Transform camera position from marker frame to world frame (2D rotation/translation)
        cam_x_in_marker = camera_pos_in_marker_frame[0] # X component in marker frame
        cam_z_in_marker = camera_pos_in_marker_frame[2] # Z component (depth) in marker frame
        
        # The rotation uses Z (depth) and X (horizontal) from the camera frame 
        # and transforms it into the World X (North) and Y (East) directions.
        cos_marker_yaw = np.cos(marker_yaw_world)
        sin_marker_yaw = np.sin(marker_yaw_world)
        
        # Rotate by marker's yaw and translate by marker's world position
        # Note: This is a simplified 2D transformation assuming the camera Z points towards the marker plane
        # The specific formula relies heavily on the definition of the marker's coordinate system (URDF).
        camera_x_world = self.marker_world_pos[0] + (cam_z_in_marker * cos_marker_yaw - cam_x_in_marker * sin_marker_yaw)
        camera_y_world = self.marker_world_pos[1] + (cam_z_in_marker * sin_marker_yaw + cam_x_in_marker * cos_marker_yaw)
        
        # Assuming camera center is robot center
        robot_x = camera_x_world
        robot_y = camera_y_world
        
        return robot_x, robot_y
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Detect ArUco markers
            annotated_frame, distance, pose = detect_aruco_markers(
                cv_image,
                marker_size=self.marker_size,
                draw_markers=True
            )
            
            if distance is not None and pose is not None:
                if self.odom_received:
                    # Calculate robot position with orientation compensation
                    robot_x, robot_y = self.calculate_robot_position(pose)
                    
                    # Update internal state for GUI update
                    self.robot_x = robot_x
                    self.robot_y = robot_y
                    
                    yaw_deg = np.degrees(self.robot_yaw)
                    
                    # Log comprehensive localization info
                    self.get_logger().info(
                        f"ID: {pose['id']} | Dist: {distance:.3f}m | Yaw: {yaw_deg:.1f}° | "
                        f"🎯 ROBOT POSITION: X={self.robot_x:.3f}m, Y={self.robot_y:.3f}m"
                    )
                    
                    # Add robot position to the annotated frame
                    cv.putText(annotated_frame, 
                              f"Robot World Pos: X={robot_x:.2f}m Y={robot_y:.2f}m",
                              (10, 30), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2, cv.LINE_AA)
                    cv.putText(annotated_frame,
                              f"Yaw: {yaw_deg:.1f} deg",
                              (10, 55), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2, cv.LINE_AA)
                              
                    # Safely schedule GUI update from the Tkinter main thread
                    self.root.after(0, self.update_canvas)
                    
                else:
                    self.get_logger().warn("Marker detected but no odometry data yet")
            else:
                self.get_logger().warn("No marker detected in frame")
            
            # Publish annotated image
            try:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
                annotated_msg.header = msg.header
                self.annotated_image_pub.publish(annotated_msg)
            except CvBridgeError as e:
                self.get_logger().error(f"Failed to publish annotated image: {e}")
                
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")
            
    def run_ros(self):
        """Starts the ROS 2 spinning loop in a separate thread."""
        try:
            rclpy.spin(self)
        except Exception as e:
            self.get_logger().error(f"ROS 2 Spin Error: {e}")
        finally:
            self.destroy_node()
            self.root.quit()

    def run(self):
        """Initializes and runs the application (ROS 2 in thread, Tkinter mainloop)."""
        # Start the ROS 2 spin in a separate thread
        self.ros_thread = threading.Thread(target=self.run_ros, daemon=True)
        self.ros_thread.start()
        
        # Start the Tkinter main loop in the main thread
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        """Handles closing the GUI window by shutting down ROS 2."""
        self.get_logger().info("Shutting down node and closing GUI...")
        rclpy.shutdown()
        self.root.destroy()
        

def main(args=None):
    rclpy.init(args=args)
    app = ArucoLocalizerGUI()
    app.run()
    # rclpy.shutdown() # Called inside on_closing

if __name__ == '__main__':
    main()