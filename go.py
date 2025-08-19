#!/usr/bin/env python3
"""
Go-2 Robot Forklift Detection Controller for Isaac Sim (ROS2 Humble)
Rotates robot in place until it detects a forklift using vLLM vision model.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import base64
import requests
import json
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import threading
import time
import os
from datetime import datetime

class Go2ForkliftController(Node):
    def __init__(self):
        super().__init__('go2_forklift_controller')
        
        # Declare parameters
        self.declare_parameter('vllm_server_url', 'http://localhost:8000')
        self.declare_parameter('model_name', 'llava-hf/llava-1.5-7b-hf')
        self.declare_parameter('rotation_speed', 0.6)  # rad/s for rotation in place
        self.declare_parameter('approach_speed', 0.8)  # m/s for moving towards forklift
        self.declare_parameter('target_distance', 0.8)  # meters - target distance to forklift
        self.declare_parameter('distance_tolerance', 0.1)  # meters - tolerance for target distance
        self.declare_parameter('detection_interval', 2.0)  # seconds
        self.declare_parameter('image_topic', '/unitree_go2/front_cam/color_image')
        self.declare_parameter('depth_topic', '/unitree_go2/front_cam/depth_image')
        self.declare_parameter('cmd_vel_topic', '/unitree_go2/cmd_vel')
        self.declare_parameter('save_detection_image', True)  # whether to save detected image
        self.declare_parameter('output_dir', './forklift_detections')  # directory to save images
        self.declare_parameter('rotation_direction', 'left')  # 'left' or 'right'
        
        # Get parameters
        self.vllm_server_url = self.get_parameter('vllm_server_url').get_parameter_value().string_value
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self.rotation_speed = self.get_parameter('rotation_speed').get_parameter_value().double_value
        self.approach_speed = self.get_parameter('approach_speed').get_parameter_value().double_value
        self.target_distance = self.get_parameter('target_distance').get_parameter_value().double_value
        self.distance_tolerance = self.get_parameter('distance_tolerance').get_parameter_value().double_value
        self.detection_interval = self.get_parameter('detection_interval').get_parameter_value().double_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.save_detection_image = self.get_parameter('save_detection_image').get_parameter_value().bool_value
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self.rotation_direction = self.get_parameter('rotation_direction').get_parameter_value().string_value
        
        # State variables
        self.current_image = None
        self.current_depth_image = None
        self.forklift_detected = False
        self.target_reached = False
        self.running = True
        self.bridge = CvBridge()
        self.image_lock = threading.Lock()
        self.depth_lock = threading.Lock()
        
        # Robot state
        self.robot_state = "SEARCHING"  # "SEARCHING", "APPROACHING", "TARGET_REACHED"
        self.is_rotating = False
        self.current_forklift_distance = None
        
        # Create output directory for saving detection images
        if self.save_detection_image:
            self.setup_output_directory()
        
        # QoS profile for image subscription (to handle potential network issues)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # ROS2 publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.image_sub = self.create_subscription(
            Image, 
            self.image_topic, 
            self.image_callback, 
            qos_profile
        )
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile
        )
        
        # Timer for main control loop (10Hz)
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        # Timer for forklift detection (runs during search and approach)
        self.detection_timer = self.create_timer(self.detection_interval, self.detection_callback)
        
        self.last_detection_time = time.time()
        
        # Wait for first image
        self.get_logger().info("Waiting for camera feed...")
        self.wait_for_image()
        
        # Test vLLM server connection
        self.test_server_connection()
        
        self.get_logger().info("Go-2 Forklift Controller initialized successfully!")
        self.get_logger().info(f"Robot will rotate {self.rotation_direction} at {self.rotation_speed} rad/s")
        self.get_logger().info(f"Target distance: {self.target_distance}m (±{self.distance_tolerance}m)")
        self.get_logger().info(f"Approach speed: {self.approach_speed} m/s")
        if self.save_detection_image:
            self.get_logger().info(f"Detection images will be saved to: {self.output_dir}")
    
    def setup_output_directory(self):
        """Create output directory for saving detection images."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.get_logger().info(f"Output directory ready: {self.output_dir}")
        except Exception as e:
            self.get_logger().error(f"Failed to create output directory {self.output_dir}: {e}")
            self.save_detection_image = False
        
    def wait_for_image(self):
        """Wait for the first image to arrive."""
        start_time = time.time()
        timeout = 10.0  # 10 seconds timeout
        
        while self.current_image is None and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            
        if self.current_image is None:
            self.get_logger().warn("No image received within timeout, continuing anyway...")
        else:
            self.get_logger().info("Camera feed received!")
    
    def test_server_connection(self):
        """Test connection to vLLM server."""
        try:
            response = requests.get(f"{self.vllm_server_url}/health", timeout=5)
            if response.status_code == 200:
                self.get_logger().info("vLLM server connection successful!")
            else:
                self.get_logger().warn(f"vLLM server responded with status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Failed to connect to vLLM server: {e}")
            self.get_logger().error("Make sure the vLLM server is running!")
    
    def image_callback(self, msg):
        """Callback for receiving images from the camera."""
        try:
            with self.image_lock:
                # Convert ROS2 Image to OpenCV format
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.current_image = cv_image.copy()
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def depth_callback(self, msg):
        """Callback for receiving depth images from the camera."""
        try:
            with self.depth_lock:
                # Convert ROS2 Image to OpenCV format
                # Depth images are typically 32FC1 (32-bit float, single channel)
                depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                self.current_depth_image = depth_image.copy()
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")
    
    def calculate_forklift_distance(self):
        """Calculate distance to forklift using depth image."""
        if self.current_depth_image is None:
            self.get_logger().warn("No depth image available for distance calculation")
            return None
        
        try:
            with self.depth_lock:
                depth_copy = self.current_depth_image.copy()
            
            # Remove invalid depth values (NaN, inf, 0)
            valid_mask = np.isfinite(depth_copy) & (depth_copy > 0)
            
            if not np.any(valid_mask):
                self.get_logger().warn("No valid depth values found in depth image")
                return None
            
            # Get only valid depth values
            valid_depths = depth_copy[valid_mask]
            
            # Since we assume only the forklift is in the scene, use statistical measures
            # to find the approximate distance to the forklift
            
            # Method 1: Use the minimum valid depth (closest point)
            min_distance = np.min(valid_depths)
            
            # Method 2: Use median of the closest 10% of points (more robust)
            sorted_depths = np.sort(valid_depths)
            closest_10_percent = int(len(sorted_depths) * 0.1)
            if closest_10_percent < 1:
                closest_10_percent = 1
            median_closest = np.median(sorted_depths[:closest_10_percent])
            
            # Method 3: Use mean of the closest 5% of points
            closest_5_percent = int(len(sorted_depths) * 0.05)
            if closest_5_percent < 1:
                closest_5_percent = 1
            mean_closest = np.mean(sorted_depths[:closest_5_percent])
            
            # Log detailed statistics
            self.get_logger().info("📏 Forklift Distance Analysis:")
            self.get_logger().info(f"   Minimum distance: {min_distance:.2f} meters")
            self.get_logger().info(f"   Median of closest 10%: {median_closest:.2f} meters")
            self.get_logger().info(f"   Mean of closest 5%: {mean_closest:.2f} meters")
            self.get_logger().info(f"   Total valid depth points: {len(valid_depths)}")
            
            # Return the median of closest points as the most robust estimate
            return median_closest
            
        except Exception as e:
            self.get_logger().error(f"Error calculating forklift distance: {e}")
            return None
    
    def save_depth_analysis_image(self, distance):
        """Save depth image with distance overlay for analysis."""
        if not self.save_detection_image or self.current_depth_image is None:
            return None
        
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forklift_depth_analysis_{timestamp}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            # Create a visualization of the depth image
            depth_copy = self.current_depth_image.copy()
            
            # Normalize depth for visualization (0-255)
            valid_mask = np.isfinite(depth_copy) & (depth_copy > 0)
            if np.any(valid_mask):
                min_depth = np.min(depth_copy[valid_mask])
                max_depth = np.max(depth_copy[valid_mask])
                
                # Create normalized depth image
                normalized_depth = np.zeros_like(depth_copy, dtype=np.uint8)
                if max_depth > min_depth:
                    normalized_depth[valid_mask] = ((depth_copy[valid_mask] - min_depth) / 
                                                   (max_depth - min_depth) * 255).astype(np.uint8)
                
                # Convert to color for better visualization
                depth_colored = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
                
                # Add distance information overlay
                cv2.putText(depth_colored, f"Distance: {distance:.2f}m", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(depth_colored, f"Analyzed: {datetime.now().strftime('%H:%M:%S')}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(depth_colored, f"Min: {min_depth:.2f}m, Max: {max_depth:.2f}m", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Save the image
                success = cv2.imwrite(filepath, depth_colored)
                
                if success:
                    self.get_logger().info(f"✅ Depth analysis image saved: {filepath}")
                    return filepath
                else:
                    self.get_logger().error(f"❌ Failed to save depth image to {filepath}")
            
            return None
                
        except Exception as e:
            self.get_logger().error(f"Error saving depth analysis image: {e}")
            return None
    
    def encode_image_to_base64(self, cv_image):
        """Convert OpenCV image to base64 string."""
        try:
            # Encode image to JPEG format
            _, buffer = cv2.imencode('.jpg', cv_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            # Convert to base64
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return image_base64
        except Exception as e:
            self.get_logger().error(f"Error encoding image: {e}")
            return None
    
    def save_forklift_image(self, cv_image, vllm_response):
        """Save the image where forklift was detected."""
        if not self.save_detection_image:
            return None
            
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forklift_detected_{timestamp}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            # Add text overlay with detection info
            overlay_image = cv_image.copy()
            
            # Add timestamp
            cv2.putText(overlay_image, f"Detected: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add detection response (truncated if too long)
            response_text = vllm_response[:50] + "..." if len(vllm_response) > 50 else vllm_response
            cv2.putText(overlay_image, f"Response: {response_text}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Save the image
            success = cv2.imwrite(filepath, overlay_image)
            
            if success:
                self.get_logger().info(f"✅ Forklift detection image saved: {filepath}")
                return filepath
            else:
                self.get_logger().error(f"❌ Failed to save image to {filepath}")
                return None
                
        except Exception as e:
            self.get_logger().error(f"Error saving forklift image: {e}")
            return None
    
    def detect_forklift(self, cv_image):
        """Send image to vLLM server to detect forklift."""
        try:
            # Encode image
            image_base64 = self.encode_image_to_base64(cv_image)
            if image_base64 is None:
                return False, "Image encoding failed"
            
            # Prepare request
            headers = {"Content-Type": "application/json"}
            prompt = ("Look at this image carefully. Is there a forklift visible in the image? "
                     "Answer with 'YES' if you can see a forklift, or 'NO' if you cannot see a forklift. "
                     "Be specific and only answer YES or NO followed by a brief explanation.")
            
            data = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                            }
                        ]
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.3  # Lower temperature for more consistent responses
            }
            
            # Send request to vLLM server
            response = requests.post(
                f"{self.vllm_server_url}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["message"]["content"].strip()
                
                # Check if response indicates forklift detection
                forklift_found = response_text.upper().startswith('YES')
                
                self.get_logger().info(f"vLLM Response: {response_text}")
                return forklift_found, response_text
            else:
                self.get_logger().error(f"vLLM server error: {response.status_code} - {response.text}")
                return False, f"Server error: {response.status_code}"
                
        except requests.exceptions.Timeout:
            self.get_logger().error("vLLM server request timed out")
            return False, "Request timeout"
        except Exception as e:
            self.get_logger().error(f"Error detecting forklift: {e}")
            return False, str(e)
    
    def move_forward(self):
        """Send command to move robot forward towards forklift."""
        twist = Twist()
        twist.linear.x = self.approach_speed
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        
        self.cmd_vel_pub.publish(twist)
    
    def rotate_in_place(self):
        """Send command to rotate robot in place."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        
        # Set rotation direction based on parameter
        if self.rotation_direction.lower() == 'left':
            twist.angular.z = self.rotation_speed  # Positive Z = turn left
        else:
            twist.angular.z = -self.rotation_speed  # Negative Z = turn right
        
        self.cmd_vel_pub.publish(twist)
        self.is_rotating = True
    
    def stop_robot(self):
        """Send command to stop robot."""
        twist = Twist()  # All zeros
        self.cmd_vel_pub.publish(twist)
        self.is_rotating = False
        if self.robot_state == "TARGET_REACHED":
            self.get_logger().info("🎯 Target reached! Robot stopped at target distance.")
        else:
            self.get_logger().info("Robot stopped!")
    
    def check_target_distance(self):
        """Check if robot has reached target distance to forklift."""
        if self.current_forklift_distance is None:
            return False
        
        distance_error = abs(self.current_forklift_distance - self.target_distance)
        
        if distance_error <= self.distance_tolerance:
            return True
        elif self.current_forklift_distance < self.target_distance:
            self.get_logger().warn(f"⚠️  Robot too close! Current: {self.current_forklift_distance:.2f}m, Target: {self.target_distance:.2f}m")
            return True  # Stop if too close
        
        return False
    
    def control_loop(self):
        """Main control loop callback (10Hz) - implements search and approach behavior."""
        if not self.running:
            self.stop_robot()
            return
        
        if self.robot_state == "SEARCHING":
            # Search phase: rotate until forklift is found
            if not self.is_rotating:
                self.get_logger().info(f"Starting rotation search ({self.rotation_direction})...")
            self.rotate_in_place()
            
        elif self.robot_state == "APPROACHING":
            # Approach phase: move towards forklift while monitoring distance
            if self.current_forklift_distance is not None:
                if self.check_target_distance():
                    # Target reached!
                    self.robot_state = "TARGET_REACHED"
                    self.target_reached = True
                    self.stop_robot()
                    self.get_logger().info(f"🎉 SUCCESS! Reached target distance of {self.target_distance}m")
                    self.get_logger().info(f"Final distance: {self.current_forklift_distance:.2f}m")
                    # Stop all timers
                    self.detection_timer.cancel()
                    self.control_timer.cancel()
                else:
                    # Continue moving forward
                    self.move_forward()
                    remaining_distance = self.current_forklift_distance - self.target_distance
                    self.get_logger().info(f"Approaching forklift... Current: {self.current_forklift_distance:.2f}m, "
                                         f"Remaining: {remaining_distance:.2f}m")
            else:
                # No distance reading, stop for safety
                self.get_logger().warn("No distance reading available, stopping for safety")
                self.stop_robot()
        
        elif self.robot_state == "TARGET_REACHED":
            # Target reached, stay stopped
            self.stop_robot()
    
    def detection_callback(self):
        """Forklift detection callback."""
        if not self.running or self.robot_state == "TARGET_REACHED":
            return
            
        if self.current_image is not None:
            if self.robot_state == "SEARCHING":
                self.get_logger().info("🔍 Checking for forklift while rotating...")
            elif self.robot_state == "APPROACHING":
                self.get_logger().info("🔍 Monitoring forklift while approaching...")
            
            with self.image_lock:
                image_copy = self.current_image.copy()
            
            forklift_found, response = self.detect_forklift(image_copy)
            
            if forklift_found:
                # Calculate distance to forklift using depth image
                forklift_distance = self.calculate_forklift_distance()
                self.current_forklift_distance = forklift_distance
                
                if self.robot_state == "SEARCHING":
                    # First detection - switch to approaching mode
                    self.get_logger().info("🎉 FORKLIFT DETECTED! Switching to approach mode.")
                    self.get_logger().info(f"Detection details: {response}")
                    
                    if forklift_distance is not None:
                        self.get_logger().info(f"🎯 FORKLIFT DISTANCE: {forklift_distance:.2f} meters")
                        
                        # Save depth analysis image
                        depth_path = self.save_depth_analysis_image(forklift_distance)
                        if depth_path:
                            self.get_logger().info(f"Depth analysis image saved to: {depth_path}")
                        
                        # Check if already at target distance
                        if self.check_target_distance():
                            self.robot_state = "TARGET_REACHED"
                            self.target_reached = True
                            self.stop_robot()
                            self.get_logger().info(f"🎉 Already at target distance! {forklift_distance:.2f}m")
                            self.detection_timer.cancel()
                            self.control_timer.cancel()
                        else:
                            # Start approaching
                            self.robot_state = "APPROACHING"
                            self.forklift_detected = True
                            self.get_logger().info(f"Starting approach to forklift. Target: {self.target_distance}m")
                    else:
                        self.get_logger().warn("⚠️  Could not calculate distance to forklift, continuing search")
                        return
                    
                    # Save the detection image
                    saved_path = self.save_forklift_image(image_copy, response)
                    if saved_path:
                        self.get_logger().info(f"Detection image saved to: {saved_path}")
                
                elif self.robot_state == "APPROACHING":
                    # Continue monitoring distance during approach
                    if forklift_distance is not None:
                        self.get_logger().info(f"Distance update: {forklift_distance:.2f}m")
                        
                        # Check if target reached during approach
                        if self.check_target_distance():
                            self.robot_state = "TARGET_REACHED"
                            self.target_reached = True
                            self.stop_robot()
                            self.get_logger().info(f"🎉 Target distance reached during approach!")
                            self.detection_timer.cancel()
                            self.control_timer.cancel()
                    else:
                        self.get_logger().warn("Lost distance reading during approach, stopping for safety")
                        self.stop_robot()
                        
            else:
                if self.robot_state == "SEARCHING":
                    self.get_logger().info("No forklift detected, continuing rotation...")
                elif self.robot_state == "APPROACHING":
                    self.get_logger().warn("⚠️  Lost sight of forklift during approach! Stopping for safety.")
                    self.stop_robot()
                    self.robot_state = "SEARCHING"  # Go back to search mode
                    self.forklift_detected = False
        else:
            self.get_logger().warn("No image available for detection")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        controller = Go2ForkliftController()
        
        controller.get_logger().info("Starting forklift detection and approach mission...")
        controller.get_logger().info(f"Phase 1: Search - Rotation speed: {controller.rotation_speed} rad/s ({controller.rotation_direction})")
        controller.get_logger().info(f"Phase 2: Approach - Speed: {controller.approach_speed} m/s to {controller.target_distance}m distance")
        controller.get_logger().info(f"Detection interval: {controller.detection_interval} seconds")
        
        # Spin the node
        rclpy.spin(controller)
        
    except KeyboardInterrupt:
        controller.get_logger().info("Mission interrupted by user")
    except Exception as e:
        controller.get_logger().error(f"Controller failed: {e}")
    finally:
        if 'controller' in locals():
            controller.stop_robot()
            controller.get_logger().info("Mission completed!")
            controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


# ============================================================================
# ROS2 LAUNCH FILE EXAMPLE (save as go2_forklift_controller.launch.py)
# ============================================================================

"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='your_package_name',  # Replace with your actual package name
            executable='controller.py',
            name='go2_forklift_controller',
            output='screen',
            parameters=[{
                'vllm_server_url': 'http://localhost:8000',
                'model_name': 'llava-hf/llava-1.5-7b-hf',
                'rotation_speed': 0.3,  # rad/s for rotation in place
                'approach_speed': 0.2,  # m/s for moving towards forklift
                'target_distance': 1.0,  # meters - target distance to forklift
                'distance_tolerance': 0.1,  # meters - tolerance for target distance
                'detection_interval': 2.0,  # seconds
                'image_topic': '/unitree_go2/front_cam/color_image',
                'depth_topic': '/unitree_go2/front_cam/depth_image',
                'cmd_vel_topic': '/cmd_vel',
                'save_detection_image': True,  # whether to save detected image
                'output_dir': './forklift_detections',  # directory to save images
                'rotation_direction': 'left'  # 'left' or 'right'
            }]
        )
    ])
"""

# ============================================================================
# USAGE INSTRUCTIONS FOR ROS2
# ============================================================================

"""
SETUP INSTRUCTIONS:

1. Start vLLM Server on GPU 1 (Terminal 1):
   CUDA_VISIBLE_DEVICES=1 vllm serve llava-hf/llava-1.5-7b-hf \
       --host 0.0.0.0 \
       --port 8000 \
       --trust-remote-code \
       --max-model-len 4096

2. Start Isaac Sim with Go-2 robot (uses GPU 0)

3. Run the controller (Terminal 2):
   # Direct execution:
   python3 controller.py
   
   # OR with ROS2 run:
   ros2 run your_package_name controller
   
   # OR with ROS2 launch:
   ros2 launch your_package_name go2_forklift_controller.launch.py

PARAMETERS (can be set via launch file or command line):
ros2 run your_package_name controller --ros-args \
    -p rotation_speed:=0.2 \
    -p approach_speed:=0.15 \
    -p target_distance:=1.5 \
    -p distance_tolerance:=0.05 \
    -p detection_interval:=1.5 \
    -p rotation_direction:=right

KEY FEATURES:
- Phase 1: SEARCHING - Robot rotates in place until forklift detected
- Phase 2: APPROACHING - Robot moves forward towards forklift
- Phase 3: TARGET_REACHED - Robot stops at target distance (1m by default)
- Continuous distance monitoring during approach
- Safety stops if forklift lost during approach
- Configurable target distance and tolerance
"""