#!/usr/bin/env python3
"""
Go-2 Robot Forklift Detection Controller for Isaac Sim (ROS2 Humble)
1. Rotates in place until it detects a forklift using vLLM vision model
2. Once detected, moves towards the forklift
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
        self.declare_parameter('forward_speed', 0.3)  # m/s - reduced for approach
        self.declare_parameter('rotation_speed', 0.4)  # rad/s for searching rotation
        self.declare_parameter('approach_speed', 0.2)  # m/s - slower approach speed
        self.declare_parameter('detection_interval', 2.0)  # seconds
        self.declare_parameter('approach_timeout', 15.0)  # seconds to approach before stopping
        self.declare_parameter('image_topic', '/unitree_go2/front_cam/color_image')
        self.declare_parameter('cmd_vel_topic', '/unitree_go2/cmd_vel')
        self.declare_parameter('save_detection_image', True)  # whether to save detected image
        self.declare_parameter('output_dir', './forklift_detections')  # directory to save images
        self.declare_parameter('stop_distance_check', False)  # whether to check if close enough to stop
        
        # Get parameters
        self.vllm_server_url = self.get_parameter('vllm_server_url').get_parameter_value().string_value
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self.forward_speed = self.get_parameter('forward_speed').get_parameter_value().double_value
        self.rotation_speed = self.get_parameter('rotation_speed').get_parameter_value().double_value
        self.approach_speed = self.get_parameter('approach_speed').get_parameter_value().double_value
        self.detection_interval = self.get_parameter('detection_interval').get_parameter_value().double_value
        self.approach_timeout = self.get_parameter('approach_timeout').get_parameter_value().double_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.save_detection_image = self.get_parameter('save_detection_image').get_parameter_value().bool_value
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self.stop_distance_check = self.get_parameter('stop_distance_check').get_parameter_value().bool_value
        
        # State variables
        self.current_image = None
        self.forklift_detected = False
        self.running = True
        self.bridge = CvBridge()
        self.image_lock = threading.Lock()
        
        # Robot behavior state
        self.robot_state = "SEARCHING"  # "SEARCHING", "APPROACHING", "ARRIVED"
        self.approach_start_time = None
        
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
        
        # Timer for main control loop (10Hz)
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        # Timer for forklift detection
        self.detection_timer = self.create_timer(self.detection_interval, self.detection_callback)
        
        # Wait for first image
        self.get_logger().info("Waiting for camera feed...")
        self.wait_for_image()
        
        # Test vLLM server connection
        self.test_server_connection()
        
        self.get_logger().info("Go-2 Forklift Controller initialized successfully!")
        self.get_logger().info("Behavior: Search by rotating -> Approach forklift when found")
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
    
    def save_forklift_image(self, cv_image, vllm_response, state):
        """Save the image where forklift was detected."""
        if not self.save_detection_image:
            return None
            
        try:
            # Generate filename with timestamp and state
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forklift_{state.lower()}_{timestamp}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            # Add text overlay with detection info
            overlay_image = cv_image.copy()
            
            # Add timestamp
            cv2.putText(overlay_image, f"Detected: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add robot state
            cv2.putText(overlay_image, f"State: {state}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Add detection response (truncated if too long)
            response_text = vllm_response[:50] + "..." if len(vllm_response) > 50 else vllm_response
            cv2.putText(overlay_image, f"Response: {response_text}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
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
            
            # Prepare request with different prompts based on state
            headers = {"Content-Type": "application/json"}
            
            if self.robot_state == "SEARCHING":
                prompt = ("Look at this image carefully. Is there a forklift visible in the image? "
                         "Answer with 'YES' if you can see a forklift anywhere in the image, or 'NO' if you cannot see a forklift. "
                         "Be specific and only answer YES or NO followed by a brief explanation.")
            else:  # APPROACHING
                prompt = ("Look at this image carefully. Is there a forklift visible in the image? "
                         "Answer with 'YES' if you can still see the forklift, or 'NO' if you lost sight of it. "
                         "If YES, also describe if the forklift appears close or far away. "
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
                
                self.get_logger().info(f"vLLM Response ({self.robot_state}): {response_text}")
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
    
    def rotate_in_place(self):
        """Send command to rotate robot in place (counterclockwise)."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = self.rotation_speed  # Positive Z = counterclockwise rotation
        
        self.cmd_vel_pub.publish(twist)
    
    def move_towards_forklift(self):
        """Send command to move robot forward towards the forklift."""
        twist = Twist()
        twist.linear.x = self.approach_speed
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        
        self.cmd_vel_pub.publish(twist)
    
    def stop_robot(self):
        """Send command to stop robot."""
        twist = Twist()  # All zeros
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info("Robot stopped!")
    
    def control_loop(self):
        """Main control loop callback (10Hz) - implements search and approach behavior."""
        if not self.running:
            self.stop_robot()
            return
        
        current_time = time.time()
        
        if self.robot_state == "SEARCHING":
            # Rotate in place to search for forklift
            self.rotate_in_place()
            
        elif self.robot_state == "APPROACHING":
            # Move towards the forklift
            self.move_towards_forklift()
            
            # Check for approach timeout
            if self.approach_start_time and (current_time - self.approach_start_time >= self.approach_timeout):
                self.get_logger().info("Approach timeout reached, stopping robot.")
                self.robot_state = "ARRIVED"
                self.stop_robot()
                self.detection_timer.cancel()
                self.control_timer.cancel()
                
        elif self.robot_state == "ARRIVED":
            # Stop the robot
            self.stop_robot()
    
    def detection_callback(self):
        """Forklift detection callback."""
        if not self.running or self.robot_state == "ARRIVED":
            return
            
        if self.current_image is not None:
            self.get_logger().info(f"Checking for forklift... (State: {self.robot_state})")
            
            with self.image_lock:
                image_copy = self.current_image.copy()
            
            forklift_found, response = self.detect_forklift(image_copy)
            
            if self.robot_state == "SEARCHING":
                if forklift_found:
                    self.get_logger().info("🎉 FORKLIFT DETECTED! Switching to approach mode.")
                    self.get_logger().info(f"Detection details: {response}")
                    
                    # Save the detection image
                    saved_path = self.save_forklift_image(image_copy, response, self.robot_state)
                    if saved_path:
                        self.get_logger().info(f"Detection image saved to: {saved_path}")
                    
                    # Switch to approaching state
                    self.robot_state = "APPROACHING"
                    self.approach_start_time = time.time()
                    self.forklift_detected = True
                else:
                    self.get_logger().info("No forklift detected, continuing to search (rotating)...")
                    
            elif self.robot_state == "APPROACHING":
                if forklift_found:
                    self.get_logger().info(f"Still tracking forklift, continuing approach... {response}")
                    
                    # Check if we should stop based on distance (if enabled and response indicates closeness)
                    if self.stop_distance_check and ("close" in response.lower() or "near" in response.lower()):
                        self.get_logger().info("Forklift appears close enough, stopping approach.")
                        self.robot_state = "ARRIVED"
                        self.stop_robot()
                        self.detection_timer.cancel()
                        self.control_timer.cancel()
                else:
                    self.get_logger().warn("Lost sight of forklift during approach! Switching back to search mode.")
                    self.robot_state = "SEARCHING"
                    self.approach_start_time = None
        else:
            self.get_logger().warn("No image available for detection")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        controller = Go2ForkliftController()
        
        controller.get_logger().info("Starting forklift detection mission with search and approach behavior...")
        controller.get_logger().info(f"Rotation speed: {controller.rotation_speed} rad/s, Approach speed: {controller.approach_speed} m/s")
        controller.get_logger().info(f"Behavior: Rotate in place to find forklift -> Move towards forklift when detected")
        controller.get_logger().info(f"Checking for forklift every {controller.detection_interval} seconds")
        controller.get_logger().info(f"Approach timeout: {controller.approach_timeout} seconds")
        
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
                'rotation_speed': 0.4,  # rad/s for searching rotation
                'approach_speed': 0.2,  # m/s - slower approach speed
                'detection_interval': 2.0,  # seconds
                'approach_timeout': 15.0,  # seconds to approach before stopping
                'image_topic': '/unitree_go2/front_cam/color_image',
                'cmd_vel_topic': '/cmd_vel',
                'save_detection_image': True,  # whether to save detected image
                'output_dir': './forklift_detections',  # directory to save images
                'stop_distance_check': False  # whether to check if close enough to stop
            }]
        )
    ])
"""

# ============================================================================
# USAGE INSTRUCTIONS FOR ROS2 - UPDATED BEHAVIOR
# ============================================================================

"""
UPDATED BEHAVIOR:

PHASE 1 - SEARCHING:
- Robot rotates in place (counterclockwise) at rotation_speed
- vLLM continuously analyzes camera feed every detection_interval seconds
- When forklift is detected, switches to PHASE 2

PHASE 2 - APPROACHING:
- Robot moves forward toward the detected forklift at approach_speed
- vLLM continues to track the forklift to ensure it's still visible
- If forklift is lost during approach, switches back to PHASE 1 (searching)
- Stops after approach_timeout seconds or when close enough (if stop_distance_check enabled)

SETUP INSTRUCTIONS:

1. Start vLLM Server on GPU 1 (Terminal 1):
   CUDA_VISIBLE_DEVICES=1 vllm serve llava-hf/llava-1.5-7b-hf \
       --host 0.0.0.0 \
       --port 8000 \
       --trust-remote-code \
       --max-model-len 4096

2. Start Isaac Sim with Go-2 robot (uses GPU 0)

3. Run the controller (Terminal 2):
   python3 controller.py

KEY PARAMETERS:
- rotation_speed: How fast to rotate when searching (default: 0.4 rad/s)
- approach_speed: How fast to move toward forklift (default: 0.2 m/s)
- approach_timeout: Max time to approach before stopping (default: 15.0 s)
- stop_distance_check: Whether to stop when forklift appears "close" (default: False)

MONITORING:
ros2 topic echo /cmd_vel  # Monitor robot commands
ros2 topic echo /unitree_go2/front_cam/color_image  # Check camera feed
"""