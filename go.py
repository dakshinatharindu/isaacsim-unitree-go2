#!/usr/bin/env python3
"""
Go-2 Robot Forklift Detection Controller for Isaac Sim (ROS2 Humble)
Moves robot forward until it detects a forklift using vLLM vision model.
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

class Go2ForkliftController(Node):
    def __init__(self):
        super().__init__('go2_forklift_controller')
        
        # Declare parameters
        self.declare_parameter('vllm_server_url', 'http://localhost:8000')
        self.declare_parameter('model_name', 'llava-hf/llava-1.5-7b-hf')
        self.declare_parameter('forward_speed', 0.5)  # m/s
        self.declare_parameter('detection_interval', 2.0)  # seconds
        self.declare_parameter('image_topic', '/unitree_go2/front_cam/color_image')
        self.declare_parameter('cmd_vel_topic', '/unitree_go2/cmd_vel')
        
        # Get parameters
        self.vllm_server_url = self.get_parameter('vllm_server_url').get_parameter_value().string_value
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self.forward_speed = self.get_parameter('forward_speed').get_parameter_value().double_value
        self.detection_interval = self.get_parameter('detection_interval').get_parameter_value().double_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        
        # State variables
        self.current_image = None
        self.forklift_detected = False
        self.running = True
        self.bridge = CvBridge()
        self.image_lock = threading.Lock()
        
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
        
        self.last_detection_time = time.time()
        
        # Wait for first image
        self.get_logger().info("Waiting for camera feed...")
        self.wait_for_image()
        
        # Test vLLM server connection
        self.test_server_connection()
        
        self.get_logger().info("Go-2 Forklift Controller initialized successfully!")
        
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
        """Send command to move robot forward."""
        twist = Twist()
        twist.linear.x = self.forward_speed
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
        """Main control loop callback (10Hz)."""
        if not self.running or self.forklift_detected:
            self.stop_robot()
            return
            
        # Move forward
        self.move_forward()
    
    def detection_callback(self):
        """Forklift detection callback."""
        if not self.running or self.forklift_detected:
            return
            
        if self.current_image is not None:
            self.get_logger().info("Checking for forklift...")
            
            with self.image_lock:
                image_copy = self.current_image.copy()
            
            forklift_found, response = self.detect_forklift(image_copy)
            
            if forklift_found:
                self.get_logger().info("🎉 FORKLIFT DETECTED! Stopping robot.")
                self.get_logger().info(f"Detection details: {response}")
                self.forklift_detected = True
                self.stop_robot()
                # Stop the detection timer
                self.detection_timer.cancel()
                # Optionally stop control timer too
                self.control_timer.cancel()
            else:
                self.get_logger().info("No forklift detected, continuing forward...")
        else:
            self.get_logger().warn("No image available for detection")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        controller = Go2ForkliftController()
        
        controller.get_logger().info("Starting forklift detection mission...")
        controller.get_logger().info(f"Moving forward at {controller.forward_speed} m/s")
        controller.get_logger().info(f"Checking for forklift every {controller.detection_interval} seconds")
        
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
                'forward_speed': 0.5,  # m/s
                'detection_interval': 2.0,  # seconds
                'image_topic': '/unitree_go2/front_cam/color_image',
                'cmd_vel_topic': '/cmd_vel'
            }]
        )
    ])
"""

# ============================================================================
# PACKAGE.XML DEPENDENCIES
# ============================================================================

"""
Add these dependencies to your package.xml:

<depend>rclpy</depend>
<depend>sensor_msgs</depend>
<depend>geometry_msgs</depend>
<depend>cv_bridge</depend>
<depend>opencv2</depend>
"""

# ============================================================================
# SETUP.PY CONFIGURATION
# ============================================================================

"""
Add this to your setup.py entry_points:

'console_scripts': [
    'controller = your_package_name.controller:main',
],
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

4. Monitor topics (optional):
   # Check if image topic is publishing
   ros2 topic echo /unitree_go2/front_cam/color_image
   
   # Check cmd_vel output
   ros2 topic echo /cmd_vel
   
   # List all topics
   ros2 topic list

PARAMETERS (can be set via launch file or command line):
ros2 run your_package_name controller --ros-args -p forward_speed:=0.3 -p detection_interval:=1.5

ROS2 SPECIFIC FEATURES:
- Uses QoS profiles for reliable image subscription
- Timer-based control loop (10Hz) and detection loop (configurable interval)
- Proper parameter declaration and retrieval
- ROS2 logging system
- Clean node lifecycle management
"""