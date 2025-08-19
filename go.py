#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import requests
import base64
import numpy as np
import time
import json

class Go2ForkliftController(Node):
    def __init__(self):
        super().__init__('go2_forklift_controller')
        
        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/unitree_go2/cmd_vel', 10)
        self.image_sub = self.create_subscription(
            Image, 
            '/unitree_go2/front_cam/color_image', 
            self.image_callback, 
            10
        )
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # VLLM server configuration
        self.vllm_url = "http://localhost:8000/v1/chat/completions"
        
        # Control variables
        self.current_image = None
        self.forklift_detected = False
        self.is_moving = False
        
        # Movement parameters
        self.angular_velocity = 0.5  # rad/s for rotation
        self.detection_interval = 2.0  # seconds between detections
        
        # Timer for periodic forklift detection
        self.detection_timer = self.create_timer(
            self.detection_interval, 
            self.check_for_forklift
        )
        
        # Timer for movement control
        self.movement_timer = self.create_timer(0.1, self.movement_control)
        
        self.get_logger().info("Go-2 Forklift Controller initialized")
        self.get_logger().info("Starting rotation to search for forklift...")
        
    def image_callback(self, msg):
        """Callback for receiving camera images"""
        try:
            # Convert ROS image to OpenCV format
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
    
    def encode_image_to_base64(self, image):
        """Convert OpenCV image to base64 string for VLLM API"""
        try:
            # Resize image to reduce payload size (optional)
            height, width = image.shape[:2]
            if width > 800:
                scale = 800 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            # Encode image as JPEG
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            # Convert to base64
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
            
        except Exception as e:
            self.get_logger().error(f"Error encoding image: {e}")
            return None
    
    def detect_forklift_with_vllm(self, image):
        """Use VLLM server to detect forklift in image"""
        try:
            # Encode image
            image_base64 = self.encode_image_to_base64(image)
            if not image_base64:
                return False
            
            # Prepare the request payload
            payload = {
                "model": "llava-hf/llava-1.5-7b-hf",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Look at this image carefully. Is there a forklift visible in this image? Answer with only 'YES' if you can see a forklift, or 'NO' if you cannot see a forklift. Be very specific - a forklift is an industrial vehicle with two prongs/forks at the front for lifting pallets."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_base64
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 10,
                "temperature": 0.1
            }
            
            # Send request to VLLM server
            response = requests.post(
                self.vllm_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content'].strip().upper()
                
                self.get_logger().info(f"VLLM Response: {answer}")
                
                # Check if forklift is detected
                return "YES" in answer
            else:
                self.get_logger().error(f"VLLM server error: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Request to VLLM server failed: {e}")
            return False
        except Exception as e:
            self.get_logger().error(f"Error in forklift detection: {e}")
            return False
    
    def check_for_forklift(self):
        """Periodic check for forklift in current image"""
        if self.current_image is not None and not self.forklift_detected:
            self.get_logger().info("Checking for forklift...")
            
            # Detect forklift using VLLM
            if self.detect_forklift_with_vllm(self.current_image):
                self.get_logger().info("🎯 FORKLIFT DETECTED! Stopping robot.")
                self.forklift_detected = True
                self.stop_robot()
            else:
                self.get_logger().info("No forklift detected, continuing search...")
    
    def movement_control(self):
        """Control robot movement"""
        if not self.forklift_detected:
            # Rotate the robot to search for forklift
            cmd = Twist()
            cmd.angular.z = self.angular_velocity
            self.cmd_vel_pub.publish(cmd)
            
            if not self.is_moving:
                self.get_logger().info("Rotating to search for forklift...")
                self.is_moving = True
    
    def stop_robot(self):
        """Stop the robot completely"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = 0.0
        
        # Send stop command multiple times to ensure it's received
        for _ in range(5):
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.1)
        
        self.get_logger().info("Robot stopped successfully!")

def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)
    
    try:
        # Create controller node
        controller = Go2ForkliftController()
        
        # Spin the node
        rclpy.spin(controller)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        if 'controller' in locals():
            controller.stop_robot()
            controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()