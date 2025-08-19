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
        self.forklift_centered = False
        self.is_moving = False
        self.search_phase = True  # True = searching, False = centering
        
        # Movement parameters
        self.angular_velocity = 0.5  # rad/s for rotation during search
        self.centering_angular_velocity = 0.15  # slower rotation for centering
        self.linear_velocity = 0.15   # m/s for forward/backward during centering
        self.detection_interval = 2.0  # seconds between detections during search
        self.centering_interval = 1.5  # seconds between centering attempts
        
        # Centering parameters
        self.max_centering_attempts = 15  # max attempts before declaring success
        self.current_centering_attempts = 0
        
        # Movement step sizes for centering
        self.rotation_step_time = 0.8  # seconds to rotate in one direction
        self.movement_step_time = 0.6  # seconds to move forward/backward
        
        # Current centering state
        self.centering_action = None  # 'rotate_left', 'rotate_right', 'move_forward', 'move_backward', 'stop'
        self.action_start_time = 0
        
        # Timer for periodic operations
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
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Convert to base64
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
            
        except Exception as e:
            self.get_logger().error(f"Error encoding image: {e}")
            return None
    
    def detect_forklift_simple(self, image):
        """Simple forklift detection - just check if forklift is present"""
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
                                "text": "Look at this image. Is there a forklift visible? Answer with only 'YES' if you can see a forklift, or 'NO' if you cannot see a forklift. A forklift is an industrial vehicle with two prongs/forks at the front for lifting pallets."
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
                
                self.get_logger().info(f"Forklift Detection: {answer}")
                return "YES" in answer
            else:
                self.get_logger().error(f"VLLM server error: {response.status_code}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Error in forklift detection: {e}")
            return False
    
    def analyze_forklift_position(self, image):
        """Analyze forklift position and distance for centering guidance"""
        try:
            # Encode image
            image_base64 = self.encode_image_to_base64(image)
            if not image_base64:
                return None
            
            # Prepare the request payload for position analysis
            payload = {
                "model": "llava-hf/llava-1.5-7b-hf",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Look at this image with a forklift. Answer these questions:\n1. Is the forklift positioned more towards the LEFT, RIGHT, or CENTER of the image?\n2. Does the forklift appear CLOSE, MEDIUM, or FAR from the camera?\n3. Is the forklift well-centered in the image? Answer YES or NO.\n\nProvide your answer in this exact format:\nPosition: [LEFT/RIGHT/CENTER]\nDistance: [CLOSE/MEDIUM/FAR]\nCentered: [YES/NO]"
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
                "max_tokens": 50,
                "temperature": 0.1
            }
            
            # Send request to VLLM server
            response = requests.post(
                self.vllm_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=12
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content'].strip()
                
                self.get_logger().info(f"Position Analysis: {answer}")
                return self.parse_position_analysis(answer)
            else:
                self.get_logger().error(f"VLLM server error: {response.status_code}")
                return None
                
        except Exception as e:
            self.get_logger().error(f"Error in position analysis: {e}")
            return None
    
    def parse_position_analysis(self, response):
        """Parse the position analysis response"""
        try:
            analysis = {
                'position': 'CENTER',
                'distance': 'MEDIUM', 
                'centered': False
            }
            
            response_upper = response.upper()
            
            # Parse position
            if 'LEFT' in response_upper:
                analysis['position'] = 'LEFT'
            elif 'RIGHT' in response_upper:
                analysis['position'] = 'RIGHT'
            else:
                analysis['position'] = 'CENTER'
            
            # Parse distance
            if 'CLOSE' in response_upper:
                analysis['distance'] = 'CLOSE'
            elif 'FAR' in response_upper:
                analysis['distance'] = 'FAR'
            else:
                analysis['distance'] = 'MEDIUM'
            
            # Parse centered status
            if 'CENTERED: YES' in response_upper or 'CENTERED:YES' in response_upper:
                analysis['centered'] = True
            
            return analysis
            
        except Exception as e:
            self.get_logger().error(f"Error parsing position analysis: {e}")
            return None
    
    def determine_centering_action(self, analysis):
        """Determine what action to take based on position analysis"""
        if not analysis:
            return 'stop'
        
        # Check if already centered
        if analysis['centered']:
            return 'stop'
        
        # Prioritize horizontal centering first
        if analysis['position'] == 'LEFT':
            return 'rotate_left'
        elif analysis['position'] == 'RIGHT':
            return 'rotate_right'
        
        # If horizontally centered, adjust distance
        if analysis['position'] == 'CENTER':
            if analysis['distance'] == 'CLOSE':
                return 'move_backward'
            elif analysis['distance'] == 'FAR':
                return 'move_forward'
            else:
                return 'stop'  # Well centered
        
        return 'stop'
    
    def check_for_forklift(self):
        """Periodic check for forklift in current image"""
        if self.current_image is None:
            return
        
        if self.search_phase and not self.forklift_detected:
            # Search phase - look for forklift
            self.get_logger().info("Searching for forklift...")
            
            if self.detect_forklift_simple(self.current_image):
                self.get_logger().info("🎯 FORKLIFT DETECTED! Starting centering process...")
                self.forklift_detected = True
                self.search_phase = False
                self.current_centering_attempts = 0
                
                # Switch to centering mode
                self.detection_timer.destroy()
                self.detection_timer = self.create_timer(
                    self.centering_interval, 
                    self.center_forklift
                )
                
                # Stop current rotation movement
                self.stop_robot()
                time.sleep(0.5)  # Brief pause before starting centering
            else:
                self.get_logger().info("No forklift detected, continuing search...")
    
    def center_forklift(self):
        """Center the forklift using incremental movements with LLM guidance"""
        if self.current_image is None or self.forklift_centered:
            return
        
        # Check if we've exceeded max attempts
        if self.current_centering_attempts >= self.max_centering_attempts:
            self.get_logger().info("🎉 CENTERING COMPLETE! (Max attempts reached)")
            self.forklift_centered = True
            self.stop_robot()
            self.detection_timer.destroy()
            return
        
        # First, verify forklift is still visible
        if not self.detect_forklift_simple(self.current_image):
            self.get_logger().warn("Lost forklift during centering! Returning to search mode...")
            self.forklift_detected = False
            self.search_phase = True
            self.centering_action = None
            
            # Switch back to search timer
            self.detection_timer.destroy()
            self.detection_timer = self.create_timer(
                self.detection_interval, 
                self.check_for_forklift
            )
            return
        
        # Analyze current position
        analysis = self.analyze_forklift_position(self.current_image)
        if not analysis:
            self.get_logger().warn("Could not analyze forklift position, retrying...")
            return
        
        # Check if centering is complete
        if analysis['centered']:
            self.get_logger().info("🎉 FORKLIFT SUCCESSFULLY CENTERED!")
            self.forklift_centered = True
            self.stop_robot()
            self.detection_timer.destroy()
            return
        
        # Determine next action
        action = self.determine_centering_action(analysis)
        
        self.get_logger().info(
            f"Centering attempt {self.current_centering_attempts + 1}/{self.max_centering_attempts} - "
            f"Position: {analysis['position']}, Distance: {analysis['distance']}, "
            f"Action: {action}"
        )
        
        # Execute the centering action
        self.execute_centering_action(action)
        self.current_centering_attempts += 1
    
    def execute_centering_action(self, action):
        """Execute a specific centering action"""
        self.centering_action = action
        self.action_start_time = time.time()
        
        if action == 'rotate_left':
            self.get_logger().info("📍 Rotating left to center forklift...")
        elif action == 'rotate_right':
            self.get_logger().info("📍 Rotating right to center forklift...")
        elif action == 'move_forward':
            self.get_logger().info("📍 Moving forward to adjust distance...")
        elif action == 'move_backward':
            self.get_logger().info("📍 Moving backward to adjust distance...")
        elif action == 'stop':
            self.get_logger().info("📍 Forklift appears centered, stopping...")
            self.centering_action = None
    
    def movement_control(self):
        """Control robot movement based on current phase and action"""
        current_time = time.time()
        
        if self.search_phase and not self.forklift_detected:
            # Search phase - rotate to find forklift
            cmd = Twist()
            cmd.angular.z = self.angular_velocity
            self.cmd_vel_pub.publish(cmd)
            
            if not self.is_moving:
                self.get_logger().info("Rotating to search for forklift...")
                self.is_moving = True
                
        elif self.centering_action and not self.forklift_centered:
            # Centering phase - execute specific actions
            cmd = Twist()
            action_duration = 0
            
            if self.centering_action == 'rotate_left':
                cmd.angular.z = self.centering_angular_velocity
                action_duration = self.rotation_step_time
            elif self.centering_action == 'rotate_right':
                cmd.angular.z = -self.centering_angular_velocity
                action_duration = self.rotation_step_time
            elif self.centering_action == 'move_forward':
                cmd.linear.x = self.linear_velocity
                action_duration = self.movement_step_time
            elif self.centering_action == 'move_backward':
                cmd.linear.x = -self.linear_velocity
                action_duration = self.movement_step_time
            
            # Execute action for specified duration
            if current_time - self.action_start_time < action_duration:
                self.cmd_vel_pub.publish(cmd)
            else:
                # Action completed, stop and wait for next analysis
                self.stop_robot()
                self.centering_action = None
    
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
        for _ in range(3):
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.05)
        
        self.is_moving = False

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