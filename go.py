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
import re

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
        
        # Depth image subscriber
        self.depth_sub = self.create_subscription(
            Image,
            '/unitree_go2/front_cam/depth_image',
            self.depth_callback,
            10
        )
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # VLLM server configuration
        self.vllm_url = "http://localhost:8000/v1/chat/completions"
        
        # Control variables
        self.current_image = None
        self.current_depth_image = None
        self.forklift_detected = False
        self.forklift_centered = False
        self.distance_measured = False
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
        
        # Distance measurement variables
        self.forklift_distance = None
        self.forklift_bbox = None
        
        # Timer for periodic operations
        self.detection_timer = self.create_timer(
            self.detection_interval, 
            self.check_for_forklift
        )
        
        # Timer for movement control
        self.movement_timer = self.create_timer(0.1, self.movement_control)
        
        self.get_logger().info("Go-2 Forklift Controller with Depth Measurement initialized")
        self.get_logger().info("Starting rotation to search for forklift...")
        
    def image_callback(self, msg):
        """Callback for receiving camera images"""
        try:
            # Convert ROS image to OpenCV format
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
    
    def depth_callback(self, msg):
        """Callback for receiving depth images"""
        try:
            # Convert ROS depth image to OpenCV format
            # Depth images are typically in 16UC1 or 32FC1 format
            if msg.encoding == "16UC1":
                self.current_depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            elif msg.encoding == "32FC1":
                self.current_depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            else:
                # Try passthrough for other formats
                self.current_depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}")
    
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
    
    def get_forklift_bounding_box(self, image):
        """Get bounding box coordinates of the forklift using VLLM with percentage values"""
        try:
            # Encode image
            image_base64 = self.encode_image_to_base64(image)
            if not image_base64:
                return None
            
            # Prepare the request payload for bounding box detection with percentage values
            payload = {
                "model": "llava-hf/llava-1.5-7b-hf",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Look at this image and identify the forklift. Please provide the bounding box of the forklift as percentage values of the image dimensions. Format: x1_percent,y1_percent,x2_percent,y2_percent where (x1,y1) is the top-left corner and (x2,y2) is the bottom-right corner. Each value should be between 0 and 100 representing the percentage from left/top of the image. Only provide the four percentage numbers separated by commas, nothing else. For example: 25.5,30.2,75.8,85.1"
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
                
                self.get_logger().info(f"Bounding Box Percentage Response: {answer}")
                return self.parse_percentage_bounding_box(answer, image.shape, image)
            else:
                self.get_logger().error(f"VLLM server error: {response.status_code}")
                return None
                
        except Exception as e:
            self.get_logger().error(f"Error in bounding box detection: {e}")
            return None
    
    def parse_percentage_bounding_box(self, response, image_shape, img):
        """Parse the percentage bounding box response and convert to pixel coordinates"""
        try:
            # Extract floating point numbers from the response
            numbers = re.findall(r'\d+\.?\d*', response)
            
            if len(numbers) < 4:
                self.get_logger().error(f"Not enough coordinates found in response: {response}")
                return None
            
            # Take the first 4 numbers as percentage coordinates
            x1_pct, y1_pct, x2_pct, y2_pct = map(float, numbers[:4])
            
            self.get_logger().info(f"Parsed percentage coordinates: ({x1_pct:.1f}%, {y1_pct:.1f}%, {x2_pct:.1f}%, {y2_pct:.1f}%)")
            
            # Get image dimensions
            height, width = image_shape[:2]
            
            # Convert percentages to pixel coordinates
            x1 = int((x1_pct / 100.0) * width)
            y1 = int((y1_pct / 100.0) * height)
            x2 = int((x2_pct / 100.0) * width)
            y2 = int((y2_pct / 100.0) * height)
            
            # Validate and clamp coordinates
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # Ensure x2 > x1 and y2 > y1
            if x2 <= x1 or y2 <= y1:
                self.get_logger().error(f"Invalid bounding box coordinates: ({x1},{y1},{x2},{y2})")
                return None
            
            bbox = {
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'width': x2 - x1,
                'height': y2 - y1,
                'x1_pct': x1_pct,
                'y1_pct': y1_pct,
                'x2_pct': x2_pct,
                'y2_pct': y2_pct
            }

            if bbox:
                # Draw rectangle
                img_with_bbox = img.copy()
                cv2.rectangle(
                    img_with_bbox,
                    (bbox['x1'], bbox['y1']),
                    (bbox['x2'], bbox['y2']),
                    (0, 255, 0), 3
                )
                # Save image with bounding box
                save_path = "forklift_bbox.jpg"
                cv2.imwrite(save_path, img_with_bbox)
                self.get_logger().info(f"Saved image with bounding box to {save_path}")
            
            self.get_logger().info(f"Converted to pixel coordinates: ({x1},{y1},{x2},{y2}) - Size: {bbox['width']}x{bbox['height']} pixels")
            return bbox
            
        except Exception as e:
            self.get_logger().error(f"Error parsing percentage bounding box: {e}")
            return None
    
    def measure_forklift_distance(self):
        """Measure distance to forklift using depth image and percentage-based bounding box"""
        if self.current_depth_image is None or self.forklift_bbox is None:
            self.get_logger().warn("Cannot measure distance - missing depth image or bounding box")
            return None
        
        try:
            # Get depth image dimensions
            depth_height, depth_width = self.current_depth_image.shape[:2]
            
            # Calculate pixel coordinates directly from percentages for depth image
            x1 = int((self.forklift_bbox['x1_pct'] / 100.0) * depth_width)
            y1 = int((self.forklift_bbox['y1_pct'] / 100.0) * depth_height)
            x2 = int((self.forklift_bbox['x2_pct'] / 100.0) * depth_width)
            y2 = int((self.forklift_bbox['y2_pct'] / 100.0) * depth_height)
            
            # Validate and clamp coordinates for depth image
            x1 = max(0, min(x1, depth_width - 1))
            y1 = max(0, min(y1, depth_height - 1))
            x2 = max(0, min(x2, depth_width - 1))
            y2 = max(0, min(y2, depth_height - 1))
            
            # Ensure valid bounding box
            if x2 <= x1 or y2 <= y1:
                self.get_logger().error(f"Invalid depth bounding box: ({x1},{y1},{x2},{y2})")
                return None
            
            self.get_logger().info(f"Depth image bounding box: ({x1},{y1},{x2},{y2}) - Size: {x2-x1}x{y2-y1} pixels")
            self.get_logger().info(f"Depth image dimensions: {depth_width}x{depth_height}")
            
            # Extract the region of interest from depth image
            roi_depth = self.current_depth_image[y1:y2, x1:x2]
            
            if roi_depth.size == 0:
                self.get_logger().error("Empty ROI in depth image")
                return None
            
            # Filter out zero/invalid depth values
            valid_depths = roi_depth[roi_depth > 0]
            
            if len(valid_depths) == 0:
                self.get_logger().warn("No valid depth values in forklift region")
                return None
            
            # Calculate median distance (more robust than mean)
            median_distance = np.median(valid_depths)
            
            # Calculate additional statistics
            min_distance = np.min(valid_depths)
            max_distance = np.max(valid_depths)
            mean_distance = np.mean(valid_depths)
            
            # Convert to meters if the depth is in millimeters (common for depth cameras)
            if median_distance > 100:  # Assume millimeters if value > 100
                distance_meters = median_distance / 1000.0
                min_meters = min_distance / 1000.0
                max_meters = max_distance / 1000.0
                mean_meters = mean_distance / 1000.0
                unit = "mm->m"
            else:  # Assume already in meters
                distance_meters = median_distance
                min_meters = min_distance
                max_meters = max_distance
                mean_meters = mean_distance
                unit = "m"
            
            self.get_logger().info(f"📏 FORKLIFT DISTANCE MEASURED:")
            self.get_logger().info(f"   Percentage bbox: ({self.forklift_bbox['x1_pct']:.1f}%, {self.forklift_bbox['y1_pct']:.1f}%, {self.forklift_bbox['x2_pct']:.1f}%, {self.forklift_bbox['y2_pct']:.1f}%)")
            self.get_logger().info(f"   Depth bbox pixels: ({x1},{y1},{x2},{y2})")
            self.get_logger().info(f"   Median distance: {distance_meters:.2f} meters ({unit})")
            self.get_logger().info(f"   Distance range: {min_meters:.2f} - {max_meters:.2f} meters")
            self.get_logger().info(f"   Mean distance: {mean_meters:.2f} meters")
            self.get_logger().info(f"   Valid depth points: {len(valid_depths)}/{roi_depth.size}")
            self.get_logger().info(f"   Region size: {x2-x1}x{y2-y1} pixels")
            
            return distance_meters
            
        except Exception as e:
            self.get_logger().error(f"Error measuring forklift distance: {e}")
            return None
    
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
            # Proceed to distance measurement
            self.start_distance_measurement()
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
            # Proceed to distance measurement
            self.start_distance_measurement()
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
    
    def start_distance_measurement(self):
        """Start the distance measurement process"""
        if self.distance_measured:
            return
        
        self.get_logger().info("🎯 Starting distance measurement process...")
        
        # Get forklift bounding box
        self.forklift_bbox = self.get_forklift_bounding_box(self.current_image)
        
        if self.forklift_bbox is None:
            self.get_logger().error("Failed to get forklift bounding box. Cannot measure distance.")
            return
        
        # Measure distance using depth image
        self.forklift_distance = self.measure_forklift_distance()
        
        if self.forklift_distance is not None:
            self.distance_measured = True
            self.get_logger().info(f"✅ TASK COMPLETE! Forklift distance: {self.forklift_distance:.2f} meters")
        else:
            self.get_logger().error("Failed to measure forklift distance")
    
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