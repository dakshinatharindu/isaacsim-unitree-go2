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
import os
from datetime import datetime
from ultralytics import YOLO

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
        
        # YOLO model initialization
        try:
            # You can use different YOLO models:
            # - yolov8n.pt (nano, fastest)
            # - yolov8s.pt (small)
            # - yolov8m.pt (medium)
            # - yolov8l.pt (large)
            # - yolov8x.pt (extra large, most accurate)
            self.yolo_model = YOLO('yolov8n.pt')  # Using nano for speed
            self.get_logger().info("YOLO model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            self.yolo_model = None
        
        # Control variables
        self.current_image = None
        self.current_depth_image = None
        self.forklift_detected = False
        self.forklift_centered = False
        self.distance_measured = False
        self.is_moving = False
        self.forklift_bbox = None
        
        # Movement parameters
        self.angular_velocity = 0.5  # rad/s for rotation
        self.detection_interval = 2.0  # seconds between detections
        
        # Create output directory for saved images
        self.output_dir = "forklift_detections"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Timer for periodic forklift detection
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
    
    def detect_forklift_with_yolo(self, image):
        """Use YOLO to detect forklift and get bounding box"""
        if self.yolo_model is None:
            self.get_logger().warn("YOLO model not available")
            return None
        
        try:
            # Run YOLO inference
            results = self.yolo_model(image, verbose=False)
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class information
                        class_id = int(box.cls[0])
                        class_name = self.yolo_model.names[class_id].lower()
                        confidence = float(box.conf[0])
                        
                        # Check if it's a forklift or related vehicle
                        # YOLO COCO classes that might be relevant:
                        # - truck (class 7)
                        # - bus (class 5) - sometimes misclassified
                        # You might need to adjust this based on your specific needs
                        forklift_related_classes = ['truck', 'bus', 'car']  # Add more as needed
                        
                        if (class_name in forklift_related_classes and confidence > 0.3) or \
                           ('fork' in class_name.lower()):
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            bbox_info = {
                                'x1': int(x1),
                                'y1': int(y1), 
                                'x2': int(x2),
                                'y2': int(y2),
                                'confidence': confidence,
                                'class': class_name,
                                'center_x': int((x1 + x2) / 2),
                                'center_y': int((y1 + y2) / 2),
                                'width': int(x2 - x1),
                                'height': int(y2 - y1)
                            }
                            
                            self.get_logger().info(f"Detected {class_name} with confidence {confidence:.2f}")
                            return bbox_info
            
            return None
            
        except Exception as e:
            self.get_logger().error(f"Error in YOLO detection: {e}")
            return None
    
    def draw_bounding_box(self, image, bbox_info):
        """Draw bounding box on image"""
        if bbox_info is None:
            return image
        
        # Make a copy of the image
        annotated_image = image.copy()
        
        # Extract coordinates
        x1, y1, x2, y2 = bbox_info['x1'], bbox_info['y1'], bbox_info['x2'], bbox_info['y2']
        confidence = bbox_info['confidence']
        class_name = bbox_info['class']
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw center point
        center_x, center_y = bbox_info['center_x'], bbox_info['center_y']
        cv2.circle(annotated_image, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Add text label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Draw background for text
        cv2.rectangle(annotated_image, 
                     (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), 
                     (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(annotated_image, label, 
                   (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return annotated_image
    
    def save_detection_image(self, image, bbox_info):
        """Save the image with bounding box"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forklift_detection_{timestamp}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            # Draw bounding box on image
            annotated_image = self.draw_bounding_box(image, bbox_info)
            
            # Save the image
            cv2.imwrite(filepath, annotated_image)
            
            self.get_logger().info(f"Detection image saved: {filepath}")
            return filepath
            
        except Exception as e:
            self.get_logger().error(f"Error saving detection image: {e}")
            return None
    
    def print_bbox_coordinates(self, bbox_info):
        """Print detailed bounding box information"""
        if bbox_info is None:
            return
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("FORKLIFT BOUNDING BOX COORDINATES:")
        self.get_logger().info(f"  Class: {bbox_info['class']}")
        self.get_logger().info(f"  Confidence: {bbox_info['confidence']:.3f}")
        self.get_logger().info(f"  Top-left corner (x1, y1): ({bbox_info['x1']}, {bbox_info['y1']})")
        self.get_logger().info(f"  Bottom-right corner (x2, y2): ({bbox_info['x2']}, {bbox_info['y2']})")
        self.get_logger().info(f"  Center point (x, y): ({bbox_info['center_x']}, {bbox_info['center_y']})")
        self.get_logger().info(f"  Width x Height: {bbox_info['width']} x {bbox_info['height']} pixels")
        self.get_logger().info("=" * 50)
    
    def check_for_forklift(self):
        """Periodic check for forklift in current image"""
        if self.current_image is None:
            return
        
        if self.search_phase and not self.forklift_detected:
            # Search phase - look for forklift
            self.get_logger().info("Searching for forklift...")
            
            # First use VLLM for initial detection
            if self.detect_forklift_with_vllm(self.current_image):
                self.get_logger().info("🎯 FORKLIFT DETECTED by VLLM! Stopping robot immediately...")
                
                # STOP ROBOT FIRST to prevent further rotation
                self.stop_robot()
                
                # Wait a moment for robot to fully stop
                time.sleep(0.5)
                
                # Capture a fresh image after stopping
                detection_image = self.current_image.copy() if self.current_image is not None else None
                
                if detection_image is not None:
                    self.get_logger().info("Running YOLO for bounding box detection...")
                    
                    # Use YOLO to get bounding box on the stopped image
                    bbox_info = self.detect_forklift_with_yolo(detection_image)
                    
                    if bbox_info is not None:
                        self.forklift_bbox = bbox_info
                        self.print_bbox_coordinates(bbox_info)
                        
                        # Save image with bounding box
                        saved_path = self.save_detection_image(detection_image, bbox_info)
                        if saved_path:
                            self.get_logger().info(f"📸 Detection image saved to: {saved_path}")
                        
                        self.forklift_detected = True
                        self.get_logger().info("✅ Forklift detection and bounding box analysis complete!")
                    else:
                        self.get_logger().warn("VLLM detected forklift but YOLO couldn't find bounding box.")
                        self.get_logger().warn("This might be a false positive. Resuming search...")
                        # Reset detection flag to continue searching
                        self.forklift_detected = False
                else:
                    self.get_logger().error("No image available after stopping robot")
                    self.forklift_detected = False
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
        for _ in range(10):  # Increased from 5 to 10 for more reliable stopping
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.05)  # Shorter intervals but more iterations
        
        self.get_logger().info("Robot stopped successfully!")
        self.is_moving = False  # Reset movement flag
        
        # Print final detection summary
        if self.forklift_bbox is not None:
            self.get_logger().info("\n🎉 FORKLIFT DETECTION COMPLETE!")
            self.print_bbox_coordinates(self.forklift_bbox)

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