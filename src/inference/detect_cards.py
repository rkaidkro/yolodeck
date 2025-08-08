#!/usr/bin/env python3
"""
Inference Script for YOLODeck
Real-time Magic card detection and classification
"""

import os
import sys
import cv2
import numpy as np
import argparse
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Any
from ultralytics import YOLO
import torch
import torch.backends.mps

class CardDetector:
    """Real-time card detection and classification"""
    
    def __init__(self, model_path: str, dataset_config: str = "configs/dataset.yaml"):
        """Initialize the card detector"""
        self.model_path = model_path
        self.dataset_config = dataset_config
        self.config = self._load_dataset_config()
        self.classes = self.config['names']
        
        # Load model
        print(f"🔄 Loading model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Check device
        self.device = self._get_device()
        print(f"📱 Using device: {self.device}")
        
        # Color mapping for different card types
        self.colors = {
            'magic_card': (255, 0, 0),      # Red
            'pokemon_card': (0, 255, 0),    # Green
            'yugioh_card': (0, 0, 255),     # Blue
            'business_card': (255, 255, 0), # Yellow
            'playing_card': (255, 0, 255)   # Magenta
        }
    
    def _load_dataset_config(self) -> Dict:
        """Load dataset configuration"""
        with open(self.dataset_config, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_device(self) -> str:
        """Get the best available device"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def detect_image(self, image_path: str, conf_threshold: float = 0.25, 
                    save_output: bool = True) -> List[Dict[str, Any]]:
        """Detect cards in a single image"""
        print(f"🔍 Detecting cards in: {image_path}")
        
        # Run inference
        results = self.model.predict(
            image_path,
            conf=conf_threshold,
            iou=0.45,
            max_det=10,
            device=self.device
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.classes.get(class_id, f"unknown_{class_id}")
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'color': self.colors.get(class_name, (128, 128, 128))
                    }
                    detections.append(detection)
        
        print(f"✅ Found {len(detections)} cards")
        
        # Save annotated image if requested
        if save_output:
            self._save_annotated_image(image_path, detections)
        
        return detections
    
    def _save_annotated_image(self, image_path: str, detections: List[Dict[str, Any]]):
        """Save image with detection annotations"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not read image: {image_path}")
            return
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            color = detection['color']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save annotated image
        output_path = Path(image_path).parent / f"detected_{Path(image_path).name}"
        cv2.imwrite(str(output_path), image)
        print(f"💾 Annotated image saved to: {output_path}")
    
    def detect_video(self, video_path: str, output_path: str = None, 
                    conf_threshold: float = 0.25, show_preview: bool = False):
        """Detect cards in video stream"""
        print(f"🎥 Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video
        if output_path is None:
            output_path = f"detected_{Path(video_path).name}"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:  # Print progress every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"📈 Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Run inference on frame
            results = self.model.predict(
                frame,
                conf=conf_threshold,
                iou=0.45,
                max_det=10,
                device=self.device
            )
            
            # Draw detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get confidence and class
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.classes.get(class_id, f"unknown_{class_id}")
                        color = self.colors.get(class_name, (128, 128, 128))
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Draw label
                        label = f"{class_name}: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        # Draw label background
                        cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10), 
                                     (int(x1) + label_size[0], int(y1)), color, -1)
                        
                        # Draw label text
                        cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write frame to output video
            out.write(frame)
            
            # Show preview if requested
            if show_preview:
                cv2.imshow('YOLODeck Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Cleanup
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        print(f"✅ Video processing completed! Output saved to: {output_path}")
    
    def detect_realtime(self, camera_id: int = 0, conf_threshold: float = 0.25):
        """Real-time card detection using webcam"""
        print(f"📹 Starting real-time detection with camera {camera_id}")
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Could not open camera {camera_id}")
            return
        
        print("🎮 Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'c' to clear detections")
        
        frame_count = 0
        start_time = cv2.getTickCount()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from camera")
                break
            
            frame_count += 1
            
            # Run inference
            results = self.model.predict(
                frame,
                conf=conf_threshold,
                iou=0.45,
                max_det=10,
                device=self.device
            )
            
            # Draw detections
            detections_count = 0
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get confidence and class
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.classes.get(class_id, f"unknown_{class_id}")
                        color = self.colors.get(class_name, (128, 128, 128))
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Draw label
                        label = f"{class_name}: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        # Draw label background
                        cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10), 
                                     (int(x1) + label_size[0], int(y1)), color, -1)
                        
                        # Draw label text
                        cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        detections_count += 1
            
            # Calculate and display FPS
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
            fps = frame_count / elapsed_time
            
            # Draw info overlay
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Detections: {detections_count}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Device: {self.device.upper()}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('YOLODeck Real-time Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = cv2.getTickCount()
                filename = f"captured_frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Frame saved as: {filename}")
            elif key == ord('c'):
                # Clear detections (just redraw without detections)
                pass
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Real-time detection stopped")

def main():
    parser = argparse.ArgumentParser(description="Detect Magic cards using YOLOv8")
    parser.add_argument("--model", required=True, help="Path to trained model (.pt file)")
    parser.add_argument("--dataset", default="configs/dataset.yaml", help="Dataset configuration file")
    parser.add_argument("--image", help="Path to image for detection")
    parser.add_argument("--video", help="Path to video for detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID for real-time detection")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", help="Output path for processed video")
    parser.add_argument("--realtime", action="store_true", help="Enable real-time detection")
    parser.add_argument("--preview", action="store_true", help="Show preview during video processing")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = CardDetector(args.model, args.dataset)
    
    if args.realtime:
        detector.detect_realtime(args.camera, args.conf)
    elif args.video:
        detector.detect_video(args.video, args.output, args.conf, args.preview)
    elif args.image:
        detections = detector.detect_image(args.image, args.conf)
        print(f"\n📋 Detection Results:")
        for i, detection in enumerate(detections):
            print(f"  {i+1}. {detection['class_name']} (confidence: {detection['confidence']:.3f})")
    else:
        print("❌ Please specify --image, --video, or --realtime")
        return

if __name__ == "__main__":
    main()
