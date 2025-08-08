#!/usr/bin/env python3
"""
Data Preparation Script for YOLODeck
Prepares and processes card images for YOLOv8 training
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

class DatasetPreparator:
    """Handles dataset preparation for YOLOv8 training"""
    
    def __init__(self, config_path: str = "configs/dataset.yaml"):
        """Initialize the dataset preparator"""
        self.config_path = config_path
        self.config = self._load_config()
        self.data_root = Path(self.config['path'])
        self.classes = self.config['names']
        self.num_classes = self.config['nc']
        
        # Create directory structure
        self._create_directories()
    
    def _load_config(self) -> Dict:
        """Load dataset configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_directories(self):
        """Create the necessary directory structure"""
        directories = [
            self.data_root / "images" / "train",
            self.data_root / "images" / "val", 
            self.data_root / "images" / "test",
            self.data_root / "labels" / "train",
            self.data_root / "labels" / "val",
            self.data_root / "labels" / "test"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
    
    def process_annotations(self, annotation_dir: str, image_dir: str):
        """Process annotations from Roboflow or LabelImg format to YOLO format"""
        annotation_path = Path(annotation_dir)
        image_path = Path(image_dir)
        
        print(f"Processing annotations from: {annotation_path}")
        print(f"Processing images from: {image_path}")
        
        # Process each annotation file
        for annotation_file in tqdm(list(annotation_path.glob("*.txt")), desc="Processing annotations"):
            image_file = image_path / f"{annotation_file.stem}.jpg"
            
            if not image_file.exists():
                print(f"Warning: Image {image_file} not found for annotation {annotation_file}")
                continue
            
            # Read image dimensions
            img = cv2.imread(str(image_file))
            if img is None:
                print(f"Warning: Could not read image {image_file}")
                continue
            
            height, width = img.shape[:2]
            
            # Process annotation file
            yolo_annotations = self._convert_to_yolo_format(annotation_file, width, height)
            
            # Save processed annotation
            output_path = self.data_root / "labels" / "train" / f"{annotation_file.stem}.txt"
            with open(output_path, 'w') as f:
                for annotation in yolo_annotations:
                    f.write(f"{annotation}\n")
            
            # Copy image to processed directory
            shutil.copy2(image_file, self.data_root / "images" / "train" / f"{annotation_file.stem}.jpg")
    
    def _convert_to_yolo_format(self, annotation_file: Path, img_width: int, img_height: int) -> List[str]:
        """Convert annotation format to YOLO format (class x_center y_center width height)"""
        yolo_annotations = []
        
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:  # At least class and 4 coordinates
                class_name = parts[0]
                x1, y1, x2, y2 = map(float, parts[1:5])
                
                # Convert to class index
                class_idx = self._get_class_index(class_name)
                if class_idx is None:
                    print(f"Warning: Unknown class {class_name}")
                    continue
                
                # Convert to YOLO format (normalized coordinates)
                x_center = (x1 + x2) / 2 / img_width
                y_center = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # Ensure coordinates are within [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                yolo_annotations.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        return yolo_annotations
    
    def _get_class_index(self, class_name: str) -> int:
        """Get class index from class name"""
        for idx, name in self.classes.items():
            if name == class_name:
                return idx
        return None
    
    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """Split dataset into train/validation/test sets"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Get all image files
        train_images_dir = self.data_root / "images" / "train"
        image_files = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png"))
        
        if not image_files:
            print("No images found in train directory")
            return
        
        # Shuffle files
        random.shuffle(image_files)
        
        # Calculate split indices
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split files
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        print(f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        
        # Move files to appropriate directories
        self._move_files(val_files, "val")
        self._move_files(test_files, "test")
    
    def _move_files(self, files: List[Path], split: str):
        """Move image and label files to specified split directory"""
        for img_file in files:
            # Move image
            src_img = img_file
            dst_img = self.data_root / "images" / split / img_file.name
            shutil.move(str(src_img), str(dst_img))
            
            # Move corresponding label
            label_name = img_file.stem + ".txt"
            src_label = self.data_root / "labels" / "train" / label_name
            dst_label = self.data_root / "labels" / split / label_name
            
            if src_label.exists():
                shutil.move(str(src_label), str(dst_label))
    
    def validate_dataset(self):
        """Validate the prepared dataset"""
        print("Validating dataset...")
        
        for split in ["train", "val", "test"]:
            images_dir = self.data_root / "images" / split
            labels_dir = self.data_root / "labels" / split
            
            if not images_dir.exists() or not labels_dir.exists():
                print(f"Warning: {split} directories not found")
                continue
            
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            label_files = list(labels_dir.glob("*.txt"))
            
            print(f"{split.capitalize()}: {len(image_files)} images, {len(label_files)} labels")
            
            # Check for missing labels
            for img_file in image_files:
                label_file = labels_dir / f"{img_file.stem}.txt"
                if not label_file.exists():
                    print(f"Warning: Missing label for {img_file}")
    
    def create_sample_data(self):
        """Create sample data structure for testing"""
        print("Creating sample data structure...")
        
        # Create sample images (colored rectangles representing cards)
        sample_data = [
            ("magic_card", (255, 0, 0)),    # Red for Magic cards
            ("pokemon_card", (0, 255, 0)),  # Green for Pokemon cards
            ("yugioh_card", (0, 0, 255)),   # Blue for Yu-Gi-Oh cards
            ("business_card", (255, 255, 0)), # Yellow for business cards
            ("playing_card", (255, 0, 255))   # Magenta for playing cards
        ]
        
        for class_name, color in sample_data:
            # Create sample image
            img = np.ones((640, 480, 3), dtype=np.uint8) * 255
            cv2.rectangle(img, (100, 100), (380, 540), color, -1)
            cv2.putText(img, class_name, (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Save image
            img_path = self.data_root / "images" / "train" / f"sample_{class_name}.jpg"
            cv2.imwrite(str(img_path), img)
            
            # Create corresponding label (YOLO format)
            label_path = self.data_root / "labels" / "train" / f"sample_{class_name}.txt"
            class_idx = self._get_class_index(class_name)
            if class_idx is not None:
                # Normalized coordinates for the rectangle
                x_center = (100 + 380) / 2 / 480  # (x1 + x2) / 2 / img_width
                y_center = (100 + 540) / 2 / 640  # (y1 + y2) / 2 / img_height
                width = (380 - 100) / 480         # (x2 - x1) / img_width
                height = (540 - 100) / 640        # (y2 - y1) / img_height
                
                with open(label_path, 'w') as f:
                    f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print("Sample data created successfully!")

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for YOLODeck training")
    parser.add_argument("--annotation-dir", help="Directory containing annotation files")
    parser.add_argument("--image-dir", help="Directory containing image files")
    parser.add_argument("--create-samples", action="store_true", help="Create sample data for testing")
    parser.add_argument("--split", action="store_true", help="Split dataset into train/val/test")
    parser.add_argument("--validate", action="store_true", help="Validate prepared dataset")
    
    args = parser.parse_args()
    
    # Initialize dataset preparator
    preparator = DatasetPreparator()
    
    if args.create_samples:
        preparator.create_sample_data()
    
    if args.annotation_dir and args.image_dir:
        preparator.process_annotations(args.annotation_dir, args.image_dir)
    
    if args.split:
        preparator.split_dataset()
    
    if args.validate:
        preparator.validate_dataset()
    
    print("Dataset preparation completed!")

if __name__ == "__main__":
    main()
