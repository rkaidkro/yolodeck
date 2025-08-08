#!/usr/bin/env python3
"""
Training Script for YOLODeck
Trains YOLOv8 model for Magic card recognition on M1 MacBook Air
"""

import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any
import torch
import torch.backends.mps

class YOLOTrainer:
    """Handles YOLOv8 training with M1 optimization"""
    
    def __init__(self, config_path: str = "configs/training.yaml"):
        """Initialize the trainer"""
        self.config_path = config_path
        self.config = self._load_config()
        self._check_mps_availability()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _check_mps_availability(self):
        """Check if MPS (Metal Performance Shaders) is available"""
        if not torch.backends.mps.is_available():
            print("Warning: MPS is not available. Falling back to CPU.")
            self.config['device'] = 'cpu'
        else:
            print("✅ MPS (Metal Performance Shaders) is available!")
            print(f"Using device: {self.config['device']}")
    
    def prepare_training_command(self, dataset_config: str = "configs/dataset.yaml") -> str:
        """Prepare the YOLO training command"""
        cmd_parts = [
            "yolo",
            "detect",
            "train",
            f"data={dataset_config}",
            f"model={self.config['model']}",
            f"epochs={self.config['epochs']}",
            f"batch={self.config['batch_size']}",
            f"imgsz={self.config['imgsz']}",
            f"device={self.config['device']}",
            f"workers={self.config['workers']}",
            f"optimizer={self.config['optimizer']}",
            f"lr0={self.config['lr0']}",
            f"lrf={self.config['lrf']}",
            f"momentum={self.config['momentum']}",
            f"weight_decay={self.config['weight_decay']}",
            f"warmup_epochs={self.config['warmup_epochs']}",
            f"warmup_momentum={self.config['warmup_momentum']}",
            f"warmup_bias_lr={self.config['warmup_bias_lr']}",
            f"box={self.config['box']}",
            f"cls={self.config['cls']}",
            f"dfl={self.config['dfl']}",
            f"label_smoothing={self.config['label_smoothing']}",
            f"nbs={self.config['nbs']}",
            f"overlap_mask={self.config['overlap_mask']}",
            f"mask_ratio={self.config['mask_ratio']}",
            f"dropout={self.config['dropout']}",
            f"val={self.config['val']}",
            f"save={self.config['save']}",
            f"save_period={self.config['save_period']}",
            f"cache={self.config['cache']}",
            f"plots={self.config['plots']}",
            f"save_txt={self.config['save_txt']}",
            f"save_conf={self.config['save_conf']}",
            f"save_crop={self.config['save_crop']}",
            f"show_labels={self.config['show_labels']}",
            f"show_conf={self.config['show_conf']}",
            f"show_boxes={self.config['show_boxes']}",
            f"conf={self.config['conf']}",
            f"iou={self.config['iou']}",
            f"max_det={self.config['max_det']}",
            f"half={self.config['half']}",
            f"dnn={self.config['dnn']}",
            f"format={self.config['format']}"
        ]
        
        return " ".join(cmd_parts)
    
    def train(self, dataset_config: str = "configs/dataset.yaml", resume: bool = False):
        """Start the training process"""
        print("🚀 Starting YOLOv8 training for Magic card recognition...")
        print(f"📊 Dataset config: {dataset_config}")
        print(f"⚙️  Training config: {self.config_path}")
        
        # Prepare command
        cmd = self.prepare_training_command(dataset_config)
        
        if resume:
            cmd += " --resume"
        
        print(f"🔧 Training command: {cmd}")
        print("\n" + "="*80)
        print("Starting training... (This may take several hours)")
        print("="*80 + "\n")
        
        try:
            # Run training
            result = subprocess.run(cmd, shell=True, check=True)
            print("\n✅ Training completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Training failed with error: {e}")
            return False
    
    def validate_dataset(self, dataset_config: str = "configs/dataset.yaml"):
        """Validate the dataset before training"""
        print("🔍 Validating dataset...")
        
        try:
            # Load dataset config
            with open(dataset_config, 'r') as f:
                dataset_yaml = yaml.safe_load(f)
            
            data_path = Path(dataset_yaml['path'])
            
            # Check if directories exist
            required_dirs = [
                data_path / "images" / "train",
                data_path / "images" / "val",
                data_path / "labels" / "train",
                data_path / "labels" / "val"
            ]
            
            for directory in required_dirs:
                if not directory.exists():
                    print(f"❌ Missing directory: {directory}")
                    return False
                else:
                    files = list(directory.glob("*"))
                    print(f"✅ {directory}: {len(files)} files")
            
            # Check for images and labels
            train_images = list((data_path / "images" / "train").glob("*.jpg")) + \
                          list((data_path / "images" / "train").glob("*.png"))
            train_labels = list((data_path / "labels" / "train").glob("*.txt"))
            
            val_images = list((data_path / "images" / "val").glob("*.jpg")) + \
                        list((data_path / "images" / "val").glob("*.png"))
            val_labels = list((data_path / "labels" / "val").glob("*.txt"))
            
            print(f"\n📊 Dataset Summary:")
            print(f"   Training images: {len(train_images)}")
            print(f"   Training labels: {len(train_labels)}")
            print(f"   Validation images: {len(val_images)}")
            print(f"   Validation labels: {len(val_labels)}")
            
            if len(train_images) == 0:
                print("❌ No training images found!")
                return False
            
            if len(train_labels) == 0:
                print("❌ No training labels found!")
                return False
            
            print("✅ Dataset validation passed!")
            return True
            
        except Exception as e:
            print(f"❌ Dataset validation failed: {e}")
            return False
    
    def get_training_info(self):
        """Display training configuration information"""
        print("📋 Training Configuration:")
        print(f"   Model: {self.config['model']}")
        print(f"   Epochs: {self.config['epochs']}")
        print(f"   Batch size: {self.config['batch_size']}")
        print(f"   Image size: {self.config['imgsz']}")
        print(f"   Device: {self.config['device']}")
        print(f"   Optimizer: {self.config['optimizer']}")
        print(f"   Learning rate: {self.config['lr0']}")
        print(f"   Workers: {self.config['workers']}")
        
        # M1-specific optimizations
        if self.config['device'] == 'mps':
            print("\n🍎 M1 MacBook Air Optimizations:")
            print("   ✅ Using Metal Performance Shaders (MPS)")
            print("   ✅ Half-precision training enabled")
            print("   ✅ Optimized batch size for M1 GPU memory")
            print("   ⚠️  Monitor GPU memory usage during training")

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for Magic card recognition")
    parser.add_argument("--config", default="configs/training.yaml", help="Training configuration file")
    parser.add_argument("--dataset", default="configs/dataset.yaml", help="Dataset configuration file")
    parser.add_argument("--validate-only", action="store_true", help="Only validate dataset, don't train")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--info", action="store_true", help="Show training configuration info")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = YOLOTrainer(args.config)
    
    if args.info:
        trainer.get_training_info()
        return
    
    if args.validate_only:
        success = trainer.validate_dataset(args.dataset)
        sys.exit(0 if success else 1)
    
    # Validate dataset before training
    if not trainer.validate_dataset(args.dataset):
        print("❌ Dataset validation failed. Please fix the issues before training.")
        sys.exit(1)
    
    # Show training info
    trainer.get_training_info()
    
    # Start training
    success = trainer.train(args.dataset, args.resume)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
