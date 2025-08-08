#!/usr/bin/env python3
"""
Quick Start Script for YOLODeck
Automatically sets up the project and runs initial training
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 YOLODeck Quick Start")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("README.md").exists():
        print("❌ Please run this script from the YOLODeck project root directory")
        return
    
    # Step 1: Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return
    
    # Step 2: Create sample data
    if not run_command("python src/data_preparation/prepare_dataset.py --create-samples", "Creating sample data"):
        return
    
    # Step 3: Split dataset
    if not run_command("python src/data_preparation/prepare_dataset.py --split", "Splitting dataset"):
        return
    
    # Step 4: Validate dataset
    if not run_command("python src/training/train_model.py --validate-only", "Validating dataset"):
        return
    
    # Step 5: Show training info
    if not run_command("python src/training/train_model.py --info", "Showing training configuration"):
        return
    
    print("\n" + "=" * 50)
    print("🎉 Quick start completed successfully!")
    print("\n📋 Next steps:")
    print("1. Add your own card images to data/raw/")
    print("2. Annotate images using Roboflow or LabelImg")
    print("3. Run: python src/data_preparation/prepare_dataset.py --annotation-dir path/to/annotations --image-dir path/to/images")
    print("4. Start training: python src/training/train_model.py")
    print("5. Test the model: python src/inference/detect_cards.py --model models/best.pt --image path/to/test/image.jpg")
    print("6. Launch web interface: streamlit run src/web_app.py")
    print("\n📚 For detailed instructions, see README.md")

if __name__ == "__main__":
    main()
