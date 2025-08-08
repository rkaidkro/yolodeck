#!/usr/bin/env python3
"""
Setup script for YOLODeck
Installs dependencies and sets up the project environment
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def check_system():
    """Check system compatibility"""
    system = platform.system()
    machine = platform.machine()
    
    print(f"🖥️  System: {system} {machine}")
    
    if system == "Darwin" and "arm" in machine.lower():
        print("✅ Apple Silicon (M1/M2) detected - MPS acceleration available!")
    elif system == "Darwin":
        print("✅ macOS detected - MPS acceleration available!")
    elif system == "Linux":
        print("✅ Linux detected")
    elif system == "Windows":
        print("✅ Windows detected")
    else:
        print("⚠️  Unknown system - some features may not work optimally")

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

def verify_installation():
    """Verify that all required packages are installed"""
    print("🔍 Verifying installation...")
    
    required_packages = [
        "torch",
        "torchvision", 
        "ultralytics",
        "opencv-python",
        "Pillow",
        "numpy",
        "matplotlib",
        "seaborn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    return True

def check_mps_availability():
    """Check if MPS (Metal Performance Shaders) is available"""
    print("🍎 Checking MPS availability...")
    
    try:
        import torch
        if torch.backends.mps.is_available():
            print("✅ MPS (Metal Performance Shaders) is available!")
            print("   This will enable GPU acceleration on Apple Silicon")
        else:
            print("⚠️  MPS is not available")
            print("   Training will use CPU (slower)")
    except ImportError:
        print("❌ PyTorch not installed")

def create_sample_data():
    """Create sample data for testing"""
    print("📊 Creating sample data...")
    
    try:
        from src.data_preparation.prepare_dataset import DatasetPreparator
        preparator = DatasetPreparator()
        preparator.create_sample_data()
        print("✅ Sample data created successfully!")
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")

def setup_project():
    """Set up the complete project"""
    print("🚀 Setting up YOLODeck...")
    print("=" * 50)
    
    # Check system requirements
    check_python_version()
    check_system()
    
    # Install dependencies
    install_dependencies()
    
    # Verify installation
    if not verify_installation():
        sys.exit(1)
    
    # Check MPS availability
    check_mps_availability()
    
    # Create sample data
    create_sample_data()
    
    print("\n" + "=" * 50)
    print("🎉 YOLODeck setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Add your card images to data/raw/")
    print("2. Annotate images using Roboflow or LabelImg")
    print("3. Run: python src/data_preparation/prepare_dataset.py --create-samples")
    print("4. Run: python src/training/train_model.py --info")
    print("5. Start training: python src/training/train_model.py")
    print("\n📚 For more information, see README.md")

def main():
    """Main setup function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--check-only":
        print("🔍 Running system check only...")
        check_python_version()
        check_system()
        verify_installation()
        check_mps_availability()
    else:
        setup_project()

if __name__ == "__main__":
    main()
