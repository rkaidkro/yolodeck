#!/usr/bin/env python3
"""
Web Application for YOLODeck
Simple Streamlit interface for testing card detection
"""

import streamlit as st
import os
import sys
import yaml
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
import torch.backends.mps

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_config():
    """Load dataset configuration"""
    config_path = project_root / "configs" / "dataset.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(model_path=None):
    """Load the trained model"""
    if model_path is None:
        models_dir = project_root / "models"
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pt"))
            if model_files:
                model_path = model_files[0]
            else:
                st.error("No trained models found. Please train a model first.")
                return None
        else:
            st.error("Models directory not found.")
            return None
    
    try:
        model = YOLO(str(model_path))
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def detect_cards(image, model, conf_threshold=0.25):
    """Detect cards in image"""
    results = model.predict(
        image,
        conf=conf_threshold,
        iou=0.45,
        max_det=10
    )
    
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class_id': class_id
                }
                detections.append(detection)
    
    return detections

def draw_detections(image, detections, config):
    """Draw detection boxes on image"""
    image_with_boxes = image.copy()
    
    colors = {
        'magic_card': (255, 0, 0),      # Red
        'pokemon_card': (0, 255, 0),    # Green
        'yugioh_card': (0, 0, 255),     # Blue
        'business_card': (255, 255, 0), # Yellow
        'playing_card': (255, 0, 255)   # Magenta
    }
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_id = detection['class_id']
        class_name = config['names'][class_id]
        color = colors.get(class_name, (128, 128, 128))
        
        # Draw bounding box
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Draw label background
        cv2.rectangle(image_with_boxes, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(image_with_boxes, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image_with_boxes

def main():
    st.set_page_config(
        page_title="YOLODeck - Magic Card Recognition",
        page_icon="🃏",
        layout="wide"
    )
    
    st.title("🃏 YOLODeck - Magic Card Recognition")
    st.markdown("Detect and classify Magic cards, Pokemon cards, Yu-Gi-Oh cards, and more!")
    
    # Load configuration
    try:
        config = load_config()
        st.sidebar.success("✅ Configuration loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load configuration: {e}")
        return
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Please train a model first using the training scripts.")
        return
    
    st.sidebar.success("✅ Model loaded successfully")
    
    # Sidebar controls
    st.sidebar.header("Detection Settings")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence for detections"
    )
    
    # Device info
    if torch.backends.mps.is_available():
        st.sidebar.info("🍎 MPS (Apple Silicon) acceleration available")
    elif torch.cuda.is_available():
        st.sidebar.info("🚀 CUDA acceleration available")
    else:
        st.sidebar.info("💻 Using CPU")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📷 Image Upload", "📹 Webcam", "📊 Model Info"])
    
    with tab1:
        st.header("Upload Image for Detection")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image containing cards to detect"
        )
        
        if uploaded_file is not None:
            # Convert uploaded file to image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image_rgb, use_column_width=True)
            
            # Run detection
            with st.spinner("Detecting cards..."):
                detections = detect_cards(image, model, conf_threshold)
            
            # Display results
            with col2:
                st.subheader("Detection Results")
                
                if detections:
                    # Draw detections on image
                    image_with_boxes = draw_detections(image, detections, config)
                    image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
                    st.image(image_with_boxes_rgb, use_column_width=True)
                    
                    # Show detection details
                    st.write(f"**Found {len(detections)} cards:**")
                    for i, detection in enumerate(detections):
                        class_name = config['names'][detection['class_id']]
                        confidence = detection['confidence']
                        st.write(f"{i+1}. **{class_name}** (confidence: {confidence:.3f})")
                else:
                    st.warning("No cards detected. Try adjusting the confidence threshold.")
    
    with tab2:
        st.header("Real-time Webcam Detection")
        st.info("⚠️ Webcam detection requires camera access and may not work in all environments.")
        
        if st.button("Start Webcam Detection"):
            st.warning("Webcam detection is not implemented in this web interface.")
            st.info("For real-time detection, use the command line tool:")
            st.code("python src/inference/detect_cards.py --model models/your_model.pt --realtime")
    
    with tab3:
        st.header("Model Information")
        
        # Model details
        st.subheader("Model Configuration")
        st.write(f"**Classes:** {len(config['names'])}")
        for class_id, class_name in config['names'].items():
            st.write(f"  - {class_id}: {class_name}")
        
        # System information
        st.subheader("System Information")
        st.write(f"**PyTorch Version:** {torch.__version__}")
        st.write(f"**MPS Available:** {torch.backends.mps.is_available()}")
        st.write(f"**CUDA Available:** {torch.cuda.is_available()}")
        
        # Performance tips
        st.subheader("Performance Tips")
        st.markdown("""
        - **Lower confidence threshold** = More detections (may include false positives)
        - **Higher confidence threshold** = Fewer detections (more precise)
        - **M1/M2 Macs** automatically use MPS acceleration for faster inference
        - **Large images** may take longer to process
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**YOLODeck** - Built with YOLOv8 and optimized for Apple Silicon. "
        "For more information, see the [README](https://github.com/rkaidkro/yolodeck)."
    )

if __name__ == "__main__":
    main()
