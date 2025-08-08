# YOLODeck - Magic Card Recognition with YOLOv8

A YOLOv8-based system for recognizing Magic: The Gathering cards and other card types, optimized for Apple Silicon (M1/M2) with Metal Performance Shaders (MPS) acceleration.

## 🎯 Problem Statement

How to fine-tune YOLOv8 on an M1 MacBook Air to recognize Magic cards and other card types, including a business analyst artefact for the process, with a detailed technical stack, workflow, and practical steps.

## 🚀 Solution Overview

1. **YOLOv8 (Ultralytics)** - Supports Apple Silicon (M1/M2) via Metal Performance Shaders (MPS) backend
2. **M1 MacBook Air Optimization** - Leverages MPS for GPU acceleration
3. **Comprehensive Dataset** - High-resolution, diverse card images with proper annotation
4. **Iterative Training** - Monitor metrics and improve accuracy through data augmentation
5. **Production Deployment** - Export trained model for app/web integration

## 🛠 Technical Stack

- **Framework**: YOLOv8 (Ultralytics)
- **Hardware**: MacBook Air (M1)
- **Backend**: PyTorch with Metal Performance Shaders (MPS)
- **Annotation Tools**: Roboflow, LabelImg
- **Data Processing**: Python, OpenCV, PIL
- **Evaluation**: Confusion Matrix, mAP, Per-class Accuracy

## 📁 Project Structure

```
yolodeck/
├── data/                   # Dataset storage
│   ├── raw/               # Raw card images
│   ├── annotated/         # Annotated images
│   └── processed/         # Processed dataset
├── models/                # Trained models
├── src/                   # Source code
│   ├── data_preparation/  # Data processing scripts
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation scripts
│   └── inference/         # Inference scripts
├── configs/               # Configuration files
├── notebooks/             # Jupyter notebooks
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎯 Key Features

- **M1 Optimization**: Full MPS backend support for GPU acceleration
- **Multi-Card Support**: Magic, Pokemon, Yu-Gi-Oh, and business cards
- **Data Augmentation**: Robust training with varied lighting, angles, backgrounds
- **Quality Assurance**: Automated annotation QA and validation
- **Production Ready**: Export models for deployment

## 📊 Performance Targets

- **Accuracy Goal**: >98% (100% is not realistic)
- **Dataset Size**: Minimum 200 images per card type
- **Split Ratio**: 80/10/10 (train/validation/test)
- **Model Size**: Start with YOLOv8n (nano) for speed

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/rkaidkro/yolodeck.git
cd yolodeck

# Install dependencies
pip install -r requirements.txt

# Verify MPS support
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## 🚀 Quick Start

1. **Prepare Dataset**:
   ```bash
   python src/data_preparation/prepare_dataset.py
   ```

2. **Train Model**:
   ```bash
   yolo detect train data=configs/dataset.yaml model=yolov8n.pt device=mps
   ```

3. **Evaluate Results**:
   ```bash
   python src/evaluation/evaluate_model.py
   ```

4. **Run Inference**:
   ```bash
   python src/inference/detect_cards.py --image path/to/card.jpg
   ```

## 📈 Training Workflow

1. **Data Collection**: Gather diverse card images (front, back, angles, lighting)
2. **Annotation**: Use Roboflow/LabelImg for bounding boxes and labels
3. **Preprocessing**: Split dataset, apply augmentation
4. **Training**: Use MPS backend with YOLOv8
5. **Evaluation**: Monitor metrics, check confusion matrix
6. **Iteration**: Improve with more data and hyperparameter tuning

## ⚠️ Key Risks & Mitigation

- **M1 GPU Speed**: Mitigate with batch size tuning
- **Data Diversity**: Ensure varied backgrounds, lighting, angles
- **Annotation Errors**: Implement automated QA checks
- **Overfitting**: Use validation set and early stopping
- **Unrealistic Expectations**: Aim for >98%, not 100% accuracy

## 🎯 Recommendations

- Start with YOLOv8n (nano) for faster iteration
- Use transfer learning from pre-trained models
- Automate annotation quality assurance
- Monitor per-class accuracy closely
- Iterate with more data for misclassified cards

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions, please open an issue on GitHub.
