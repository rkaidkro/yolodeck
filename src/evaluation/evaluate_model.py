#!/usr/bin/env python3
"""
Evaluation Script for YOLODeck
Evaluates trained YOLOv8 model performance and generates detailed metrics
"""

import os
import sys
import yaml
import argparse
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from ultralytics import YOLO

class ModelEvaluator:
    """Handles model evaluation and performance analysis"""
    
    def __init__(self, model_path: str, dataset_config: str = "configs/dataset.yaml"):
        """Initialize the evaluator"""
        self.model_path = model_path
        self.dataset_config = dataset_config
        self.config = self._load_dataset_config()
        self.model = YOLO(model_path)
        self.classes = self.config['names']
        
    def _load_dataset_config(self) -> Dict:
        """Load dataset configuration"""
        with open(self.dataset_config, 'r') as f:
            return yaml.safe_load(f)
    
    def run_validation(self, split: str = "val") -> Dict[str, Any]:
        """Run model validation on specified dataset split"""
        print(f"🔍 Running validation on {split} split...")
        
        data_path = Path(self.config['path'])
        val_images_dir = data_path / "images" / split
        
        if not val_images_dir.exists():
            print(f"❌ Validation images directory not found: {val_images_dir}")
            return {}
        
        # Run validation using YOLO
        results = self.model.val(
            data=self.dataset_config,
            split=split,
            conf=0.001,
            iou=0.6,
            max_det=300,
            save_txt=True,
            save_conf=True,
            save_json=True
        )
        
        print("✅ Validation completed!")
        return results
    
    def generate_confusion_matrix(self, results_dir: str = "runs/detect/val"):
        """Generate and visualize confusion matrix"""
        print("📊 Generating confusion matrix...")
        
        results_path = Path(results_dir)
        confusion_matrix_file = results_path / "confusion_matrix.png"
        
        if confusion_matrix_file.exists():
            print(f"✅ Confusion matrix saved to: {confusion_matrix_file}")
            return str(confusion_matrix_file)
        else:
            print("⚠️  Confusion matrix not found. Run validation first.")
            return None
    
    def analyze_per_class_performance(self, results_dir: str = "runs/detect/val"):
        """Analyze performance for each card class"""
        print("🎯 Analyzing per-class performance...")
        
        results_path = Path(results_dir)
        results_file = results_path / "results.json"
        
        if not results_file.exists():
            print("❌ Results file not found. Run validation first.")
            return {}
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Extract per-class metrics
        class_metrics = {}
        for class_name in self.classes.values():
            class_metrics[class_name] = {
                'precision': results.get(f'{class_name}_precision', 0),
                'recall': results.get(f'{class_name}_recall', 0),
                'f1_score': results.get(f'{class_name}_f1', 0),
                'ap': results.get(f'{class_name}_ap', 0)
            }
        
        return class_metrics
    
    def create_performance_report(self, output_dir: str = "evaluation_results"):
        """Create comprehensive performance report"""
        print("📋 Creating performance report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Run validation
        results = self.run_validation()
        
        # Generate confusion matrix
        confusion_matrix_path = self.generate_confusion_matrix()
        
        # Analyze per-class performance
        class_metrics = self.analyze_per_class_performance()
        
        # Create performance visualization
        self._create_performance_plots(class_metrics, output_path)
        
        # Generate text report
        self._generate_text_report(results, class_metrics, output_path)
        
        print(f"✅ Performance report saved to: {output_path}")
        return output_path
    
    def _create_performance_plots(self, class_metrics: Dict, output_path: Path):
        """Create performance visualization plots"""
        if not class_metrics:
            return
        
        # Prepare data for plotting
        classes = list(class_metrics.keys())
        metrics = ['precision', 'recall', 'f1_score', 'ap']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('YOLODeck Model Performance by Card Class', fontsize=16)
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            values = [class_metrics[cls][metric] for cls in classes]
            
            bars = ax.bar(classes, values, color=['red', 'green', 'blue', 'yellow', 'magenta'])
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, results: Dict, class_metrics: Dict, output_path: Path):
        """Generate text-based performance report"""
        report_file = output_path / 'performance_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("YOLODeck Model Performance Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall metrics
            f.write("Overall Performance Metrics:\n")
            f.write("-" * 30 + "\n")
            if hasattr(results, 'box') and results.box:
                f.write(f"mAP@0.5: {results.box.map50:.4f}\n")
                f.write(f"mAP@0.5:0.95: {results.box.map:.4f}\n")
            f.write(f"Precision: {getattr(results, 'precision', 'N/A')}\n")
            f.write(f"Recall: {getattr(results, 'recall', 'N/A')}\n\n")
            
            # Per-class performance
            f.write("Per-Class Performance:\n")
            f.write("-" * 25 + "\n")
            for class_name, metrics in class_metrics.items():
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
                f.write(f"  Average Precision: {metrics['ap']:.4f}\n")
            
            # Recommendations
            f.write("\n\nRecommendations:\n")
            f.write("-" * 15 + "\n")
            
            # Find worst performing classes
            worst_classes = sorted(class_metrics.items(), 
                                 key=lambda x: x[1]['f1_score'])[:2]
            
            f.write("Classes that need improvement:\n")
            for class_name, metrics in worst_classes:
                f.write(f"  - {class_name}: F1-Score = {metrics['f1_score']:.4f}\n")
                f.write(f"    Consider adding more training data for this class.\n")
            
            f.write("\nGeneral recommendations:\n")
            f.write("  - Aim for >98% accuracy (100% is not realistic)\n")
            f.write("  - Monitor per-class accuracy closely\n")
            f.write("  - Iterate with more data for misclassified cards\n")
            f.write("  - Use data augmentation for better generalization\n")
    
    def test_inference_speed(self, test_images_dir: str = "data/processed/images/test"):
        """Test model inference speed"""
        print("⚡ Testing inference speed...")
        
        test_path = Path(test_images_dir)
        if not test_path.exists():
            print(f"❌ Test images directory not found: {test_path}")
            return {}
        
        image_files = list(test_path.glob("*.jpg")) + list(test_path.glob("*.png"))
        
        if not image_files:
            print("❌ No test images found")
            return {}
        
        # Limit to first 10 images for speed test
        test_images = image_files[:10]
        
        import time
        times = []
        
        for img_file in test_images:
            start_time = time.time()
            self.model.predict(str(img_file), conf=0.25, iou=0.45)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        print(f"✅ Average inference time: {avg_time:.4f} seconds")
        print(f"✅ FPS: {fps:.2f}")
        
        return {
            'avg_inference_time': avg_time,
            'fps': fps,
            'test_images': len(test_images)
        }

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model for Magic card recognition")
    parser.add_argument("--model", required=True, help="Path to trained model (.pt file)")
    parser.add_argument("--dataset", default="configs/dataset.yaml", help="Dataset configuration file")
    parser.add_argument("--output", default="evaluation_results", help="Output directory for results")
    parser.add_argument("--speed-test", action="store_true", help="Run inference speed test")
    parser.add_argument("--confusion-matrix", action="store_true", help="Generate confusion matrix only")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model, args.dataset)
    
    if args.confusion_matrix:
        confusion_matrix_path = evaluator.generate_confusion_matrix()
        if confusion_matrix_path:
            print(f"Confusion matrix: {confusion_matrix_path}")
        return
    
    if args.speed_test:
        speed_results = evaluator.test_inference_speed()
        if speed_results:
            print(f"Speed test results: {speed_results}")
        return
    
    # Run full evaluation
    output_path = evaluator.create_performance_report(args.output)
    print(f"\n🎉 Evaluation completed! Results saved to: {output_path}")

if __name__ == "__main__":
    main()
