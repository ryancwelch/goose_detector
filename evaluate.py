from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    # Set working directory to project root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Path to the best trained model
    model_path = 'goose_project/goose_detector4/weights/best.pt'
    
    # Load the trained model
    model = YOLO(model_path)
    
    print(f"Loaded model from {model_path}")
    
    # Run validation on test dataset
    results = model.val(
        data='/home/rwelch/goose/data/data.yaml',
        split='test',        # Use test split for final evaluation
        imgsz=640,           # Same image size as training
        batch=16,            # Batch size
        device='1',          # Use GPU 1
        verbose=True,        # Show detailed output
        plots=True           # Generate evaluation plots
    )
    
    # Extract and print key metrics
    metrics = results.box
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"mAP50      : {metrics.map50:.4f}")
    print(f"mAP50-95   : {metrics.map:.4f}")
    
    # Handle precision and recall which may be arrays
    if hasattr(metrics, 'p') and isinstance(metrics.p, np.ndarray):
        print(f"Precision  : {metrics.p.mean():.4f}")
    elif hasattr(metrics, 'p'):
        print(f"Precision  : {metrics.p:.4f}")
    else:
        print("Precision  : Not available")
        
    if hasattr(metrics, 'r') and isinstance(metrics.r, np.ndarray):
        print(f"Recall     : {metrics.r.mean():.4f}")
    elif hasattr(metrics, 'r'):
        print(f"Recall     : {metrics.r:.4f}")
    else:
        print("Recall     : Not available")
    
    # Calculate F1 score if precision and recall are available
    if hasattr(metrics, 'p') and hasattr(metrics, 'r'):
        p = metrics.p.mean() if isinstance(metrics.p, np.ndarray) else metrics.p
        r = metrics.r.mean() if isinstance(metrics.r, np.ndarray) else metrics.r
        f1 = 2 * (p * r) / (p + r + 1e-10)
        print(f"F1-Score   : {f1:.4f}")
    else:
        print("F1-Score   : Not available")
    
    # Show confusion matrix and other plots that were generated
    results_dir = Path(model.predictor.args.save_dir)
    print(f"\nDetailed results and plots saved to: {results_dir}")
    
    # Return paths to the plots for easy access
    confusion_matrix = results_dir / 'confusion_matrix.png'
    pr_curve = results_dir / 'PR_curve.png'
    f1_curve = results_dir / 'F1_curve.png'
    
    if confusion_matrix.exists():
        print(f"Confusion Matrix: {confusion_matrix}")
    if pr_curve.exists():
        print(f"PR Curve: {pr_curve}")
    if f1_curve.exists():
        print(f"F1 Curve: {f1_curve}")

if __name__ == "__main__":
    main() 