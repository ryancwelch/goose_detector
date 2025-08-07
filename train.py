from ultralytics import YOLO
import os

def main():
    # Set working directory to project root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Load a YOLOv8n model (nano size)
    model = YOLO('yolov8n.pt')  # load a pretrained model
    
    # Train the model using our dataset
    results = model.train(
        data='/home/rwelch/goose/data/data.yaml',  # Path to data YAML file
        epochs=100,              # Number of epochs
        imgsz=640,               # Image size
        batch=24,                # Increased batch size since GPU has enough memory
        name='goose_detector',   # Name for the run
        patience=20,             # Early stopping patience
        save=True,               # Save checkpoints
        device='1',              # GPU device (use 'cpu' if no GPU)
        project='goose_project', # Project name
        verbose=True,            # Verbose output
        save_period=5            # Save checkpoint every 5 epochs
    )
    
    # Validate the model on the test dataset
    results = model.val()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 