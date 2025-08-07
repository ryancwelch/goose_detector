from ultralytics import YOLO
import cv2
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description='Detect Canadian geese using webcam')
    parser.add_argument('--model', type=str, default='goose_project/goose_detector4/weights/best.pt', 
                        help='Path to the trained model weights')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='Device to run inference on (cpu, 0, 1, etc.)')
    parser.add_argument('--source', type=int, default=0, help='Webcam source (usually 0 for built-in)')
    args = parser.parse_args()
    
    # Load the model
    model = YOLO(args.model)
    print(f"Model loaded from {args.model}")
    print(f"Running on {args.device}")
    
    # Open webcam
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get webcam properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam resolution: {frame_width}x{frame_height}")
    
    # Display window
    cv2.namedWindow("Canadian Goose Detector", cv2.WINDOW_NORMAL)
    
    # FPS calculation variables
    prev_time = time.time()
    fps = 0
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Run inference
        results = model.predict(
            source=frame,
            conf=args.conf,
            device=args.device,
            verbose=False
        )
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Get the rendered frame with detections
        annotated_frame = results[0].plot()
        
        # Add FPS text
        cv2.putText(
            annotated_frame, 
            f"FPS: {fps:.1f}", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Show the frame
        cv2.imshow("Canadian Goose Detector", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped")

if __name__ == "__main__":
    main() 