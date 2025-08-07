#!/usr/bin/env python3
from ultralytics import YOLO
import cv2
import time
import argparse
import os
import signal
import sys
import RPi.GPIO as GPIO
from datetime import datetime
import threading

# GPIO setup for alarm/notification
BUZZER_PIN = 17  # GPIO pin for buzzer
LED_PIN = 18     # GPIO pin for LED
RELAY_PIN = 27   # GPIO pin for relay (optional for additional devices)

class GooseDetector:
    def __init__(self, model_path, confidence=0.25, device='cpu', source=0, 
                 cooldown=10, gpio_enabled=True, save_detections=True):
        """
        Initialize the goose detector
        
        Args:
            model_path: Path to the YOLOv8 model weights
            confidence: Detection confidence threshold
            device: Device to run inference on ('cpu' for Raspberry Pi)
            source: Camera source (usually 0 for built-in/USB webcam)
            cooldown: Time in seconds between triggered actions
            gpio_enabled: Whether to enable GPIO for physical notifications
            save_detections: Whether to save detection images
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device
        self.source = source
        self.cooldown = cooldown
        self.last_trigger_time = 0
        self.gpio_enabled = gpio_enabled
        self.save_detections = save_detections
        self.detection_count = 0
        self.running = True
        
        # Create directory for saved detections
        if self.save_detections:
            self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goose_detections")
            os.makedirs(self.save_dir, exist_ok=True)
        
        # Setup GPIO if enabled
        if self.gpio_enabled:
            self.setup_gpio()
            
        # Register signal handler for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_gpio(self):
        """Setup GPIO pins for notifications"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Setup buzzer pin
            GPIO.setup(BUZZER_PIN, GPIO.OUT)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            
            # Setup LED pin
            GPIO.setup(LED_PIN, GPIO.OUT)
            GPIO.output(LED_PIN, GPIO.LOW)
            
            # Setup relay pin
            GPIO.setup(RELAY_PIN, GPIO.OUT)
            GPIO.output(RELAY_PIN, GPIO.LOW)
            
            print("GPIO setup complete")
        except Exception as e:
            print(f"GPIO setup failed: {e}")
            self.gpio_enabled = False
    
    def trigger_alert(self):
        """Trigger alerts when a goose is detected"""
        if time.time() - self.last_trigger_time < self.cooldown:
            return
        
        print("🚨 GOOSE DETECTED! 🚨")
        self.last_trigger_time = time.time()
        
        if self.gpio_enabled:
            # Start alert in a separate thread to avoid blocking the main detection loop
            threading.Thread(target=self._alert_sequence).start()
    
    def _alert_sequence(self):
        """Run the physical alert sequence"""
        try:
            # Turn on LED
            GPIO.output(LED_PIN, GPIO.HIGH)
            
            # Pulse buzzer
            for _ in range(5):
                GPIO.output(BUZZER_PIN, GPIO.HIGH)
                time.sleep(0.1)
                GPIO.output(BUZZER_PIN, GPIO.LOW)
                time.sleep(0.1)
            
            # Activate relay briefly (if connected to water sprinkler, etc.)
            GPIO.output(RELAY_PIN, GPIO.HIGH)
            time.sleep(1)
            GPIO.output(RELAY_PIN, GPIO.LOW)
            
            # Keep LED on for a few seconds
            time.sleep(3)
            
            # Turn off LED
            GPIO.output(LED_PIN, GPIO.LOW)
        except Exception as e:
            print(f"Alert sequence failed: {e}")
    
    def save_detection_image(self, frame):
        """Save the detection image with timestamp"""
        if not self.save_detections:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"goose_{timestamp}_{self.detection_count}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"Detection saved: {filepath}")
        self.detection_count += 1
    
    def run(self):
        """Run the goose detection loop"""
        # Open webcam
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        # Get webcam properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Webcam resolution: {frame_width}x{frame_height}")
        
        # Display window (if running with display)
        try:
            cv2.namedWindow("Canadian Goose Detector", cv2.WINDOW_NORMAL)
            has_display = True
        except:
            has_display = False
            print("Running in headless mode (no display)")
        
        # FPS calculation variables
        prev_time = time.time()
        fps = 0
        
        print("Goose detection started. Press Ctrl+C to exit.")
        
        while self.running:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                time.sleep(1)  # Wait before retrying
                continue
            
            # Run inference
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                device=self.device,
                verbose=False
            )
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # Check if goose detected
            result = results[0]
            detections = result.boxes.data.cpu().numpy()
            
            goose_detected = False
            for detection in detections:
                goose_detected = True
                break
            
            # Get the rendered frame with detections
            annotated_frame = result.plot()
            
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
            
            # Trigger alert if goose detected
            if goose_detected:
                self.trigger_alert()
                self.save_detection_image(annotated_frame)
            
            # Show the frame if display is available
            if has_display:
                cv2.imshow("Canadian Goose Detector", annotated_frame)
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
        
        # Release resources
        cap.release()
        if has_display:
            cv2.destroyAllWindows()
        
        # Cleanup GPIO
        if self.gpio_enabled:
            GPIO.cleanup()
        
        print("Detection stopped")
    
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C and other termination signals"""
        print("\nShutting down...")
        self.running = False
        
        # Cleanup GPIO
        if self.gpio_enabled:
            GPIO.cleanup()
        
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Detect Canadian geese and trigger alerts')
    parser.add_argument('--model', type=str, default='goose_project/goose_detector4/weights/best.pt', 
                        help='Path to the trained model weights')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='Device to run inference on (cpu, 0, 1, etc.)')
    parser.add_argument('--source', type=int, default=0, help='Webcam source (usually 0 for built-in)')
    parser.add_argument('--cooldown', type=int, default=10, help='Cooldown between alerts in seconds')
    parser.add_argument('--no-gpio', action='store_true', help='Disable GPIO for testing on non-Pi systems')
    parser.add_argument('--no-save', action='store_true', help='Disable saving detection images')
    
    args = parser.parse_args()
    
    detector = GooseDetector(
        model_path=args.model,
        confidence=args.conf,
        device=args.device,
        source=args.source,
        cooldown=args.cooldown,
        gpio_enabled=not args.no_gpio,
        save_detections=not args.no_save
    )
    
    detector.run()

if __name__ == "__main__":
    main() 