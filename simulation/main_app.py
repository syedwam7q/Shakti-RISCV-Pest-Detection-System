"""
Main Pest Detection Application
Integrates all components for complete system demonstration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time
import argparse
import json
import numpy as np
from datetime import datetime
from typing import Optional

from algorithms.pest_detector import PestDetector
from simulation.camera_simulator import CameraSimulator, AlertSimulator


class PestDetectionSystem:
    """
    Main pest detection system that integrates all components.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pest detection system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.detector = PestDetector()
        self.camera = CameraSimulator(
            image_dir=self.config.get("image_dir"),
            resolution=tuple(self.config.get("camera_resolution", [640, 480])),
            fps=self.config.get("fps", 30)
        )
        self.alert_system = AlertSimulator()
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "pests_detected": 0,
            "alerts_sent": 0,
            "start_time": time.time()
        }
        
        self.running = False
        self.save_results = self.config.get("save_results", False)
        
        if self.save_results:
            self.results_file = self.config.get("results_file", "detection_results.json")
            self.detection_history = []
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from file or use defaults."""
        default_config = {
            "image_dir": None,
            "camera_resolution": [640, 480],
            "fps": 30,
            "detection_interval": 1.0,  # seconds between detections
            "save_results": False,
            "results_file": "detection_results.json",
            "alert_threshold": 0.6,
            "display_video": True
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                print(f"Loaded configuration from {config_path}")
            except Exception as e:
                print(f"Error loading config: {e}. Using defaults.")
        
        return default_config
    
    def process_frame(self, image: np.ndarray) -> dict:
        """
        Process a single frame for pest detection.
        
        Args:
            image: Input image from camera
            
        Returns:
            Detection results
        """
        # Perform pest detection
        results = self.detector.detect_pests(image)
        
        # Update statistics
        self.stats["frames_processed"] += 1
        if results.get("pest_detected", False):
            self.stats["pests_detected"] += 1
        
        # Add timestamp and frame info
        results["timestamp"] = datetime.now().isoformat()
        results["frame_number"] = self.stats["frames_processed"]
        
        # Save results if enabled
        if self.save_results:
            self.detection_history.append(results)
        
        return results
    
    def handle_alerts(self, results: dict):
        """
        Handle alert generation based on detection results.
        
        Args:
            results: Detection results from process_frame
        """
        if not results.get("pest_detected", False):
            return
        
        confidence = results.get("confidence", 0.0)
        if confidence < self.config["alert_threshold"]:
            return
        
        # Trigger appropriate alerts
        pest_type = results.get("class", "unknown")
        severity = results.get("severity", "unknown")
        
        # Always display results
        self.alert_system.display_alert(results)
        
        # Trigger buzzer for medium/high severity
        if severity in ["medium", "high"]:
            self.alert_system.trigger_buzzer_alert(pest_type, severity)
            self.stats["alerts_sent"] += 1
        
        # Send IoT alert for high severity or critical pests
        if severity == "high" or pest_type in ["leaf_spot", "powdery_mildew"]:
            self.alert_system.send_iot_alert(results)
    
    def display_frame(self, image: np.ndarray, results: dict):
        """
        Display frame with detection results overlay.
        
        Args:
            image: Input image
            results: Detection results
        """
        if not self.config["display_video"]:
            return
        
        # Create display image
        display_img = image.copy()
        
        # Add status text
        status_text = f"Status: {'PEST DETECTED' if results.get('pest_detected') else 'HEALTHY'}"
        color = (0, 0, 255) if results.get('pest_detected') else (0, 255, 0)
        cv2.putText(display_img, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Add detection info
        if results.get("pest_detected"):
            pest_type = results.get("class", "unknown")
            confidence = results.get("confidence", 0.0)
            info_text = f"{pest_type}: {confidence:.1%}"
            cv2.putText(display_img, info_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Add frame counter
        frame_text = f"Frame: {self.stats['frames_processed']}"
        cv2.putText(display_img, frame_text, (10, display_img.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show image
        cv2.imshow('Pest Detection System', display_img)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False
        elif key == ord('s'):
            # Save current frame
            filename = f"detection_frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, display_img)
            print(f"Frame saved as {filename}")
    
    def run_continuous(self):
        """Run continuous pest detection."""
        print("Starting Pest Detection System...")
        print("Press 'q' to quit, 's' to save current frame")
        print("-" * 50)
        
        self.running = True
        last_detection_time = 0
        
        try:
            for image, source_info in self.camera.stream_frames():
                if not self.running:
                    break
                
                current_time = time.time()
                
                # Perform detection at specified intervals
                if current_time - last_detection_time >= self.config["detection_interval"]:
                    results = self.process_frame(image)
                    self.handle_alerts(results)
                    last_detection_time = current_time
                    
                    # Print simple status
                    status = "🔴 PEST" if results.get("pest_detected") else "🟢 OK"
                    print(f"Frame {self.stats['frames_processed']:4d}: {status} - {source_info}")
                
                # Display frame
                self.display_frame(image, results if 'results' in locals() else {})
                
                # Brief pause to prevent overwhelming output
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping system...")
        finally:
            self.cleanup()
    
    def run_batch(self, image_paths: list):
        """
        Run batch processing on a list of images.
        
        Args:
            image_paths: List of image file paths
        """
        print(f"Processing {len(image_paths)} images in batch mode...")
        
        for i, image_path in enumerate(image_paths, 1):
            if not os.path.exists(image_path):
                print(f"  {i:3d}. Error: File not found - {image_path}")
                continue
            
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                print(f"  {i:3d}. Error: Cannot read - {image_path}")
                continue
            
            results = self.process_frame(image)
            
            # Print results
            status = "🔴 PEST DETECTED" if results.get("pest_detected") else "🟢 HEALTHY"
            pest_info = f" - {results.get('class', 'unknown')} ({results.get('confidence', 0):.1%})" if results.get("pest_detected") else ""
            
            print(f"  {i:3d}. {status}{pest_info} - {os.path.basename(image_path)}")
        
        self.print_statistics()
        if self.save_results:
            self.save_detection_history()
    
    def print_statistics(self):
        """Print system statistics."""
        runtime = time.time() - self.stats["start_time"]
        fps = self.stats["frames_processed"] / runtime if runtime > 0 else 0
        
        print("\n" + "="*50)
        print("SYSTEM STATISTICS")
        print("="*50)
        print(f"Runtime: {runtime:.1f} seconds")
        print(f"Frames processed: {self.stats['frames_processed']}")
        print(f"Average FPS: {fps:.1f}")
        print(f"Pests detected: {self.stats['pests_detected']}")
        print(f"Detection rate: {100 * self.stats['pests_detected'] / max(1, self.stats['frames_processed']):.1f}%")
        print(f"Alerts sent: {self.stats['alerts_sent']}")
        print("="*50)
    
    def save_detection_history(self):
        """Save detection history to file."""
        if not self.detection_history:
            return
        
        try:
            with open(self.results_file, 'w') as f:
                json.dump({
                    "statistics": self.stats,
                    "detections": self.detection_history
                }, f, indent=2)
            print(f"Detection history saved to {self.results_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        cv2.destroyAllWindows()
        self.print_statistics()
        self.alert_system.get_alert_summary()
        
        if self.save_results:
            self.save_detection_history()
        
        print("System shutdown complete.")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Pest Detection System')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--mode', choices=['continuous', 'batch'], default='continuous',
                       help='Operation mode')
    parser.add_argument('--images', nargs='+', help='Image files for batch processing')
    parser.add_argument('--generate-dataset', type=int, help='Generate test dataset with N images')
    
    args = parser.parse_args()
    
    # Generate dataset if requested
    if args.generate_dataset:
        camera = CameraSimulator()
        camera.save_test_dataset("datasets/synthetic", args.generate_dataset)
        return
    
    # Initialize system
    system = PestDetectionSystem(args.config)
    
    # Run in specified mode
    if args.mode == 'continuous':
        system.run_continuous()
    elif args.mode == 'batch' and args.images:
        system.run_batch(args.images)
    else:
        print("Invalid mode or missing arguments. Use --help for usage.")


if __name__ == "__main__":
    main()