"""
Camera Simulator for Testing Pest Detection System
Simulates camera input when hardware is not available
"""

import cv2
import numpy as np
import os
import time
from typing import Optional, Iterator, Tuple
import random


class CameraSimulator:
    """
    Simulates camera input for testing pest detection algorithms.
    Can load images from files or generate synthetic test images.
    """
    
    def __init__(self, 
                 image_dir: Optional[str] = None,
                 resolution: Tuple[int, int] = (640, 480),
                 fps: int = 30):
        """
        Initialize camera simulator.
        
        Args:
            image_dir: Directory containing test images
            resolution: Simulated camera resolution
            fps: Frames per second for video simulation
        """
        self.image_dir = image_dir
        self.resolution = resolution
        self.fps = fps
        self.frame_delay = 1.0 / fps
        
        self.test_images = []
        self.current_frame = 0
        
        if image_dir and os.path.exists(image_dir):
            self._load_test_images()
        else:
            print("No image directory provided. Will generate synthetic images.")
    
    def _load_test_images(self):
        """Load test images from directory."""
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        for filename in os.listdir(self.image_dir):
            if filename.lower().endswith(supported_formats):
                image_path = os.path.join(self.image_dir, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    # Resize to target resolution
                    image = cv2.resize(image, self.resolution)
                    self.test_images.append((image, filename))
        
        print(f"Loaded {len(self.test_images)} test images")
    
    def get_frame(self) -> Tuple[np.ndarray, str]:
        """
        Get next frame from simulator.
        
        Returns:
            Tuple of (image, source_info)
        """
        if self.test_images:
            # Cycle through loaded images
            image, filename = self.test_images[self.current_frame % len(self.test_images)]
            self.current_frame += 1
            return image.copy(), f"File: {filename}"
        else:
            # Generate synthetic image
            return self._generate_synthetic_frame()
    
    def _generate_synthetic_frame(self) -> Tuple[np.ndarray, str]:
        """Generate synthetic test image."""
        image = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        # Create base leaf-like background
        image[:, :, 1] = 80 + random.randint(0, 50)  # Green base
        image[:, :, 0] = random.randint(20, 60)      # Some blue
        image[:, :, 2] = random.randint(30, 80)      # Some red
        
        # Add some texture
        noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Randomly add "pest" features
        if random.random() < 0.3:  # 30% chance of pest
            self._add_pest_features(image)
            source_info = "Synthetic: with pest"
        else:
            source_info = "Synthetic: healthy"
        
        return image, source_info
    
    def _add_pest_features(self, image: np.ndarray):
        """Add synthetic pest features to image."""
        h, w = image.shape[:2]
        
        # Add some dark spots (simulating disease)
        for _ in range(random.randint(1, 5)):
            center = (random.randint(20, w-20), random.randint(20, h-20))
            radius = random.randint(5, 15)
            cv2.circle(image, center, radius, (20, 40, 20), -1)
        
        # Add some small bright spots (simulating insects)
        for _ in range(random.randint(0, 3)):
            center = (random.randint(10, w-10), random.randint(10, h-10))
            radius = random.randint(2, 5)
            cv2.circle(image, center, radius, (200, 180, 150), -1)
    
    def stream_frames(self) -> Iterator[Tuple[np.ndarray, str]]:
        """
        Stream frames continuously.
        
        Yields:
            Tuple of (image, source_info)
        """
        while True:
            start_time = time.time()
            
            frame, info = self.get_frame()
            yield frame, info
            
            # Maintain FPS timing
            elapsed = time.time() - start_time
            if elapsed < self.frame_delay:
                time.sleep(self.frame_delay - elapsed)
    
    def save_test_dataset(self, output_dir: str, num_images: int = 100):
        """
        Generate and save a test dataset.
        
        Args:
            output_dir: Directory to save images
            num_images: Number of images to generate
        """
        os.makedirs(output_dir, exist_ok=True)
        
        pest_count = 0
        healthy_count = 0
        
        for i in range(num_images):
            image, info = self._generate_synthetic_frame()
            
            if "pest" in info.lower():
                filename = f"pest_{pest_count:03d}.jpg"
                pest_count += 1
            else:
                filename = f"healthy_{healthy_count:03d}.jpg"
                healthy_count += 1
            
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, image)
        
        print(f"Generated {num_images} test images:")
        print(f"  - {healthy_count} healthy images")
        print(f"  - {pest_count} pest images")
        print(f"Saved to: {output_dir}")


class AlertSimulator:
    """
    Simulates alert mechanisms (buzzer, display, IoT) for testing.
    """
    
    def __init__(self):
        self.alert_history = []
    
    def trigger_buzzer_alert(self, pest_type: str, severity: str):
        """Simulate buzzer alert."""
        alert_msg = f"BUZZER ALERT: {pest_type} detected - Severity: {severity}"
        print(f"🔊 {alert_msg}")
        self.alert_history.append(("buzzer", alert_msg, time.time()))
    
    def display_alert(self, detection_results: dict):
        """Simulate display alert."""
        print("\n" + "="*50)
        print("📱 DISPLAY ALERT")
        print("="*50)
        print(f"Status: {'🔴 PEST DETECTED' if detection_results['pest_detected'] else '🟢 HEALTHY'}")
        print(f"Type: {detection_results['class']}")
        print(f"Confidence: {detection_results['confidence']:.2%}")
        print(f"Severity: {detection_results['severity']}")
        
        if detection_results['pest_detected']:
            print("\nRecommendations:")
            for i, rec in enumerate(detection_results['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("="*50)
        
        self.alert_history.append(("display", detection_results, time.time()))
    
    def send_iot_alert(self, detection_results: dict):
        """Simulate IoT notification."""
        if detection_results['pest_detected']:
            message = f"IoT Alert: {detection_results['class']} detected on your crops. Confidence: {detection_results['confidence']:.1%}. Take action recommended."
            print(f"📡 {message}")
            self.alert_history.append(("iot", message, time.time()))
    
    def get_alert_summary(self):
        """Get summary of all alerts."""
        print(f"\nAlert History ({len(self.alert_history)} total alerts):")
        for alert_type, content, timestamp in self.alert_history[-10:]:  # Show last 10
            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            if alert_type == "display":
                print(f"  [{time_str}] DISPLAY: {content['class']} ({content['confidence']:.1%})")
            else:
                print(f"  [{time_str}] {alert_type.upper()}: {content}")


# Example usage and testing
if __name__ == "__main__":
    # Test camera simulator
    camera = CameraSimulator()
    alert_system = AlertSimulator()
    
    print("Testing Camera Simulator...")
    print("Press Ctrl+C to stop")
    
    try:
        frame_count = 0
        for image, info in camera.stream_frames():
            frame_count += 1
            
            # Display frame info
            print(f"Frame {frame_count}: {info} - Shape: {image.shape}")
            
            # Simulate some detection results
            if "pest" in info.lower():
                results = {
                    "pest_detected": True,
                    "class": "aphid",
                    "confidence": 0.75,
                    "severity": "medium",
                    "recommendations": ["Apply neem oil", "Remove affected leaves"]
                }
                alert_system.trigger_buzzer_alert("aphid", "medium")
                alert_system.display_alert(results)
                alert_system.send_iot_alert(results)
            
            # Stop after 10 frames for testing
            if frame_count >= 10:
                break
                
    except KeyboardInterrupt:
        print("\nSimulation stopped")
    
    # Show alert summary
    alert_system.get_alert_summary()
    
    # Generate test dataset
    print("\nGenerating test dataset...")
    camera.save_test_dataset("test_dataset", 20)