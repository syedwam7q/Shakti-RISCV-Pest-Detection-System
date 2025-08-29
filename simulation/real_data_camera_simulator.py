#!/usr/bin/env python3
"""
Real Data Camera Simulator for PlantVillage Dataset
Simulates camera input using real crop disease images for authentic testing
"""

import cv2
import numpy as np
import os
import time
import random
from typing import Optional, Iterator, Tuple, List, Dict
from pathlib import Path
import json


class RealDataCameraSimulator:
    """
    Camera simulator that uses real PlantVillage dataset images.
    Provides ground truth labels for validation and testing.
    """
    
    def __init__(self, 
                 dataset_dir: str,
                 resolution: Tuple[int, int] = (640, 480),
                 fps: int = 30,
                 shuffle: bool = True,
                 max_images_per_class: Optional[int] = None):
        """
        Initialize real data camera simulator.
        
        Args:
            dataset_dir: Directory containing organized PlantVillage data
            resolution: Target resolution for images
            fps: Simulated frames per second
            shuffle: Whether to shuffle image order
            max_images_per_class: Limit images per class (for faster testing)
        """
        self.dataset_dir = dataset_dir
        self.resolution = resolution
        self.fps = fps
        self.frame_delay = 1.0 / fps
        self.shuffle = shuffle
        self.max_images_per_class = max_images_per_class
        
        self.images = []  # List of (image_array, class_name, filename, filepath)
        self.current_index = 0
        self.class_counts = {}
        self.class_info = {}
        
        # Load real dataset
        self._load_real_dataset()
        
        # Shuffle if requested
        if shuffle:
            random.shuffle(self.images)
    
    def _load_real_dataset(self):
        """Load real PlantVillage dataset from organized directories."""
        dataset_path = Path(self.dataset_dir)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        
        print(f"📁 Loading real dataset from: {self.dataset_dir}")
        
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        total_loaded = 0
        
        # Scan class directories
        for class_dir in dataset_path.iterdir():
            if not class_dir.is_dir() or class_dir.name.startswith('.'):
                continue
            
            class_name = class_dir.name
            class_images = []
            
            # Get all image files in class directory
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in supported_formats:
                    class_images.append(img_file)
            
            # Limit images per class if specified
            if self.max_images_per_class and len(class_images) > self.max_images_per_class:
                class_images = random.sample(class_images, self.max_images_per_class)
            
            # Load images for this class
            loaded_count = 0
            for img_path in class_images:
                try:
                    image = cv2.imread(str(img_path))
                    if image is not None:
                        # Resize to target resolution
                        image = cv2.resize(image, self.resolution)
                        
                        # Store image data
                        self.images.append((image, class_name, img_path.name, str(img_path)))
                        loaded_count += 1
                        total_loaded += 1
                        
                except Exception as e:
                    print(f"   ⚠️  Error loading {img_path}: {e}")
            
            if loaded_count > 0:
                self.class_counts[class_name] = loaded_count
                self.class_info[class_name] = {
                    'count': loaded_count,
                    'description': self._get_class_description(class_name),
                    'severity': self._get_class_severity(class_name)
                }
                print(f"   ✅ {class_name}: {loaded_count} images")
        
        if not self.images:
            raise ValueError("No images loaded from dataset")
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Total images: {total_loaded}")
        print(f"   Classes: {len(self.class_counts)} - {list(self.class_counts.keys())}")
        print(f"   Resolution: {self.resolution}")
        
    def _get_class_description(self, class_name: str) -> str:
        """Get human-readable description of pest/disease class."""
        descriptions = {
            'healthy': 'Healthy crop with no visible issues',
            'leaf_spot': 'Fungal leaf spot disease with dark circular lesions',
            'spider_mites': 'Spider mite infestation causing stippling and webbing',
            'aphid': 'Aphid infestation with small insects on leaves',
            'whitefly': 'Whitefly infestation with small white flying insects',
            'powdery_mildew': 'Powdery mildew fungal infection with white coating',
            'bacterial_blight': 'Bacterial infection causing leaf blight',
            'mosaic_virus': 'Viral infection causing mosaic patterns on leaves'
        }
        return descriptions.get(class_name, f'Plant condition: {class_name}')
    
    def _get_class_severity(self, class_name: str) -> str:
        """Get severity level for pest/disease class."""
        severities = {
            'healthy': 'none',
            'leaf_spot': 'medium',
            'spider_mites': 'high',
            'aphid': 'medium',
            'whitefly': 'medium',
            'powdery_mildew': 'low',
            'bacterial_blight': 'high',
            'mosaic_virus': 'high'
        }
        return severities.get(class_name, 'medium')
    
    def get_frame(self) -> Tuple[np.ndarray, Dict]:
        """
        Get next frame from real dataset.
        
        Returns:
            Tuple of (image, metadata_dict)
        """
        if not self.images:
            raise ValueError("No images available")
        
        # Get current image
        image, class_name, filename, filepath = self.images[self.current_index]
        
        # Prepare metadata
        metadata = {
            'ground_truth': class_name,
            'filename': filename,
            'filepath': filepath,
            'class_description': self.class_info[class_name]['description'],
            'severity': self.class_info[class_name]['severity'],
            'pest_detected': class_name != 'healthy',
            'source': 'real_plantvillage',
            'frame_index': self.current_index,
            'total_frames': len(self.images)
        }
        
        # Advance to next image (cycle through)
        self.current_index = (self.current_index + 1) % len(self.images)
        
        return image.copy(), metadata
    
    def get_random_frame(self) -> Tuple[np.ndarray, Dict]:
        """Get a random frame from the dataset."""
        original_index = self.current_index
        self.current_index = random.randint(0, len(self.images) - 1)
        frame, metadata = self.get_frame()
        self.current_index = original_index
        return frame, metadata
    
    def get_frames_by_class(self, class_name: str, count: int = 5) -> List[Tuple[np.ndarray, Dict]]:
        """Get specific frames from a particular class."""
        class_frames = []
        
        for image, cls, filename, filepath in self.images:
            if cls == class_name:
                metadata = {
                    'ground_truth': cls,
                    'filename': filename,
                    'filepath': filepath,
                    'class_description': self.class_info[cls]['description'],
                    'severity': self.class_info[cls]['severity'],
                    'pest_detected': cls != 'healthy',
                    'source': 'real_plantvillage'
                }
                class_frames.append((image.copy(), metadata))
                
                if len(class_frames) >= count:
                    break
        
        return class_frames
    
    def stream_frames(self, duration: Optional[float] = None) -> Iterator[Tuple[np.ndarray, Dict]]:
        """
        Stream frames continuously.
        
        Args:
            duration: Maximum duration in seconds (None for unlimited)
            
        Yields:
            Tuple of (image, metadata_dict)
        """
        start_time = time.time()
        
        while True:
            # Check duration limit
            if duration and (time.time() - start_time) > duration:
                break
                
            frame_start = time.time()
            
            frame, metadata = self.get_frame()
            yield frame, metadata
            
            # Maintain FPS timing
            elapsed = time.time() - frame_start
            if elapsed < self.frame_delay:
                time.sleep(self.frame_delay - elapsed)
    
    def get_class_distribution(self) -> Dict:
        """Get distribution of classes in the dataset."""
        return {
            'counts': self.class_counts.copy(),
            'total': len(self.images),
            'percentages': {cls: (count / len(self.images)) * 100 
                          for cls, count in self.class_counts.items()}
        }
    
    def export_sample_frames(self, output_dir: str, samples_per_class: int = 3):
        """Export sample frames for visualization."""
        os.makedirs(output_dir, exist_ok=True)
        
        for class_name in self.class_counts.keys():
            class_frames = self.get_frames_by_class(class_name, samples_per_class)
            
            for i, (image, metadata) in enumerate(class_frames):
                filename = f"sample_{class_name}_{i+1:02d}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, image)
        
        print(f"📸 Exported sample frames to: {output_dir}")


class RealDataSimulationSystem:
    """
    Complete simulation system using real PlantVillage data.
    Integrates camera simulator with pest detection and alerts.
    """
    
    def __init__(self, 
                 dataset_dir: str,
                 detector_type: str = "enhanced",
                 max_images_per_class: int = 50):
        """Initialize real data simulation system."""
        self.dataset_dir = dataset_dir
        self.detector_type = detector_type
        
        # Initialize components
        self.camera = RealDataCameraSimulator(
            dataset_dir, 
            max_images_per_class=max_images_per_class
        )
        
        # Initialize detector based on type
        if detector_type == "real_model":
            self._init_real_model_detector()
        else:
            self._init_enhanced_detector()
        
        # Initialize alert system
        from camera_simulator import AlertSimulator
        self.alert_system = AlertSimulator()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'correct_predictions': 0,
            'pest_detected': 0,
            'healthy_detected': 0,
            'processing_times': [],
            'class_accuracy': {}
        }
    
    def _init_enhanced_detector(self):
        """Initialize enhanced pest detector."""
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from algorithms.enhanced_pest_detector import EnhancedPestDetector
        self.detector = EnhancedPestDetector()
    
    def _init_real_model_detector(self):
        """Initialize real trained model detector."""
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        try:
            from demo_with_real_model import RealModelPestDetector
            self.detector = RealModelPestDetector()
            print("✅ Using real trained PlantVillage model")
        except Exception as e:
            print(f"⚠️  Real model not available: {e}")
            print("   Falling back to enhanced detector")
            self._init_enhanced_detector()
    
    def process_frame(self, image: np.ndarray, ground_truth: Dict) -> Dict:
        """Process a single frame and compare with ground truth."""
        start_time = time.time()
        
        # Run pest detection
        if hasattr(self.detector, 'detect_pests'):
            detection_result = self.detector.detect_pests(image)
        else:
            # Fallback for different detector interfaces
            detection_result = {
                'pest_detected': True,
                'class': 'unknown',
                'confidence': 0.5,
                'severity': 'medium'
            }
        
        processing_time = (time.time() - start_time) * 1000
        
        # Compare with ground truth
        gt_class = ground_truth['ground_truth']
        predicted_class = detection_result.get('class', 'unknown')
        
        # Determine if prediction is correct
        gt_is_pest = gt_class != 'healthy'
        pred_is_pest = detection_result.get('pest_detected', False)
        
        # Simple accuracy: correct if both agree on pest/healthy
        is_correct = (gt_is_pest == pred_is_pest)
        
        # More detailed accuracy: correct if exact class match
        exact_match = (gt_class == predicted_class)
        
        # Update statistics
        self._update_stats(gt_class, predicted_class, is_correct, exact_match, processing_time)
        
        # Prepare result
        result = {
            'detection': detection_result,
            'ground_truth': ground_truth,
            'accuracy': {
                'pest_vs_healthy_correct': is_correct,
                'exact_class_correct': exact_match
            },
            'processing_time_ms': processing_time,
            'frame_info': {
                'filename': ground_truth['filename'],
                'gt_class': gt_class,
                'pred_class': predicted_class,
                'confidence': detection_result.get('confidence', 0.0)
            }
        }
        
        return result
    
    def _update_stats(self, gt_class: str, pred_class: str, is_correct: bool, exact_match: bool, processing_time: float):
        """Update internal statistics."""
        self.stats['total_processed'] += 1
        self.stats['processing_times'].append(processing_time)
        
        if is_correct:
            self.stats['correct_predictions'] += 1
        
        if gt_class == 'healthy':
            self.stats['healthy_detected'] += 1
        else:
            self.stats['pest_detected'] += 1
        
        # Per-class accuracy
        if gt_class not in self.stats['class_accuracy']:
            self.stats['class_accuracy'][gt_class] = {'correct': 0, 'total': 0}
        
        self.stats['class_accuracy'][gt_class]['total'] += 1
        if exact_match:
            self.stats['class_accuracy'][gt_class]['correct'] += 1
    
    def run_validation_demo(self, num_frames: int = 25):
        """Run validation demo comparing predictions to ground truth."""
        print(f"🧪 REAL DATA VALIDATION DEMO")
        print("=" * 50)
        print(f"Processing {num_frames} real PlantVillage images...")
        print(f"Dataset: {self.dataset_dir}")
        print("-" * 50)
        
        correct_count = 0
        
        for i in range(num_frames):
            # Get real image with ground truth
            image, ground_truth = self.camera.get_frame()
            
            # Process with our detector
            result = self.process_frame(image, ground_truth)
            
            # Display result
            gt_class = ground_truth['ground_truth']
            pred_class = result['detection'].get('class', 'unknown')
            confidence = result['detection'].get('confidence', 0.0)
            is_correct = result['accuracy']['pest_vs_healthy_correct']
            
            if is_correct:
                correct_count += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"Frame {i+1:2d}: {status} GT:{gt_class} → PRED:{pred_class} ({confidence:.2f}) [{result['processing_time_ms']:.1f}ms]")
            
            # Show alerts for pest detections
            if result['detection'].get('pest_detected', False):
                self.alert_system.send_iot_alert(result['detection'])
        
        # Show final statistics
        self._show_validation_results()
    
    def run_continuous_demo(self, duration: float = 60.0):
        """Run continuous processing demo."""
        print(f"📹 CONTINUOUS REAL DATA DEMO")
        print("=" * 50)
        print(f"Running for {duration} seconds with real images...")
        print("Press Ctrl+C to stop early")
        print("-" * 50)
        
        try:
            frame_count = 0
            for image, ground_truth in self.camera.stream_frames(duration=duration):
                frame_count += 1
                
                result = self.process_frame(image, ground_truth)
                
                # Display progress
                gt_class = ground_truth['ground_truth']
                pred_class = result['detection'].get('class', 'unknown')
                confidence = result['detection'].get('confidence', 0.0)
                
                print(f"Frame {frame_count:3d}: {gt_class} → {pred_class} ({confidence:.2f})", end='\r')
                
                # Handle alerts
                if result['detection'].get('pest_detected', False) and confidence > 0.7:
                    print(f"\n🚨 PEST ALERT: {pred_class} detected in {ground_truth['filename']}")
                    self.alert_system.trigger_buzzer_alert(pred_class, result['detection'].get('severity', 'medium'))
                
                time.sleep(0.1)  # Brief pause for readability
        
        except KeyboardInterrupt:
            print(f"\n\nDemo stopped by user after {frame_count} frames")
        
        self._show_validation_results()
    
    def _show_validation_results(self):
        """Display validation results and statistics."""
        if self.stats['total_processed'] == 0:
            return
        
        print(f"\n" + "="*60)
        print("📊 VALIDATION RESULTS")
        print("="*60)
        
        # Overall accuracy
        accuracy = (self.stats['correct_predictions'] / self.stats['total_processed']) * 100
        print(f"Overall Accuracy: {accuracy:.1f}% ({self.stats['correct_predictions']}/{self.stats['total_processed']})")
        
        # Processing performance
        avg_time = np.mean(self.stats['processing_times'])
        fps = 1000 / avg_time if avg_time > 0 else 0
        print(f"Processing Speed: {avg_time:.1f} ms avg | {fps:.1f} FPS")
        
        # Class distribution in test
        print(f"Test Distribution:")
        print(f"  Healthy: {self.stats['healthy_detected']}")
        print(f"  Pest/Disease: {self.stats['pest_detected']}")
        
        # Per-class accuracy
        if self.stats['class_accuracy']:
            print(f"\nPer-Class Accuracy:")
            for class_name, class_stats in self.stats['class_accuracy'].items():
                if class_stats['total'] > 0:
                    class_acc = (class_stats['correct'] / class_stats['total']) * 100
                    print(f"  {class_name}: {class_acc:.1f}% ({class_stats['correct']}/{class_stats['total']})")
        
        # Dataset info
        distribution = self.camera.get_class_distribution()
        print(f"\nDataset Info:")
        print(f"  Total Classes: {len(distribution['counts'])}")
        print(f"  Available Images: {distribution['total']}")
        for cls, count in distribution['counts'].items():
            print(f"    {cls}: {count} images")
        
        print("="*60)


def main():
    """Demo the real data simulation system."""
    dataset_dirs = [
        "datasets/real/plantvillage_organized",
        "datasets/real/plantvillage_organized_train",
        "datasets/real/plantvillage_organized_test"
    ]
    
    # Find available dataset
    dataset_dir = None
    for dir_path in dataset_dirs:
        if os.path.exists(dir_path):
            dataset_dir = dir_path
            break
    
    if not dataset_dir:
        print("❌ No organized PlantVillage dataset found!")
        print("💡 Please organize dataset first:")
        print("   python3 tools/organize_real_data.py datasets/real/plantvillage --type plantvillage --validate --split")
        return
    
    print(f"🌱 REAL DATA SIMULATION DEMO")
    print("=" * 50)
    
    # Initialize simulation system
    try:
        simulator = RealDataSimulationSystem(
            dataset_dir=dataset_dir,
            detector_type="enhanced",  # or "real_model"
            max_images_per_class=30    # Limit for demo speed
        )
        
        print(f"✅ Simulation system initialized")
        print(f"📊 Dataset loaded: {len(simulator.camera.images)} images")
        
        # Run demo
        choice = input("\nSelect demo:\n1. Validation (25 frames)\n2. Continuous (60 sec)\n3. Class samples\nChoice (1-3): ").strip()
        
        if choice == "1":
            simulator.run_validation_demo(num_frames=25)
        elif choice == "2":
            simulator.run_continuous_demo(duration=60.0)
        elif choice == "3":
            # Show samples from each class
            for class_name in simulator.camera.class_counts.keys():
                print(f"\n📷 Samples from {class_name}:")
                frames = simulator.camera.get_frames_by_class(class_name, 3)
                for i, (image, metadata) in enumerate(frames):
                    result = simulator.process_frame(image, metadata)
                    pred = result['detection'].get('class', 'unknown')
                    conf = result['detection'].get('confidence', 0.0)
                    print(f"   {i+1}. {metadata['filename']} → Predicted: {pred} ({conf:.2f})")
        else:
            print("Running quick validation demo...")
            simulator.run_validation_demo(num_frames=10)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()