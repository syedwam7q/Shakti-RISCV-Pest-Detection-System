#!/usr/bin/env python3
"""
Enhanced Pest Detection Application - Real Data Integration
Supports both synthetic simulation and real PlantVillage dataset processing
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
from typing import Optional, Union, Dict, List
from pathlib import Path

from algorithms.pest_detector import PestDetector
from algorithms.enhanced_pest_detector import EnhancedPestDetector
from simulation.camera_simulator import CameraSimulator, AlertSimulator
from simulation.real_data_camera_simulator import RealDataCameraSimulator


class EnhancedPestDetectionSystem:
    """
    Enhanced pest detection system supporting both synthetic and real data.
    """
    
    def __init__(self, config_path: Optional[str] = None, data_mode: str = "synthetic"):
        """
        Initialize enhanced pest detection system.
        
        Args:
            config_path: Path to configuration file
            data_mode: "synthetic", "real", or "mixed"
        """
        self.config = self._load_config(config_path)
        self.data_mode = data_mode
        
        # Initialize components based on data mode
        self.detector = self._init_detector()
        self.camera = self._init_camera()
        self.alert_system = AlertSimulator()
        
        # Enhanced statistics
        self.stats = {
            "frames_processed": 0,
            "pests_detected": 0,
            "healthy_detected": 0,
            "alerts_sent": 0,
            "start_time": time.time(),
            "processing_times": [],
            "accuracy_stats": {"correct": 0, "total": 0},
            "class_detections": {}
        }
        
        self.running = False
        self.save_results = self.config.get("save_results", False)
        
        if self.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_file = f"demo_results/pest_detection_demo_{timestamp}.json"
            os.makedirs("demo_results", exist_ok=True)
            self.detection_history = []
        
        print(f"✅ Enhanced system initialized in '{data_mode}' mode")
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from file or use defaults."""
        default_config = {
            "image_dir": None,
            "real_dataset_dir": "datasets/real/plantvillage_organized",
            "camera_resolution": [640, 480],
            "fps": 30,
            "detection_interval": 1.0,
            "save_results": True,
            "alert_threshold": 0.6,
            "display_video": False,
            "max_images_per_class": 100,
            "detector_type": "enhanced",  # "basic", "enhanced", "real_model"
            "validation_mode": False
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                print(f"✅ Configuration loaded from {config_path}")
            except Exception as e:
                print(f"⚠️  Config error: {e}. Using defaults.")
        
        return default_config
    
    def _init_detector(self):
        """Initialize appropriate detector based on configuration."""
        detector_type = self.config.get("detector_type", "enhanced")
        
        if detector_type == "real_model":
            try:
                from demo_with_real_model import RealModelPestDetector
                print("🧠 Using real trained PlantVillage model")
                return RealModelPestDetector()
            except Exception as e:
                print(f"⚠️  Real model not available: {e}")
                print("   Falling back to enhanced detector")
                return EnhancedPestDetector()
        
        elif detector_type == "enhanced":
            print("🤖 Using enhanced rule-based detector")
            return EnhancedPestDetector()
        
        else:  # basic
            print("🔧 Using basic detector")
            return PestDetector()
    
    def _init_camera(self):
        """Initialize appropriate camera simulator."""
        if self.data_mode == "real":
            # Use real PlantVillage data
            dataset_dirs = [
                self.config.get("real_dataset_dir"),
                "datasets/real/plantvillage_organized",
                "datasets/real/plantvillage_organized_test",
                "datasets/real/plantvillage_organized_train"
            ]
            
            for dataset_dir in dataset_dirs:
                if dataset_dir and os.path.exists(dataset_dir):
                    print(f"📊 Using real dataset: {dataset_dir}")
                    return RealDataCameraSimulator(
                        dataset_dir=dataset_dir,
                        resolution=tuple(self.config.get("camera_resolution", [640, 480])),
                        fps=self.config.get("fps", 30),
                        max_images_per_class=self.config.get("max_images_per_class", 100)
                    )
            
            print("❌ No real dataset found, falling back to synthetic")
            self.data_mode = "synthetic"
        
        # Use synthetic data (default)
        print("🎭 Using synthetic camera simulation")
        return CameraSimulator(
            image_dir=self.config.get("image_dir"),
            resolution=tuple(self.config.get("camera_resolution", [640, 480])),
            fps=self.config.get("fps", 30)
        )
    
    def process_frame(self, image: np.ndarray, ground_truth: Optional[Dict] = None) -> dict:
        """
        Process a single frame for pest detection.
        
        Args:
            image: Input image from camera
            ground_truth: Ground truth data (for real data validation)
            
        Returns:
            Detection results with validation info
        """
        start_time = time.time()
        
        # Perform pest detection
        results = self.detector.detect_pests(image)
        
        processing_time = (time.time() - start_time) * 1000
        self.stats["processing_times"].append(processing_time)
        
        # Update basic statistics
        self.stats["frames_processed"] += 1
        if results.get("pest_detected", False):
            self.stats["pests_detected"] += 1
        else:
            self.stats["healthy_detected"] += 1
        
        # Track class detections
        detected_class = results.get("class", "unknown")
        self.stats["class_detections"][detected_class] = \
            self.stats["class_detections"].get(detected_class, 0) + 1
        
        # Validation against ground truth (real data mode)
        validation_info = {}
        if ground_truth and self.config.get("validation_mode", False):
            validation_info = self._validate_prediction(results, ground_truth)
        
        # Enhanced results
        enhanced_results = {
            **results,
            "timestamp": datetime.now().isoformat(),
            "frame_number": self.stats["frames_processed"],
            "processing_time_ms": processing_time,
            "data_source": self.data_mode,
            "ground_truth": ground_truth,
            "validation": validation_info
        }
        
        # Save results if enabled
        if self.save_results:
            self.detection_history.append(enhanced_results)
        
        return enhanced_results
    
    def _validate_prediction(self, prediction: Dict, ground_truth: Dict) -> Dict:
        """Validate prediction against ground truth."""
        gt_class = ground_truth.get("ground_truth", "unknown")
        pred_class = prediction.get("class", "unknown")
        
        # Basic pest vs healthy validation
        gt_is_pest = gt_class != "healthy"
        pred_is_pest = prediction.get("pest_detected", False)
        
        pest_healthy_correct = (gt_is_pest == pred_is_pest)
        exact_class_correct = (gt_class == pred_class)
        
        # Update accuracy stats
        self.stats["accuracy_stats"]["total"] += 1
        if pest_healthy_correct:
            self.stats["accuracy_stats"]["correct"] += 1
        
        return {
            "ground_truth_class": gt_class,
            "predicted_class": pred_class,
            "pest_vs_healthy_correct": pest_healthy_correct,
            "exact_class_correct": exact_class_correct,
            "confidence": prediction.get("confidence", 0.0)
        }
    
    def handle_alerts(self, results: dict):
        """Handle alert generation based on detection results."""
        if not results.get("pest_detected", False):
            return
        
        confidence = results.get("confidence", 0.0)
        if confidence < self.config["alert_threshold"]:
            return
        
        # Trigger appropriate alerts
        pest_type = results.get("class", "unknown")
        severity = results.get("severity", "unknown")
        
        # Always display results for detected pests
        self.alert_system.display_alert(results)
        
        # Trigger buzzer for medium/high severity
        if severity in ["medium", "high"]:
            self.alert_system.trigger_buzzer_alert(pest_type, severity)
            self.stats["alerts_sent"] += 1
        
        # Send IoT alert for high severity
        if severity == "high":
            self.alert_system.send_iot_alert(results)
    
    def display_frame(self, image: np.ndarray, results: dict):
        """Display frame with detection results overlay."""
        if not self.config["display_video"]:
            return
        
        display_img = image.copy()
        
        # Status overlay
        status_text = f"Status: {'🔴 PEST' if results.get('pest_detected') else '🟢 HEALTHY'}"
        color = (0, 0, 255) if results.get('pest_detected') else (0, 255, 0)
        cv2.putText(display_img, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Detection info
        if results.get("pest_detected"):
            pest_type = results.get("class", "unknown")
            confidence = results.get("confidence", 0.0)
            info_text = f"{pest_type}: {confidence:.1%}"
            cv2.putText(display_img, info_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Ground truth info (real data mode)
        if results.get("ground_truth"):
            gt_text = f"GT: {results['ground_truth'].get('ground_truth', 'unknown')}"
            cv2.putText(display_img, gt_text, (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Frame info
        frame_text = f"Frame: {self.stats['frames_processed']} | Mode: {self.data_mode}"
        cv2.putText(display_img, frame_text, (10, display_img.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Enhanced Pest Detection System', display_img)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False
        elif key == ord('s'):
            filename = f"detection_frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, display_img)
            print(f"📸 Frame saved as {filename}")
    
    def run_continuous(self, duration: Optional[float] = None):
        """Run continuous pest detection."""
        print(f"🚀 Starting Enhanced Pest Detection System")
        print(f"   Mode: {self.data_mode.upper()} data")
        print(f"   Detector: {type(self.detector).__name__}")
        print(f"   Press 'q' to quit, 's' to save frame")
        print("-" * 60)
        
        self.running = True
        last_detection_time = 0
        start_time = time.time()
        
        try:
            if hasattr(self.camera, 'stream_frames'):
                # Real data camera with metadata
                for image, metadata in self.camera.stream_frames(duration=duration):
                    if not self.running:
                        break
                    
                    current_time = time.time()
                    
                    # Process frame at specified intervals
                    if current_time - last_detection_time >= self.config["detection_interval"]:
                        results = self.process_frame(image, metadata if isinstance(metadata, dict) else None)
                        self.handle_alerts(results)
                        last_detection_time = current_time
                        
                        # Display progress
                        self._print_frame_status(results, metadata)
                    
                    # Display frame
                    self.display_frame(image, results if 'results' in locals() else {})
                    
                    # Check duration limit
                    if duration and (current_time - start_time) > duration:
                        break
            
            else:
                # Synthetic camera (original interface)
                for image, source_info in self.camera.stream_frames():
                    if not self.running:
                        break
                    
                    current_time = time.time()
                    
                    if current_time - last_detection_time >= self.config["detection_interval"]:
                        results = self.process_frame(image)
                        self.handle_alerts(results)
                        last_detection_time = current_time
                        
                        # Display progress
                        status = "🔴 PEST" if results.get("pest_detected") else "🟢 OK"
                        print(f"Frame {self.stats['frames_processed']:4d}: {status} - {source_info}")
                    
                    self.display_frame(image, results if 'results' in locals() else {})
                    
                    # Check duration
                    if duration and (current_time - start_time) > duration:
                        break
                    
        except KeyboardInterrupt:
            print("\n⏹️  Stopping system...")
        finally:
            self.cleanup()
    
    def _print_frame_status(self, results: dict, metadata: Union[str, Dict]):
        """Print frame processing status."""
        if isinstance(metadata, dict):
            # Real data with ground truth
            gt_class = metadata.get("ground_truth", "unknown")
            pred_class = results.get("class", "unknown")
            confidence = results.get("confidence", 0.0)
            
            # Validation check
            validation = results.get("validation", {})
            if validation:
                is_correct = validation.get("pest_vs_healthy_correct", False)
                status_icon = "✅" if is_correct else "❌"
            else:
                status_icon = "🔴" if results.get("pest_detected") else "🟢"
            
            print(f"Frame {self.stats['frames_processed']:4d}: {status_icon} GT:{gt_class} → PRED:{pred_class} ({confidence:.2f}) [{results.get('processing_time_ms', 0):.1f}ms]")
        else:
            # Synthetic data
            status = "🔴 PEST" if results.get("pest_detected") else "🟢 OK"
            print(f"Frame {self.stats['frames_processed']:4d}: {status} - {metadata}")
    
    def run_validation_demo(self, num_frames: int = 30):
        """Run validation demo comparing predictions to ground truth."""
        if self.data_mode != "real":
            print("❌ Validation demo requires real data mode")
            return
        
        print(f"🧪 VALIDATION DEMO - {num_frames} frames")
        print("=" * 60)
        
        self.config["validation_mode"] = True
        
        for i in range(num_frames):
            image, metadata = self.camera.get_frame()
            results = self.process_frame(image, metadata)
            
            # Print detailed validation
            validation = results.get("validation", {})
            gt_class = validation.get("ground_truth_class", "unknown")
            pred_class = validation.get("predicted_class", "unknown") 
            confidence = validation.get("confidence", 0.0)
            is_correct = validation.get("pest_vs_healthy_correct", False)
            
            status_icon = "✅" if is_correct else "❌"
            print(f"{i+1:2d}. {status_icon} GT:{gt_class} → PRED:{pred_class} ({confidence:.2f}) [{results.get('processing_time_ms', 0):.1f}ms]")
            
            # Show alerts for pest detections
            if results.get("pest_detected") and confidence > 0.7:
                self.alert_system.send_iot_alert(results)
        
        self._print_validation_summary()
    
    def _print_validation_summary(self):
        """Print validation results summary."""
        accuracy_stats = self.stats["accuracy_stats"]
        if accuracy_stats["total"] == 0:
            return
        
        accuracy = (accuracy_stats["correct"] / accuracy_stats["total"]) * 100
        avg_processing_time = np.mean(self.stats["processing_times"])
        fps = 1000 / avg_processing_time if avg_processing_time > 0 else 0
        
        print(f"\n" + "="*60)
        print("🎯 VALIDATION RESULTS")
        print("="*60)
        print(f"Accuracy: {accuracy:.1f}% ({accuracy_stats['correct']}/{accuracy_stats['total']})")
        print(f"Processing Speed: {avg_processing_time:.1f} ms avg | {fps:.1f} FPS")
        print(f"Pest Detections: {self.stats['pests_detected']}")
        print(f"Healthy Detections: {self.stats['healthy_detected']}")
        
        if self.stats["class_detections"]:
            print(f"Class Distribution:")
            for cls, count in self.stats["class_detections"].items():
                print(f"  {cls}: {count}")
        
        print("="*60)
    
    def run_performance_benchmark(self, num_frames: int = 100):
        """Run performance benchmark."""
        print(f"⚡ PERFORMANCE BENCHMARK - {num_frames} frames")
        print("=" * 50)
        
        processing_times = []
        
        for i in range(num_frames):
            if hasattr(self.camera, 'get_frame'):
                if self.data_mode == "real":
                    image, metadata = self.camera.get_frame()
                else:
                    image, _ = self.camera.get_frame()
            else:
                image, _ = self.camera.get_frame()
            
            start_time = time.time()
            results = self.detector.detect_pests(image)
            processing_time = (time.time() - start_time) * 1000
            processing_times.append(processing_time)
            
            if (i + 1) % 25 == 0:
                print(f"   Processed {i+1}/{num_frames} frames...")
        
        # Calculate statistics
        avg_time = np.mean(processing_times)
        std_time = np.std(processing_times)
        min_time = np.min(processing_times)
        max_time = np.max(processing_times)
        fps = 1000 / avg_time
        
        print(f"\n📊 BENCHMARK RESULTS:")
        print(f"   Average: {avg_time:.1f} ± {std_time:.1f} ms")
        print(f"   Range: {min_time:.1f} - {max_time:.1f} ms")
        print(f"   Theoretical FPS: {fps:.1f}")
        print(f"   Data Mode: {self.data_mode}")
        print(f"   Detector: {type(self.detector).__name__}")
        
        if fps >= 30:
            print(f"   ⚡ EXCELLENT: Real-time capable")
        elif fps >= 15:
            print(f"   🎯 GOOD: Near real-time")
        else:
            print(f"   ⚠️  ACCEPTABLE: Batch processing")
    
    def print_statistics(self):
        """Print comprehensive system statistics."""
        runtime = time.time() - self.stats["start_time"]
        fps = self.stats["frames_processed"] / runtime if runtime > 0 else 0
        avg_processing_time = np.mean(self.stats["processing_times"]) if self.stats["processing_times"] else 0
        
        print("\n" + "="*60)
        print("📊 SYSTEM STATISTICS")
        print("="*60)
        print(f"Runtime: {runtime:.1f} seconds")
        print(f"Frames processed: {self.stats['frames_processed']}")
        print(f"Average FPS: {fps:.1f}")
        print(f"Average processing time: {avg_processing_time:.1f} ms")
        print(f"Data mode: {self.data_mode}")
        print(f"Detector type: {type(self.detector).__name__}")
        print()
        print(f"Detection Results:")
        print(f"  Pests detected: {self.stats['pests_detected']}")
        print(f"  Healthy detected: {self.stats['healthy_detected']}")
        print(f"  Detection rate: {100 * self.stats['pests_detected'] / max(1, self.stats['frames_processed']):.1f}%")
        print(f"  Alerts sent: {self.stats['alerts_sent']}")
        
        # Accuracy stats (real data mode)
        if self.stats["accuracy_stats"]["total"] > 0:
            accuracy = (self.stats["accuracy_stats"]["correct"] / self.stats["accuracy_stats"]["total"]) * 100
            print(f"  Validation accuracy: {accuracy:.1f}%")
        
        # Class distribution
        if self.stats["class_detections"]:
            print(f"\nClass Detections:")
            for cls, count in self.stats["class_detections"].items():
                percentage = (count / self.stats['frames_processed']) * 100
                print(f"  {cls}: {count} ({percentage:.1f}%)")
        
        print("="*60)
    
    def cleanup(self):
        """Clean up resources and save results."""
        cv2.destroyAllWindows()
        self.print_statistics()
        self.alert_system.get_alert_summary()
        
        if self.save_results and hasattr(self, 'detection_history'):
            self._save_detection_history()
        
        print("✅ System shutdown complete.")
    
    def _save_detection_history(self):
        """Save comprehensive detection history."""
        if not self.detection_history:
            return
        
        try:
            report_data = {
                "system_info": {
                    "data_mode": self.data_mode,
                    "detector_type": type(self.detector).__name__,
                    "config": self.config,
                    "timestamp": datetime.now().isoformat()
                },
                "statistics": self.stats,
                "detections": self.detection_history
            }
            
            with open(self.results_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"📄 Results saved to: {self.results_file}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")


def main():
    """Enhanced main function with extended command-line interface."""
    parser = argparse.ArgumentParser(description='Enhanced Pest Detection System')
    
    # Basic options
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--mode', choices=['continuous', 'validation', 'benchmark'], 
                       default='continuous', help='Operation mode')
    
    # Data options  
    parser.add_argument('--data-mode', choices=['synthetic', 'real'], 
                       default='synthetic', help='Data source mode')
    parser.add_argument('--real-dataset', help='Real dataset directory path')
    
    # Demo options
    parser.add_argument('--frames', type=int, default=30, 
                       help='Number of frames for validation/benchmark')
    parser.add_argument('--duration', type=float, 
                       help='Duration in seconds for continuous mode')
    
    # Detector options
    parser.add_argument('--detector', choices=['basic', 'enhanced', 'real_model'],
                       default='enhanced', help='Detector type to use')
    
    # Output options
    parser.add_argument('--save-results', action='store_true', 
                       help='Save detection results to file')
    parser.add_argument('--display', action='store_true', 
                       help='Show video display window')
    
    args = parser.parse_args()
    
    # Prepare configuration
    config_overrides = {}
    if args.real_dataset:
        config_overrides['real_dataset_dir'] = args.real_dataset
    if args.detector:
        config_overrides['detector_type'] = args.detector
    if args.save_results:
        config_overrides['save_results'] = True
    if args.display:
        config_overrides['display_video'] = True
    
    # Initialize system
    try:
        # Create a temporary config file if we have overrides
        temp_config_path = None
        if config_overrides:
            import tempfile
            temp_config = {}
            if args.config and os.path.exists(args.config):
                with open(args.config, 'r') as f:
                    temp_config = json.load(f)
            temp_config.update(config_overrides)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(temp_config, f)
                temp_config_path = f.name
        
        system = EnhancedPestDetectionSystem(temp_config_path or args.config, args.data_mode)
        
        # Clean up temp file
        if temp_config_path:
            os.unlink(temp_config_path)
        
        print(f"🎯 Mode: {args.mode} | Data: {args.data_mode} | Detector: {args.detector}")
        
        # Run in specified mode
        if args.mode == 'continuous':
            system.run_continuous(duration=args.duration)
        
        elif args.mode == 'validation':
            if args.data_mode != 'real':
                print("❌ Validation mode requires --data-mode real")
                return
            system.run_validation_demo(num_frames=args.frames)
        
        elif args.mode == 'benchmark':
            system.run_performance_benchmark(num_frames=args.frames)
        
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()