"""
Basic functionality tests for pest detection system
Run this to verify everything is working correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from algorithms.image_processor import ImageProcessor
from algorithms.pest_detector import PestDetector
from simulation.camera_simulator import CameraSimulator, AlertSimulator


def test_image_processor():
    """Test image processing functionality."""
    print("Testing Image Processor...")
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize processor
    processor = ImageProcessor()
    
    try:
        # Test preprocessing
        processed = processor.preprocess_image(test_image)
        assert processed.shape[:2] == (224, 224), f"Expected (224, 224), got {processed.shape[:2]}"
        assert processed.dtype == np.float32, f"Expected float32, got {processed.dtype}"
        assert 0 <= processed.max() <= 1, f"Expected values in [0,1], got max {processed.max()}"
        
        # Test feature extraction
        features = processor.extract_features(processed)
        assert len(features) > 0, "No features extracted"
        
        # Test segmentation
        regions = processor.segment_regions(processed)
        assert isinstance(regions, list), "Regions should be a list"
        
        print("✅ Image Processor: All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Image Processor: Test failed - {e}")
        return False


def test_pest_detector():
    """Test pest detection functionality."""
    print("Testing Pest Detector...")
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize detector
    detector = PestDetector()
    
    try:
        # Test detection
        results = detector.detect_pests(test_image)
        
        # Verify result structure
        expected_keys = ["pest_detected", "class", "confidence", "affected_regions", "severity", "recommendations"]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
        
        assert isinstance(results["pest_detected"], bool), "pest_detected should be boolean"
        assert isinstance(results["confidence"], (int, float)), "confidence should be numeric"
        assert 0 <= results["confidence"] <= 1, f"Confidence should be in [0,1], got {results['confidence']}"
        assert isinstance(results["recommendations"], list), "recommendations should be a list"
        
        print("✅ Pest Detector: All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Pest Detector: Test failed - {e}")
        return False


def test_camera_simulator():
    """Test camera simulation functionality."""
    print("Testing Camera Simulator...")
    
    try:
        # Initialize simulator
        camera = CameraSimulator(resolution=(320, 240))
        
        # Test single frame
        frame, info = camera.get_frame()
        assert frame.shape == (240, 320, 3), f"Expected (240, 320, 3), got {frame.shape}"
        assert isinstance(info, str), "Info should be string"
        assert frame.dtype == np.uint8, f"Expected uint8, got {frame.dtype}"
        
        # Test multiple frames
        frame_count = 0
        for frame, info in camera.stream_frames():
            frame_count += 1
            assert frame.shape == (240, 320, 3), "Frame shape changed"
            if frame_count >= 3:  # Test just a few frames
                break
        
        print("✅ Camera Simulator: All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Camera Simulator: Test failed - {e}")
        return False


def test_alert_system():
    """Test alert system functionality."""
    print("Testing Alert System...")
    
    try:
        # Initialize alert system
        alert_system = AlertSimulator()
        
        # Test different alert types
        alert_system.trigger_buzzer_alert("aphid", "high")
        
        test_results = {
            "pest_detected": True,
            "class": "whitefly",
            "confidence": 0.85,
            "severity": "medium",
            "recommendations": ["Apply insecticidal soap", "Remove affected leaves"]
        }
        alert_system.display_alert(test_results)
        alert_system.send_iot_alert(test_results)
        
        # Verify history
        assert len(alert_system.alert_history) > 0, "No alerts recorded"
        
        print("✅ Alert System: All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Alert System: Test failed - {e}")
        return False


def test_integration():
    """Test integration of all components."""
    print("Testing System Integration...")
    
    try:
        # Initialize all components
        detector = PestDetector()
        camera = CameraSimulator()
        alert_system = AlertSimulator()
        
        # Simulate complete detection pipeline
        for i in range(3):  # Test 3 frames
            # Get frame
            frame, info = camera.get_frame()
            
            # Detect pests
            results = detector.detect_pests(frame)
            
            # Handle alerts
            if results.get("pest_detected", False):
                alert_system.display_alert(results)
                if results.get("severity") in ["medium", "high"]:
                    alert_system.trigger_buzzer_alert(results["class"], results["severity"])
        
        print("✅ System Integration: All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ System Integration: Test failed - {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Running Pest Detection System Tests")
    print("=" * 50)
    
    tests = [
        test_image_processor,
        test_pest_detector,
        test_camera_simulator,
        test_alert_system,
        test_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Your system is ready.")
        print("\nNext steps:")
        print("1. Run: python simulation/main_app.py --mode continuous")
        print("2. Try: python simulation/main_app.py --generate-dataset 50")
        print("3. See: docs/getting_started.md for more options")
    else:
        print("⚠️  Some tests failed. Check error messages above.")
        print("Make sure you have installed all requirements: pip install -r requirements.txt")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)