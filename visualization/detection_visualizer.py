"""
Advanced Detection Visualizer for Shakti RISC-V Pest Detection
=============================================================

Comprehensive visual representation system with:
- Bounding box overlays with confidence scores
- Real-time performance metrics display
- Color-coded severity indicators
- Detection history visualization
- Memory-optimized for embedded systems
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import os

from .bounding_box_detector import BoundingBoxDetector


class DetectionVisualizer:
    """
    Advanced detection visualization system optimized for Shakti RISC-V.
    Provides comprehensive visual feedback for pest detection results.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.bbox_detector = BoundingBoxDetector()
        self.detection_history = []
        self.performance_metrics = {
            'frame_count': 0,
            'total_processing_time': 0.0,
            'fps_history': [],
            'detection_count': 0
        }
        
        # Visual settings optimized for embedded display
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 1
        self.line_thickness = 2
        
        # Color scheme for different elements
        self.colors = {
            'healthy': (0, 255, 0),
            'pest_detected': (0, 0, 255),
            'warning': (0, 165, 255),
            'info': (255, 255, 255),
            'background': (0, 0, 0),
            'overlay': (50, 50, 50)
        }
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for embedded systems."""
        return {
            'display_bounding_boxes': True,
            'display_confidence_heatmap': True,
            'display_performance_metrics': True,
            'display_detection_history': True,
            'max_history_items': 20,  # Memory constraint
            'confidence_threshold': 0.5,
            'enable_annotations': True,
            'save_annotated_frames': False,
            'output_resolution': (640, 480),  # Optimized for embedded display
            'overlay_transparency': 0.7
        }
    
    def create_comprehensive_visualization(self, image: np.ndarray, 
                                         detection_results: Dict,
                                         processing_time: float = 0.0) -> np.ndarray:
        """
        Create comprehensive visualization with all visual elements.
        
        Args:
            image: Input image
            detection_results: Results from pest detection algorithm
            processing_time: Time taken for processing (for performance metrics)
            
        Returns:
            Fully annotated image with all visual elements
        """
        start_time = time.time()
        
        # Create working copy
        annotated_image = image.copy()
        
        # Update performance metrics
        self._update_performance_metrics(processing_time)
        
        # Extract or detect regions for bounding boxes
        regions = self._extract_regions(image, detection_results)
        
        # Apply bounding boxes if enabled
        if self.config['display_bounding_boxes'] and regions:
            annotated_image = self.bbox_detector.draw_bounding_boxes(annotated_image, regions)
        
        # Apply confidence heatmap if enabled
        if self.config['display_confidence_heatmap'] and regions:
            heatmap_overlay = self.bbox_detector.create_confidence_heatmap(image, regions)
            annotated_image = cv2.addWeighted(
                annotated_image, 
                1 - self.config['overlay_transparency'], 
                heatmap_overlay, 
                self.config['overlay_transparency'], 
                0
            )
        
        # Add main status overlay
        annotated_image = self._add_status_overlay(annotated_image, detection_results)
        
        # Add performance metrics if enabled
        if self.config['display_performance_metrics']:
            annotated_image = self._add_performance_overlay(annotated_image)
        
        # Add detection summary
        if regions:
            summary = self.bbox_detector.get_detection_summary(regions)
            annotated_image = self._add_detection_summary_overlay(annotated_image, summary)
        
        # Add detection history if enabled
        if self.config['display_detection_history']:
            annotated_image = self._add_history_overlay(annotated_image)
        
        # Update detection history
        self._update_detection_history(detection_results, regions)
        
        # Save frame if configured
        if self.config['save_annotated_frames']:
            self._save_annotated_frame(annotated_image, detection_results)
        
        visualization_time = time.time() - start_time
        print(f"Visualization time: {visualization_time:.3f}s")
        
        return annotated_image
    
    def _extract_regions(self, image: np.ndarray, detection_results: Dict) -> List[Dict]:
        """Extract or detect regions for bounding box visualization."""
        # If regions are already provided in detection results, use them
        if 'regions' in detection_results:
            return detection_results['regions']
        
        # Otherwise, detect regions based on detection results
        if detection_results.get('pest_detected', False):
            confidence_map = None
            if 'confidence_map' in detection_results:
                confidence_map = detection_results['confidence_map']
            
            regions = self.bbox_detector.detect_pest_regions(image, confidence_map)
            
            # Enhance regions with detection results
            for region in regions:
                if detection_results.get('class'):
                    region['pest_type'] = detection_results['class']
                if detection_results.get('confidence'):
                    region['confidence'] = max(region['confidence'], detection_results['confidence'])
            
            return regions
        
        return []
    
    def _add_status_overlay(self, image: np.ndarray, detection_results: Dict) -> np.ndarray:
        """Add main status overlay with detection information."""
        overlay = image.copy()
        
        # Status header
        pest_detected = detection_results.get('pest_detected', False)
        status_text = "🔴 PEST DETECTED" if pest_detected else "🟢 HEALTHY CROP"
        status_color = self.colors['pest_detected'] if pest_detected else self.colors['healthy']
        
        # Draw status background
        cv2.rectangle(overlay, (10, 10), (350, 60), self.colors['overlay'], -1)
        
        # Draw status text
        cv2.putText(overlay, status_text, (20, 35), 
                   self.font_face, self.font_scale + 0.2, status_color, self.font_thickness + 1)
        
        # Add detailed information if pest detected
        if pest_detected:
            pest_class = detection_results.get('class', 'Unknown')
            confidence = detection_results.get('confidence', 0.0)
            severity = detection_results.get('severity', 'low')
            
            detail_text = f"Type: {pest_class} | Confidence: {confidence:.1%} | Severity: {severity.upper()}"
            cv2.putText(overlay, detail_text, (20, 55), 
                       self.font_face, self.font_scale - 0.1, self.colors['info'], self.font_thickness)
        
        # Blend overlay
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        return result
    
    def _add_performance_overlay(self, image: np.ndarray) -> np.ndarray:
        """Add performance metrics overlay."""
        height, width = image.shape[:2]
        
        # Calculate current FPS
        current_fps = self._calculate_current_fps()
        avg_fps = np.mean(self.performance_metrics['fps_history'][-10:]) if self.performance_metrics['fps_history'] else 0
        
        # Performance text
        perf_lines = [
            f"FPS: {current_fps:.1f} (Avg: {avg_fps:.1f})",
            f"Frames: {self.performance_metrics['frame_count']}",
            f"Detections: {self.performance_metrics['detection_count']}"
        ]
        
        # Draw performance overlay
        overlay_height = len(perf_lines) * 25 + 20
        cv2.rectangle(image, (width - 250, 10), (width - 10, overlay_height), 
                     self.colors['overlay'], -1)
        
        for i, line in enumerate(perf_lines):
            y_pos = 35 + i * 25
            cv2.putText(image, line, (width - 240, y_pos),
                       self.font_face, self.font_scale, self.colors['info'], self.font_thickness)
        
        return image
    
    def _add_detection_summary_overlay(self, image: np.ndarray, summary: Dict) -> np.ndarray:
        """Add detection summary overlay."""
        height, width = image.shape[:2]
        
        # Summary information
        total_regions = summary.get('total_regions', 0)
        pest_types = summary.get('pest_types', {})
        avg_confidence = summary.get('average_confidence', 0.0)
        
        # Create summary text
        summary_lines = [
            f"Regions: {total_regions}",
            f"Avg Conf: {avg_confidence:.1%}"
        ]
        
        # Add pest type breakdown
        for pest_type, count in pest_types.items():
            if count > 0:
                summary_lines.append(f"{pest_type}: {count}")
        
        # Draw summary overlay
        overlay_height = len(summary_lines) * 20 + 15
        y_start = height - overlay_height - 10
        
        cv2.rectangle(image, (10, y_start), (200, height - 10), 
                     self.colors['overlay'], -1)
        
        for i, line in enumerate(summary_lines):
            y_pos = y_start + 20 + i * 20
            cv2.putText(image, line, (20, y_pos),
                       self.font_face, self.font_scale - 0.1, self.colors['info'], 1)
        
        return image
    
    def _add_history_overlay(self, image: np.ndarray) -> np.ndarray:
        """Add detection history overlay."""
        if not self.detection_history:
            return image
        
        height, width = image.shape[:2]
        
        # Get recent history
        recent_history = self.detection_history[-5:]  # Last 5 detections
        
        # Create history display
        history_lines = ["Recent Detections:"]
        for i, entry in enumerate(recent_history):
            timestamp = datetime.fromtimestamp(entry['timestamp']).strftime("%H:%M:%S")
            pest_type = entry.get('pest_type', 'unknown')
            confidence = entry.get('confidence', 0.0)
            history_lines.append(f"{timestamp}: {pest_type} ({confidence:.1%})")
        
        # Draw history overlay
        overlay_height = len(history_lines) * 18 + 15
        x_start = width - 280
        y_start = height - overlay_height - 10
        
        cv2.rectangle(image, (x_start, y_start), (width - 10, height - 10), 
                     self.colors['overlay'], -1)
        
        for i, line in enumerate(history_lines):
            y_pos = y_start + 18 + i * 18
            font_scale = self.font_scale - 0.2 if i > 0 else self.font_scale - 0.1
            cv2.putText(image, line, (x_start + 10, y_pos),
                       self.font_face, font_scale, self.colors['info'], 1)
        
        return image
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance tracking metrics."""
        self.performance_metrics['frame_count'] += 1
        self.performance_metrics['total_processing_time'] += processing_time
        
        # Calculate and store FPS
        if processing_time > 0:
            fps = 1.0 / processing_time
            self.performance_metrics['fps_history'].append(fps)
            
            # Keep only recent FPS values (memory constraint)
            if len(self.performance_metrics['fps_history']) > 30:
                self.performance_metrics['fps_history'] = self.performance_metrics['fps_history'][-30:]
    
    def _calculate_current_fps(self) -> float:
        """Calculate current FPS from recent measurements."""
        if not self.performance_metrics['fps_history']:
            return 0.0
        return self.performance_metrics['fps_history'][-1]
    
    def _update_detection_history(self, detection_results: Dict, regions: List[Dict]):
        """Update detection history for visualization."""
        if detection_results.get('pest_detected', False):
            entry = {
                'timestamp': time.time(),
                'pest_type': detection_results.get('class', 'unknown'),
                'confidence': detection_results.get('confidence', 0.0),
                'severity': detection_results.get('severity', 'low'),
                'region_count': len(regions)
            }
            
            self.detection_history.append(entry)
            self.performance_metrics['detection_count'] += 1
            
            # Maintain history size (memory constraint)
            if len(self.detection_history) > self.config['max_history_items']:
                self.detection_history = self.detection_history[-self.config['max_history_items']:]
    
    def _save_annotated_frame(self, annotated_image: np.ndarray, detection_results: Dict):
        """Save annotated frame if configured."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            pest_status = "pest" if detection_results.get('pest_detected') else "healthy"
            filename = f"annotated_{pest_status}_{timestamp}.jpg"
            
            output_path = "/Users/navyamudgal/Works/ACAD/Pest-Detection/output/annotated_frames"
            os.makedirs(output_path, exist_ok=True)
            
            filepath = os.path.join(output_path, filename)
            cv2.imwrite(filepath, annotated_image)
            
            # Also save metadata
            metadata = {
                'timestamp': timestamp,
                'detection_results': detection_results,
                'performance_metrics': self.performance_metrics.copy()
            }
            
            metadata_path = filepath.replace('.jpg', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"Error saving annotated frame: {e}")
    
    def get_visualization_statistics(self) -> Dict:
        """Get comprehensive visualization statistics."""
        return {
            'performance_metrics': self.performance_metrics.copy(),
            'detection_history_count': len(self.detection_history),
            'configuration': self.config.copy(),
            'recent_detections': self.detection_history[-10:] if self.detection_history else []
        }
    
    def reset_statistics(self):
        """Reset all statistics and history."""
        self.detection_history.clear()
        self.performance_metrics = {
            'frame_count': 0,
            'total_processing_time': 0.0,
            'fps_history': [],
            'detection_count': 0
        }
    
    def export_detection_report(self, output_path: str) -> bool:
        """Export comprehensive detection report."""
        try:
            report_data = {
                'export_timestamp': datetime.now().isoformat(),
                'statistics': self.get_visualization_statistics(),
                'detection_summary': {
                    'total_detections': self.performance_metrics['detection_count'],
                    'total_frames': self.performance_metrics['frame_count'],
                    'detection_rate': (self.performance_metrics['detection_count'] / 
                                     max(self.performance_metrics['frame_count'], 1)),
                    'average_fps': np.mean(self.performance_metrics['fps_history']) 
                                  if self.performance_metrics['fps_history'] else 0
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error exporting detection report: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Create test visualizer
    visualizer = DetectionVisualizer()
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create test detection results
    test_results = {
        'pest_detected': True,
        'class': 'aphid',
        'confidence': 0.85,
        'severity': 'medium'
    }
    
    # Create visualization
    annotated = visualizer.create_comprehensive_visualization(test_image, test_results, 0.033)
    
    # Get statistics
    stats = visualizer.get_visualization_statistics()
    print(f"Visualization created. Frame count: {stats['performance_metrics']['frame_count']}")
    
    # Export report
    report_path = "/Users/navyamudgal/Works/ACAD/Pest-Detection/output/reports/test_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    success = visualizer.export_detection_report(report_path)
    print(f"Report exported: {success}")