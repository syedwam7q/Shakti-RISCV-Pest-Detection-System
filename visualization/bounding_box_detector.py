"""
Advanced Bounding Box Detection System
=====================================

Detects and visualizes pest regions with bounding boxes.
Optimized for Shakti RISC-V embedded systems.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import time


class BoundingBoxDetector:
    """
    Advanced bounding box detection for pest regions.
    Designed for efficient processing on Shakti E-class RISC-V.
    """
    
    def __init__(self):
        self.detection_history = []
        self.confidence_threshold = 0.5
        self.min_region_area = 100  # Minimum pest region size
        self.max_regions = 10       # Maximum regions to detect (memory constraint)
        
        # Color scheme for different pest types
        self.pest_colors = {
            'healthy': (0, 255, 0),      # Green
            'aphid': (0, 0, 255),        # Red
            'whitefly': (255, 255, 0),   # Yellow
            'leaf_spot': (128, 0, 128),  # Purple
            'powdery_mildew': (255, 165, 0),  # Orange
            'unknown': (128, 128, 128)   # Gray
        }
    
    def detect_pest_regions(self, image: np.ndarray, confidence_map: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Detect bounding boxes for pest regions in the image.
        
        Args:
            image: Input image (BGR format)
            confidence_map: Optional confidence map for region detection
            
        Returns:
            List of detected regions with bounding boxes and metadata
        """
        regions = []
        
        try:
            # Convert to grayscale for processing efficiency
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding for pest region detection
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Morphological operations to clean up regions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours and create bounding boxes
            for i, contour in enumerate(contours[:self.max_regions]):
                area = cv2.contourArea(contour)
                
                if area > self.min_region_area:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate confidence based on various factors
                    confidence = self._calculate_region_confidence(image, contour, confidence_map)
                    
                    if confidence >= self.confidence_threshold:
                        # Classify pest type (simplified for embedded system)
                        pest_type = self._classify_region(image[y:y+h, x:x+w])
                        
                        region_data = {
                            'bbox': (x, y, w, h),
                            'confidence': confidence,
                            'pest_type': pest_type,
                            'area': area,
                            'center': (x + w//2, y + h//2),
                            'contour': contour,
                            'severity': self._assess_severity(confidence, area)
                        }
                        
                        regions.append(region_data)
            
            # Sort by confidence (highest first)
            regions.sort(key=lambda r: r['confidence'], reverse=True)
            
            return regions
            
        except Exception as e:
            print(f"Error in pest region detection: {e}")
            return []
    
    def _calculate_region_confidence(self, image: np.ndarray, contour: np.ndarray, 
                                   confidence_map: Optional[np.ndarray] = None) -> float:
        """Calculate confidence score for a detected region."""
        try:
            # Basic confidence calculation based on contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                return 0.0
            
            # Compactness measure
            compactness = 4 * np.pi * area / (perimeter * perimeter)
            
            # Base confidence from geometric properties
            confidence = min(compactness * 0.8, 0.9)
            
            # If confidence map is provided, incorporate it
            if confidence_map is not None:
                x, y, w, h = cv2.boundingRect(contour)
                roi_confidence = np.mean(confidence_map[y:y+h, x:x+w])
                confidence = (confidence + roi_confidence) / 2
            
            return float(confidence)
            
        except Exception:
            return 0.5  # Default confidence
    
    def _classify_region(self, region_image: np.ndarray) -> str:
        """
        Simple pest classification for detected regions.
        Optimized for embedded systems - uses basic color/texture analysis.
        """
        try:
            if region_image.size == 0:
                return 'unknown'
            
            # Calculate basic color statistics
            mean_color = np.mean(region_image, axis=(0, 1))
            
            # Simple classification based on color characteristics
            # This is a simplified version - real implementation would use trained model
            b, g, r = mean_color
            
            if g > r and g > b:  # Greenish - likely healthy
                return 'healthy'
            elif r > g and r > b:  # Reddish - might be disease/pest
                if r > 150:
                    return 'leaf_spot'
                else:
                    return 'aphid'
            elif b > r and b > g:  # Bluish - unusual, might be artifact
                return 'unknown'
            elif r > 100 and g > 100:  # Yellowish
                return 'whitefly'
            else:
                return 'powdery_mildew'
                
        except Exception:
            return 'unknown'
    
    def _assess_severity(self, confidence: float, area: float) -> str:
        """Assess pest infestation severity."""
        severity_score = confidence * np.log(area + 1) / 10
        
        if severity_score < 0.3:
            return 'low'
        elif severity_score < 0.6:
            return 'medium'
        else:
            return 'high'
    
    def draw_bounding_boxes(self, image: np.ndarray, regions: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and annotations on the image.
        
        Args:
            image: Input image
            regions: List of detected regions with metadata
            
        Returns:
            Annotated image with bounding boxes
        """
        annotated_image = image.copy()
        
        for region in regions:
            bbox = region['bbox']
            x, y, w, h = bbox
            pest_type = region['pest_type']
            confidence = region['confidence']
            severity = region['severity']
            
            # Get color for pest type
            color = self.pest_colors.get(pest_type, self.pest_colors['unknown'])
            
            # Adjust color intensity based on severity
            if severity == 'high':
                thickness = 3
                font_scale = 0.8
            elif severity == 'medium':
                thickness = 2
                font_scale = 0.6
                color = tuple(int(c * 0.8) for c in color)  # Dim the color
            else:
                thickness = 1
                font_scale = 0.5
                color = tuple(int(c * 0.6) for c in color)  # More dim
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, thickness)
            
            # Prepare label text
            label = f"{pest_type}: {confidence:.1%}"
            if severity != 'low':
                label += f" ({severity})"
            
            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            # Draw text background
            cv2.rectangle(
                annotated_image,
                (x, y - text_height - baseline - 5),
                (x + text_width, y),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                annotated_image, label,
                (x, y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # White text
                1
            )
            
            # Draw center point
            center = region['center']
            cv2.circle(annotated_image, center, 3, color, -1)
        
        return annotated_image
    
    def create_confidence_heatmap(self, image: np.ndarray, regions: List[Dict]) -> np.ndarray:
        """
        Create a confidence heatmap overlay.
        
        Args:
            image: Input image
            regions: List of detected regions
            
        Returns:
            Heatmap visualization
        """
        height, width = image.shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        for region in regions:
            x, y, w, h = region['bbox']
            confidence = region['confidence']
            
            # Create confidence map for this region
            heatmap[y:y+h, x:x+w] = np.maximum(
                heatmap[y:y+h, x:x+w], 
                confidence
            )
        
        # Convert to color heatmap
        heatmap_normalized = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        
        # Blend with original image
        alpha = 0.4
        result = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return result
    
    def get_detection_summary(self, regions: List[Dict]) -> Dict:
        """
        Generate summary statistics for detected regions.
        
        Returns:
            Dictionary with detection statistics
        """
        if not regions:
            return {
                'total_regions': 0,
                'pest_types': {},
                'severity_distribution': {},
                'average_confidence': 0.0,
                'max_confidence': 0.0
            }
        
        pest_types = {}
        severity_distribution = {'low': 0, 'medium': 0, 'high': 0}
        confidences = []
        
        for region in regions:
            # Count pest types
            pest_type = region['pest_type']
            pest_types[pest_type] = pest_types.get(pest_type, 0) + 1
            
            # Count severity levels
            severity = region['severity']
            severity_distribution[severity] += 1
            
            # Collect confidences
            confidences.append(region['confidence'])
        
        return {
            'total_regions': len(regions),
            'pest_types': pest_types,
            'severity_distribution': severity_distribution,
            'average_confidence': np.mean(confidences),
            'max_confidence': np.max(confidences),
            'timestamp': time.time()
        }


# Example usage for testing
if __name__ == "__main__":
    detector = BoundingBoxDetector()
    
    # Create a test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Detect regions
    regions = detector.detect_pest_regions(test_image)
    
    # Draw bounding boxes
    annotated = detector.draw_bounding_boxes(test_image, regions)
    
    # Get summary
    summary = detector.get_detection_summary(regions)
    
    print(f"Detected {summary['total_regions']} regions")
    print(f"Pest types: {summary['pest_types']}")
    print(f"Average confidence: {summary['average_confidence']:.2%}")