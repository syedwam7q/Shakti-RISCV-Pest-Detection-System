"""
Image Processing Module for Pest Detection
Optimized for RISC-V Shakti processor deployment
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging

class ImageProcessor:
    """
    Core image processing class for pest detection.
    Designed to be lightweight and efficient for embedded systems.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize image processor with target dimensions.
        
        Args:
            target_size: Target image size for processing (width, height)
        """
        self.target_size = target_size
        self.logger = logging.getLogger(__name__)
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for pest detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image ready for detection
        """
        try:
            # Resize to target size
            processed = cv2.resize(image, self.target_size)
            
            # Convert to RGB if needed (OpenCV uses BGR by default)
            if len(processed.shape) == 3 and processed.shape[2] == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values to [0, 1]
            processed = processed.astype(np.float32) / 255.0
            
            # Apply noise reduction
            processed = self.denoise_image(processed)
            
            # Enhance contrast
            processed = self.enhance_contrast(processed)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {e}")
            raise
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising filter to reduce image noise.
        Uses Gaussian blur for efficiency on embedded systems.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        # Convert back to uint8 for OpenCV operations
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Apply Gaussian blur for denoising
        denoised = cv2.GaussianBlur(img_uint8, (3, 3), 0)
        
        # Convert back to float32
        return denoised.astype(np.float32) / 255.0
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using histogram equalization.
        
        Args:
            image: Input image
            
        Returns:
            Contrast-enhanced image
        """
        # Convert to uint8 for OpenCV operations
        img_uint8 = (image * 255).astype(np.uint8)
        
        if len(img_uint8.shape) == 3:
            # For color images, apply CLAHE to LAB color space
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img_uint8)
        
        return enhanced.astype(np.float32) / 255.0
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from preprocessed image.
        Uses edge detection and color analysis for pest identification.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Feature vector for classification
        """
        features = []
        
        # Convert to uint8 for feature extraction
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Edge features using Canny edge detection
        edges = cv2.Canny(img_uint8, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        # Color features
        if len(img_uint8.shape) == 3:
            # Mean color values
            mean_colors = np.mean(img_uint8, axis=(0, 1))
            features.extend(mean_colors)
            
            # Color variance
            color_variance = np.var(img_uint8, axis=(0, 1))
            features.extend(color_variance)
        else:
            # Grayscale statistics
            features.extend([np.mean(img_uint8), np.var(img_uint8)])
        
        # Texture features using Local Binary Patterns (simplified)
        texture_features = self.extract_texture_features(img_uint8)
        features.extend(texture_features)
        
        return np.array(features)
    
    def extract_texture_features(self, image: np.ndarray) -> List[float]:
        """
        Extract texture features using simplified Local Binary Patterns.
        
        Args:
            image: Input image (uint8)
            
        Returns:
            List of texture features
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Simple texture analysis using gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Texture statistics
        texture_mean = np.mean(gradient_magnitude)
        texture_std = np.std(gradient_magnitude)
        texture_energy = np.sum(gradient_magnitude**2) / (gray.shape[0] * gray.shape[1])
        
        return [texture_mean, texture_std, texture_energy]
    
    def segment_regions(self, image: np.ndarray, threshold: float = 0.5) -> List[np.ndarray]:
        """
        Segment image into potential pest/disease regions.
        
        Args:
            image: Preprocessed image
            threshold: Segmentation threshold
            
        Returns:
            List of segmented regions
        """
        # Convert to uint8 for segmentation
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Convert to grayscale for thresholding
        if len(img_uint8.shape) == 3:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_uint8
        
        # Apply threshold
        _, binary = cv2.threshold(gray, int(threshold * 255), 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract regions
        regions = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small regions
                x, y, w, h = cv2.boundingRect(contour)
                region = img_uint8[y:y+h, x:x+w]
                regions.append(region)
        
        return regions

# Example usage and testing
if __name__ == "__main__":
    # Test the image processor
    processor = ImageProcessor()
    
    # Create a dummy image for testing
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Process the image
    processed = processor.preprocess_image(test_image)
    features = processor.extract_features(processed)
    regions = processor.segment_regions(processed)
    
    print(f"Processed image shape: {processed.shape}")
    print(f"Feature vector size: {len(features)}")
    print(f"Number of regions found: {len(regions)}")