"""
Pest Detection Algorithm
Lightweight classifier designed for RISC-V embedded systems
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
import os
from .image_processor import ImageProcessor

class PestDetector:
    """
    Main pest detection classifier.
    Uses traditional machine learning for efficiency on embedded systems.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize pest detector.
        
        Args:
            model_path: Path to pre-trained model file
        """
        self.image_processor = ImageProcessor()
        self.model = None
        self.classes = ["healthy", "aphid", "whitefly", "leaf_spot", "powdery_mildew"]
        self.confidence_threshold = 0.6
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.initialize_simple_classifier()
    
    def initialize_simple_classifier(self):
        """
        Initialize a simple rule-based classifier for testing.
        This will be replaced with a trained model.
        """
        self.model = SimpleRuleBasedClassifier()
    
    def detect_pests(self, image: np.ndarray) -> Dict[str, any]:
        """
        Main pest detection function.
        
        Args:
            image: Input image (BGR format from camera)
            
        Returns:
            Detection results dictionary
        """
        try:
            # Preprocess image
            processed_image = self.image_processor.preprocess_image(image)
            
            # Extract features
            features = self.image_processor.extract_features(processed_image)
            
            # Classify
            prediction, confidence = self.model.predict(features)
            
            # Determine if pest/disease detected
            is_pest_detected = (prediction != "healthy" and 
                              confidence > self.confidence_threshold)
            
            # Get detailed analysis
            regions = self.image_processor.segment_regions(processed_image)
            
            results = {
                "pest_detected": is_pest_detected,
                "class": prediction,
                "confidence": confidence,
                "affected_regions": len(regions),
                "severity": self.calculate_severity(confidence, len(regions)),
                "recommendations": self.get_recommendations(prediction)
            }
            
            return results
            
        except Exception as e:
            return {
                "error": f"Detection failed: {str(e)}",
                "pest_detected": False,
                "class": "error",
                "confidence": 0.0
            }
    
    def calculate_severity(self, confidence: float, num_regions: int) -> str:
        """
        Calculate pest infestation severity.
        
        Args:
            confidence: Classification confidence
            num_regions: Number of affected regions
            
        Returns:
            Severity level as string
        """
        if confidence < 0.6 or num_regions == 0:
            return "none"
        elif confidence < 0.75 or num_regions < 3:
            return "low"
        elif confidence < 0.9 or num_regions < 6:
            return "medium"
        else:
            return "high"
    
    def get_recommendations(self, pest_class: str) -> List[str]:
        """
        Get treatment recommendations based on detected pest/disease.
        
        Args:
            pest_class: Detected pest or disease class
            
        Returns:
            List of recommendations
        """
        recommendations = {
            "healthy": ["Continue regular monitoring", "Maintain good plant hygiene"],
            "aphid": [
                "Apply neem oil spray",
                "Introduce ladybugs or lacewings",
                "Remove heavily infested leaves",
                "Increase plant spacing for better air circulation"
            ],
            "whitefly": [
                "Use yellow sticky traps",
                "Apply insecticidal soap",
                "Remove affected leaves",
                "Consider reflective mulches"
            ],
            "leaf_spot": [
                "Remove and destroy infected leaves",
                "Improve air circulation",
                "Apply copper-based fungicide",
                "Water at soil level to avoid leaf wetness"
            ],
            "powdery_mildew": [
                "Apply baking soda solution",
                "Improve air circulation",
                "Reduce nitrogen fertilization",
                "Remove infected plant parts"
            ]
        }
        
        return recommendations.get(pest_class, ["Consult agricultural expert"])
    
    def load_model(self, model_path: str):
        """
        Load pre-trained model from file.
        
        Args:
            model_path: Path to model file
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.initialize_simple_classifier()
    
    def save_model(self, model_path: str):
        """
        Save trained model to file.
        
        Args:
            model_path: Path to save model
        """
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")


class SimpleRuleBasedClassifier:
    """
    Simple rule-based classifier for initial testing.
    Will be replaced with trained machine learning model.
    """
    
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> Dict:
        """Initialize classification rules based on feature thresholds."""
        return {
            "edge_density_threshold": 0.1,
            "color_variance_threshold": 1000,
            "texture_energy_threshold": 0.5
        }
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Make prediction based on simple rules.
        
        Args:
            features: Feature vector from image
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if len(features) < 6:
            return "error", 0.0
        
        edge_density = features[0]
        color_mean = features[1:4] if len(features) > 3 else features[1:2]
        texture_energy = features[-1] if len(features) > 6 else 0.0
        
        # Simple rule-based classification
        if edge_density > self.rules["edge_density_threshold"]:
            if np.mean(color_mean) < 0.3:  # Dark regions might indicate disease
                return "leaf_spot", 0.7
            elif texture_energy > self.rules["texture_energy_threshold"]:
                return "powdery_mildew", 0.65
            else:
                return "aphid", 0.6
        elif np.var(color_mean) > 0.1:  # High color variance
            return "whitefly", 0.65
        else:
            return "healthy", 0.8
    
    def train(self, features_list: List[np.ndarray], labels: List[str]):
        """
        Simple training method (adjust thresholds based on data).
        
        Args:
            features_list: List of feature vectors
            labels: Corresponding labels
        """
        # This is a placeholder for actual training
        # In practice, you would analyze the features and adjust rules
        print("Training simple classifier...")
        
        # Calculate optimal thresholds based on training data
        healthy_features = [f for f, l in zip(features_list, labels) if l == "healthy"]
        pest_features = [f for f, l in zip(features_list, labels) if l != "healthy"]
        
        if healthy_features and pest_features:
            healthy_edge_mean = np.mean([f[0] for f in healthy_features])
            pest_edge_mean = np.mean([f[0] for f in pest_features])
            
            # Adjust threshold to be between means
            self.rules["edge_density_threshold"] = (healthy_edge_mean + pest_edge_mean) / 2
        
        print("Training completed!")


# Example usage
if __name__ == "__main__":
    # Test the pest detector
    detector = PestDetector()
    
    # Create a test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Detect pests
    results = detector.detect_pests(test_image)
    
    print("Detection Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    if results["pest_detected"]:
        print(f"\nAlert: {results['class']} detected with {results['confidence']:.2f} confidence!")
        print(f"Severity: {results['severity']}")
        print("Recommendations:")
        for rec in results["recommendations"]:
            print(f"  - {rec}")