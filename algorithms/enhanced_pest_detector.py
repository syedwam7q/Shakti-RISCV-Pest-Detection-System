"""
Enhanced Pest Detection System
Combines rule-based and ML approaches for optimal real-world performance
"""

import numpy as np
import cv2
import os
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from .image_processor import ImageProcessor
from .ml_classifier import MLClassifier
from .pest_detector import SimpleRuleBasedClassifier


class EnhancedPestDetector:
    """
    Enhanced pest detector that combines multiple detection approaches:
    1. Machine Learning classifier (when trained on real data)
    2. Rule-based classifier (fallback and baseline)
    3. Ensemble methods for improved accuracy
    """
    
    def __init__(self, ml_model_path: Optional[str] = None, use_ensemble: bool = True):
        """
        Initialize enhanced pest detector.
        
        Args:
            ml_model_path: Path to trained ML model (optional)
            use_ensemble: Whether to use ensemble of multiple classifiers
        """
        self.image_processor = ImageProcessor()
        self.logger = logging.getLogger(__name__)
        
        # Initialize classifiers
        self.ml_classifier = None
        self.rule_classifier = SimpleRuleBasedClassifier()
        self.use_ensemble = use_ensemble
        
        # Enhanced class definitions with detailed characteristics
        self.class_definitions = {
            "healthy": {
                "description": "Healthy crop without pests or diseases",
                "symptoms": ["Normal green coloration", "No spots or lesions", "Uniform texture"],
                "treatment": ["Continue regular monitoring", "Maintain good plant hygiene"],
                "severity_levels": {"none": 0.0, "low": 0.0, "medium": 0.0, "high": 0.0}
            },
            "aphid": {
                "description": "Small soft-bodied insects that feed on plant sap",
                "symptoms": ["Small dark/green clusters", "Leaf curling", "Sticky honeydew", "Yellowing leaves"],
                "treatment": [
                    "Apply neem oil spray",
                    "Introduce beneficial insects (ladybugs, lacewings)",
                    "Remove heavily infested leaves",
                    "Use insecticidal soap",
                    "Improve air circulation"
                ],
                "severity_levels": {"low": 0.6, "medium": 0.75, "high": 0.9}
            },
            "whitefly": {
                "description": "Small white flying insects that damage plants",
                "symptoms": ["Small white flying insects", "Yellowing leaves", "Sticky honeydew", "Sooty mold"],
                "treatment": [
                    "Use yellow sticky traps",
                    "Apply insecticidal soap or oil",
                    "Remove heavily infested leaves",
                    "Use reflective mulches",
                    "Introduce parasitic wasps"
                ],
                "severity_levels": {"low": 0.65, "medium": 0.8, "high": 0.95}
            },
            "leaf_spot": {
                "description": "Fungal disease causing circular spots on leaves",
                "symptoms": ["Dark circular spots", "Yellow halos around spots", "Leaf yellowing", "Defoliation"],
                "treatment": [
                    "Remove and destroy infected leaves",
                    "Improve air circulation",
                    "Apply copper-based fungicide",
                    "Water at soil level to avoid leaf wetness",
                    "Avoid overhead watering"
                ],
                "severity_levels": {"low": 0.7, "medium": 0.85, "high": 0.95}
            },
            "powdery_mildew": {
                "description": "Fungal disease with white powdery coating",
                "symptoms": ["White powdery coating", "Leaf distortion", "Yellowing", "Stunted growth"],
                "treatment": [
                    "Apply baking soda solution (1 tsp per quart water)",
                    "Improve air circulation and sunlight exposure",
                    "Remove infected plant parts",
                    "Apply neem oil or sulfur-based fungicide",
                    "Avoid overhead watering"
                ],
                "severity_levels": {"low": 0.65, "medium": 0.8, "high": 0.9}
            },
            "spider_mites": {
                "description": "Tiny spider-like pests that cause stippled damage",
                "symptoms": ["Fine webbing", "Stippled/speckled leaves", "Bronze/yellow coloration", "Tiny moving dots"],
                "treatment": [
                    "Increase humidity around plants",
                    "Apply miticide or insecticidal soap",
                    "Remove heavily infested leaves",
                    "Introduce predatory mites",
                    "Regular water spraying"
                ],
                "severity_levels": {"low": 0.7, "medium": 0.85, "high": 0.95}
            }
        }
        
        self.confidence_threshold = 0.6
        self.severity_thresholds = {"low": 0.6, "medium": 0.75, "high": 0.9}
        
        # Load ML model if available
        if ml_model_path and os.path.exists(ml_model_path):
            self._load_ml_model(ml_model_path)
    
    def _load_ml_model(self, model_path: str):
        """Load trained ML model."""
        try:
            self.ml_classifier = MLClassifier()
            self.ml_classifier.load_model(model_path)
            print(f"✅ ML classifier loaded from: {model_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load ML model: {e}")
            self.ml_classifier = None
    
    def detect_pests(self, image: np.ndarray, method: str = "auto") -> Dict[str, any]:
        """
        Enhanced pest detection with multiple methods.
        
        Args:
            image: Input image (BGR format from camera)
            method: "auto", "ml", "rule", "ensemble"
            
        Returns:
            Enhanced detection results dictionary
        """
        try:
            # Preprocess image
            processed_image = self.image_processor.preprocess_image(image)
            
            # Choose detection method
            if method == "auto":
                # Use ML if available, otherwise rule-based
                if self.ml_classifier and self.ml_classifier.is_trained:
                    method = "ml"
                else:
                    method = "rule"
            
            # Perform detection based on method
            if method == "ml" and self.ml_classifier:
                results = self._detect_with_ml(processed_image)
            elif method == "ensemble" and self.ml_classifier:
                results = self._detect_with_ensemble(processed_image)
            else:
                results = self._detect_with_rules(processed_image)
            
            # Enhance results with additional analysis
            enhanced_results = self._enhance_detection_results(results, processed_image, image)
            
            return enhanced_results
            
        except Exception as e:
            return self._create_error_result(str(e))
    
    def _detect_with_ml(self, processed_image: np.ndarray) -> Dict[str, any]:
        """Detect pests using ML classifier."""
        # Use ML classifier
        prediction, confidence = self.ml_classifier.predict(processed_image)
        
        # Validate prediction
        if prediction not in self.class_definitions:
            # Fallback to rule-based if ML gives unknown class
            return self._detect_with_rules(processed_image)
        
        is_pest_detected = (prediction != "healthy" and confidence > self.confidence_threshold)
        
        return {
            "method": "machine_learning",
            "pest_detected": is_pest_detected,
            "class": prediction,
            "confidence": float(confidence),
            "model_type": getattr(self.ml_classifier, 'model_type', 'unknown')
        }
    
    def _detect_with_rules(self, processed_image: np.ndarray) -> Dict[str, any]:
        """Detect pests using rule-based classifier."""
        # Extract features
        features = self.image_processor.extract_features(processed_image)
        
        # Use rule-based classifier
        prediction, confidence = self.rule_classifier.predict(features)
        
        is_pest_detected = (prediction != "healthy" and confidence > self.confidence_threshold)
        
        return {
            "method": "rule_based",
            "pest_detected": is_pest_detected,
            "class": prediction,
            "confidence": float(confidence)
        }
    
    def _detect_with_ensemble(self, processed_image: np.ndarray) -> Dict[str, any]:
        """Detect pests using ensemble of ML and rule-based methods."""
        # Get predictions from both methods
        ml_result = self._detect_with_ml(processed_image)
        rule_result = self._detect_with_rules(processed_image)
        
        # Ensemble logic: weighted average with higher weight for ML if confident
        ml_confidence = ml_result.get("confidence", 0.0)
        rule_confidence = rule_result.get("confidence", 0.0)
        
        if ml_confidence > 0.8:
            # High ML confidence - trust ML more
            weight_ml = 0.7
            weight_rule = 0.3
        elif ml_confidence > 0.6:
            # Medium ML confidence - balanced
            weight_ml = 0.5
            weight_rule = 0.5
        else:
            # Low ML confidence - trust rules more
            weight_ml = 0.3
            weight_rule = 0.7
        
        # Choose final prediction based on highest weighted confidence
        if ml_result["class"] == rule_result["class"]:
            # Both agree - use weighted confidence
            final_class = ml_result["class"]
            final_confidence = weight_ml * ml_confidence + weight_rule * rule_confidence
        else:
            # Disagreement - choose higher confidence
            if ml_confidence > rule_confidence:
                final_class = ml_result["class"]
                final_confidence = ml_confidence
            else:
                final_class = rule_result["class"]
                final_confidence = rule_confidence
        
        is_pest_detected = (final_class != "healthy" and final_confidence > self.confidence_threshold)
        
        return {
            "method": "ensemble",
            "pest_detected": is_pest_detected,
            "class": final_class,
            "confidence": float(final_confidence),
            "ml_prediction": ml_result,
            "rule_prediction": rule_result
        }
    
    def _enhance_detection_results(self, base_results: Dict, processed_image: np.ndarray, original_image: np.ndarray) -> Dict[str, any]:
        """Enhance detection results with additional analysis."""
        
        # Get additional image analysis
        regions = self.image_processor.segment_regions(processed_image)
        
        # Calculate enhanced severity
        severity = self._calculate_enhanced_severity(
            base_results["confidence"], 
            len(regions), 
            base_results["class"]
        )
        
        # Get detailed recommendations
        recommendations = self._get_enhanced_recommendations(
            base_results["class"], 
            severity, 
            len(regions)
        )
        
        # Calculate risk assessment
        risk_level = self._assess_risk_level(base_results["class"], base_results["confidence"], severity)
        
        # Enhanced results
        enhanced_results = {
            **base_results,
            "affected_regions": len(regions),
            "severity": severity,
            "risk_level": risk_level,
            "recommendations": recommendations,
            "class_info": self.class_definitions.get(base_results["class"], {}),
            "detection_metadata": {
                "image_size": processed_image.shape,
                "total_regions_detected": len(regions),
                "processing_method": base_results.get("method", "unknown")
            }
        }
        
        # Add confidence breakdown if ensemble method
        if base_results.get("method") == "ensemble":
            enhanced_results["confidence_breakdown"] = {
                "ml_confidence": base_results.get("ml_prediction", {}).get("confidence", 0),
                "rule_confidence": base_results.get("rule_prediction", {}).get("confidence", 0),
                "ensemble_confidence": base_results["confidence"]
            }
        
        return enhanced_results
    
    def _calculate_enhanced_severity(self, confidence: float, num_regions: int, pest_class: str) -> str:
        """Calculate enhanced severity based on pest type and characteristics."""
        if confidence < self.confidence_threshold or pest_class == "healthy":
            return "none"
        
        # Get pest-specific severity thresholds
        pest_info = self.class_definitions.get(pest_class, {})
        severity_levels = pest_info.get("severity_levels", self.severity_thresholds)
        
        # Factor in number of affected regions
        region_factor = min(num_regions / 10.0, 1.0)  # Normalize to 0-1
        adjusted_confidence = confidence + (region_factor * 0.1)  # Slight boost for more regions
        
        if adjusted_confidence >= severity_levels.get("high", 0.9):
            return "high"
        elif adjusted_confidence >= severity_levels.get("medium", 0.75):
            return "medium"
        elif adjusted_confidence >= severity_levels.get("low", 0.6):
            return "low"
        else:
            return "none"
    
    def _get_enhanced_recommendations(self, pest_class: str, severity: str, num_regions: int) -> List[str]:
        """Get enhanced treatment recommendations."""
        recommendations = []
        
        # Base recommendations from class definition
        pest_info = self.class_definitions.get(pest_class, {})
        base_treatments = pest_info.get("treatment", ["Consult agricultural expert"])
        
        if pest_class == "healthy":
            return base_treatments
        
        # Add severity-specific recommendations
        if severity == "high":
            recommendations.append("🚨 URGENT: Immediate treatment required")
            recommendations.extend(base_treatments)
            recommendations.append("Consider consulting agricultural specialist")
            recommendations.append("Monitor daily for treatment effectiveness")
        elif severity == "medium":
            recommendations.append("⚠️ MODERATE: Treatment recommended within 24-48 hours")
            recommendations.extend(base_treatments)
            recommendations.append("Monitor every 2-3 days")
        elif severity == "low":
            recommendations.append("ℹ️ MILD: Preventive treatment recommended")
            recommendations.extend(base_treatments[:2])  # Most important treatments
            recommendations.append("Weekly monitoring sufficient")
        
        # Add region-specific recommendations
        if num_regions > 5:
            recommendations.append(f"Multiple affected areas detected ({num_regions} regions)")
            recommendations.append("Consider systemic treatment approach")
        
        return recommendations
    
    def _assess_risk_level(self, pest_class: str, confidence: float, severity: str) -> str:
        """Assess overall risk level to crops."""
        if pest_class == "healthy":
            return "minimal"
        
        # Risk assessment based on pest type and severity
        high_risk_pests = ["leaf_spot", "powdery_mildew"]  # Diseases spread quickly
        medium_risk_pests = ["aphid", "spider_mites"]      # Can multiply rapidly
        
        if pest_class in high_risk_pests and severity in ["medium", "high"]:
            return "critical"
        elif pest_class in high_risk_pests or severity == "high":
            return "high"
        elif severity == "medium":
            return "moderate"
        else:
            return "low"
    
    def _create_error_result(self, error_message: str) -> Dict[str, any]:
        """Create error result dictionary."""
        return {
            "pest_detected": False,
            "class": "error",
            "confidence": 0.0,
            "severity": "none",
            "risk_level": "unknown",
            "error": error_message,
            "recommendations": ["Check image quality and try again"],
            "method": "error"
        }
    
    def train_ml_model(self, dataset_dir: str, model_type: str = "random_forest") -> Dict:
        """
        Train ML classifier on real dataset.
        
        Args:
            dataset_dir: Path to training dataset
            model_type: Type of ML model to train
            
        Returns:
            Training results
        """
        print(f"🎓 Training ML model on real dataset...")
        
        # Initialize ML classifier
        self.ml_classifier = MLClassifier(model_type=model_type)
        
        # Load dataset and train
        X, y = self.ml_classifier.load_dataset(dataset_dir)
        results = self.ml_classifier.train(X, y)
        
        return results
    
    def save_trained_model(self, model_path: str):
        """Save trained ML model."""
        if self.ml_classifier and self.ml_classifier.is_trained:
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            self.ml_classifier.save_model(model_path)
            print(f"💾 Enhanced model saved to: {model_path}")
        else:
            raise RuntimeError("No trained ML model to save")
    
    def get_supported_classes(self) -> List[str]:
        """Get list of supported pest/disease classes."""
        return list(self.class_definitions.keys())
    
    def get_class_info(self, class_name: str) -> Dict:
        """Get detailed information about a specific class."""
        return self.class_definitions.get(class_name, {})


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Test enhanced detector
    detector = EnhancedPestDetector()
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test different detection methods
    methods = ["rule", "ml", "ensemble"] if detector.ml_classifier else ["rule"]
    
    for method in methods:
        print(f"\n🔍 Testing {method} detection:")
        results = detector.detect_pests(test_image, method=method)
        
        print(f"  Method: {results.get('method', 'unknown')}")
        print(f"  Pest detected: {results['pest_detected']}")
        print(f"  Class: {results['class']}")
        print(f"  Confidence: {results['confidence']:.3f}")
        print(f"  Severity: {results['severity']}")
        print(f"  Risk level: {results['risk_level']}")
    
    # Show supported classes
    print(f"\n📋 Supported classes: {detector.get_supported_classes()}")
    
    # Train on dataset if provided
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
        if os.path.exists(dataset_dir):
            print(f"\n🎓 Training on dataset: {dataset_dir}")
            training_results = detector.train_ml_model(dataset_dir)
            
            # Save model
            model_path = "models/enhanced_pest_detector.pkl"
            detector.save_trained_model(model_path)
        else:
            print(f"❌ Dataset directory not found: {dataset_dir}")