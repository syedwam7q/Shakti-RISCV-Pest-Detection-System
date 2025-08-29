"""
Machine Learning Classifier for Real Pest Detection
Uses Random Forest and SVM for real-world pest/disease classification
"""

import numpy as np
import cv2
import pickle
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import json
import logging

try:
    from .image_processor import ImageProcessor
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from algorithms.image_processor import ImageProcessor


class MLClassifier:
    """
    Machine Learning based pest classifier for real-world data.
    Supports Random Forest and SVM algorithms optimized for embedded systems.
    """
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize ML classifier.
        
        Args:
            model_type: "random_forest" or "svm"
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.classes = []
        self.is_trained = False
        self.feature_extractor = ImageProcessor()
        self.logger = logging.getLogger(__name__)
        
        # Model hyperparameters optimized for embedded systems
        self.model_params = {
            "random_forest": {
                "n_estimators": 50,  # Reduced for faster inference
                "max_depth": 10,     # Limited depth for memory efficiency
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42
            },
            "svm": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "probability": True,  # Enable probability estimates
                "random_state": 42
            }
        }
    
    def extract_advanced_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract advanced features optimized for pest detection.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Feature vector for classification
        """
        features = []
        
        # Convert to uint8 for OpenCV operations
        if image.dtype == np.float32:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image
        
        # 1. Basic statistical features
        if len(img_uint8.shape) == 3:
            # Color image features
            for channel in range(3):
                channel_data = img_uint8[:, :, channel]
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.median(channel_data),
                    np.percentile(channel_data, 25),
                    np.percentile(channel_data, 75)
                ])
        else:
            # Grayscale features
            features.extend([
                np.mean(img_uint8),
                np.std(img_uint8),
                np.median(img_uint8),
                np.percentile(img_uint8, 25),
                np.percentile(img_uint8, 75)
            ])
        
        # 2. Edge and texture features
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY) if len(img_uint8.shape) == 3 else img_uint8
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        # Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude),
            np.max(gradient_magnitude)
        ])
        
        # 3. Contour-based features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Number of contours (potential pest regions)
            features.append(len(contours))
            
            # Largest contour features
            largest_contour = max(contours, key=cv2.contourArea)
            features.extend([
                cv2.contourArea(largest_contour),
                cv2.arcLength(largest_contour, True),
                len(largest_contour)
            ])
            
            # Average contour size
            avg_contour_area = np.mean([cv2.contourArea(c) for c in contours])
            features.append(avg_contour_area)
        else:
            # No contours found
            features.extend([0, 0, 0, 0, 0])
        
        # 4. Color distribution features (if color image)
        if len(img_uint8.shape) == 3:
            # HSV color space analysis
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
            
            # Dominant colors (simplified)
            for i in range(3):
                hist = cv2.calcHist([hsv], [i], None, [256], [0, 256])
                features.extend([
                    np.argmax(hist),  # Dominant value
                    np.sum(hist > np.max(hist) * 0.1)  # Number of significant values
                ])
        
        # 5. Local Binary Pattern (simplified)
        # This helps detect texture patterns typical of pests/diseases
        lbp_features = self._calculate_lbp_features(gray)
        features.extend(lbp_features)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_lbp_features(self, gray_image: np.ndarray) -> List[float]:
        """
        Calculate simplified Local Binary Pattern features.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            LBP-based texture features
        """
        # Simplified 3x3 LBP calculation
        h, w = gray_image.shape
        lbp_values = []
        
        # Sample points for LBP calculation
        sample_points = [(i, j) for i in range(1, h-1, 10) for j in range(1, w-1, 10)]
        
        for y, x in sample_points[:100]:  # Limit samples for efficiency
            center = gray_image[y, x]
            pattern = 0
            
            # 8-neighbor LBP
            neighbors = [
                (y-1, x-1), (y-1, x), (y-1, x+1),
                (y, x+1), (y+1, x+1), (y+1, x),
                (y+1, x-1), (y, x-1)
            ]
            
            for i, (ny, nx) in enumerate(neighbors):
                if gray_image[ny, nx] >= center:
                    pattern |= (1 << i)
            
            lbp_values.append(pattern)
        
        # Statistical features from LBP values
        if lbp_values:
            return [
                np.mean(lbp_values),
                np.std(lbp_values),
                np.max(lbp_values),
                np.min(lbp_values),
                len(set(lbp_values))  # Number of unique patterns
            ]
        else:
            return [0, 0, 0, 0, 0]
    
    def load_dataset(self, dataset_dir: str) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load images and labels from dataset directory.
        
        Args:
            dataset_dir: Path to dataset directory with class subdirectories
            
        Returns:
            Tuple of (feature_vectors, labels)
        """
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        features = []
        labels = []
        image_paths = []
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Load images from each class directory
        for class_dir in dataset_path.iterdir():
            if not class_dir.is_dir() or class_dir.name.startswith('.'):
                continue
            
            class_name = class_dir.name
            self.logger.info(f"Loading class: {class_name}")
            
            class_images = 0
            for image_file in class_dir.iterdir():
                if image_file.suffix.lower() in image_extensions:
                    try:
                        # Load and process image
                        image = cv2.imread(str(image_file))
                        if image is not None:
                            # Preprocess image
                            processed_image = self.feature_extractor.preprocess_image(image)
                            
                            # Extract features
                            feature_vector = self.extract_advanced_features(processed_image)
                            
                            features.append(feature_vector)
                            labels.append(class_name)
                            image_paths.append(str(image_file))
                            class_images += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing {image_file}: {e}")
            
            print(f"  📁 {class_name}: {class_images} images loaded")
        
        if not features:
            raise ValueError("No images found in dataset directory")
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Update class list
        self.classes = sorted(list(set(labels)))
        
        print(f"\n📊 Dataset loaded:")
        print(f"   Total images: {len(features)}")
        print(f"   Classes: {len(self.classes)} - {self.classes}")
        print(f"   Feature dimensions: {X.shape[1]}")
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict:
        """
        Train the classifier on feature data.
        
        Args:
            X: Feature vectors
            y: Labels
            test_size: Fraction of data to use for testing
            
        Returns:
            Training results dictionary
        """
        print(f"\n🤖 Training {self.model_type} classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(**self.model_params["random_forest"])
        elif self.model_type == "svm":
            self.model = SVC(**self.model_params["svm"])
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Train model
        print("   Training in progress...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        print(f"   ✅ Training completed!")
        print(f"   📊 Training accuracy: {train_accuracy:.3f}")
        print(f"   📊 Test accuracy: {test_accuracy:.3f}")
        
        # Generate detailed report
        report = classification_report(y_test, test_pred, output_dict=True)
        
        self.is_trained = True
        
        results = {
            "model_type": self.model_type,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "classification_report": report,
            "classes": self.classes,
            "feature_dimensions": X.shape[1],
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        return results
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict pest class for a single image.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Preprocess image
        processed_image = self.feature_extractor.preprocess_image(image)
        
        # Extract features
        features = self.extract_advanced_features(processed_image)
        features_scaled = self.scaler.transform([features])
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        
        # Get confidence (probability)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
        else:
            # For models without probability estimates
            confidence = 0.8  # Default confidence
        
        return prediction, confidence
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Predict pest classes for multiple images.
        
        Args:
            images: List of input images
            
        Returns:
            List of (predicted_class, confidence) tuples
        """
        results = []
        for image in images:
            pred, conf = self.predict(image)
            results.append((pred, conf))
        return results
    
    def save_model(self, model_path: str):
        """
        Save trained model to file.
        
        Args:
            model_path: Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("No trained model to save")
        
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model components
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'classes': self.classes,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, model_path)
        print(f"💾 Model saved to: {model_path}")
    
    def load_model(self, model_path: str):
        """
        Load trained model from file.
        
        Args:
            model_path: Path to saved model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.classes = model_data['classes']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        print(f"📂 Model loaded from: {model_path}")
        print(f"   Model type: {self.model_type}")
        print(f"   Classes: {self.classes}")


# Example usage and training script
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ml_classifier.py <dataset_directory> [model_type]")
        print("Model types: random_forest (default), svm")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else "random_forest"
    
    # Initialize classifier
    classifier = MLClassifier(model_type=model_type)
    
    try:
        # Load dataset
        X, y = classifier.load_dataset(dataset_dir)
        
        # Train model
        results = classifier.train(X, y)
        
        # Print detailed results
        print("\n" + "="*50)
        print("🎯 TRAINING RESULTS")
        print("="*50)
        print(f"Model Type: {results['model_type']}")
        print(f"Training Accuracy: {results['train_accuracy']:.3f}")
        print(f"Test Accuracy: {results['test_accuracy']:.3f}")
        print(f"Feature Dimensions: {results['feature_dimensions']}")
        print()
        
        # Per-class results
        print("Per-Class Performance:")
        report = results['classification_report']
        for class_name in classifier.classes:
            if class_name in report:
                metrics = report[class_name]
                print(f"  {class_name}:")
                print(f"    Precision: {metrics['precision']:.3f}")
                print(f"    Recall: {metrics['recall']:.3f}")
                print(f"    F1-Score: {metrics['f1-score']:.3f}")
        
        # Save model
        model_path = f"models/{model_type}_pest_classifier.pkl"
        os.makedirs("models", exist_ok=True)
        classifier.save_model(model_path)
        
        print(f"\n✅ Training completed successfully!")
        print(f"💾 Model saved to: {model_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)