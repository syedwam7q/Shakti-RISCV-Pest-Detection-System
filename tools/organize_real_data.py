#!/usr/bin/env python3
"""
Real Data Organizer
Helps organize downloaded pest/disease images into proper structure
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List
import argparse
import cv2


class RealDataOrganizer:
    """Organizes downloaded plant disease datasets into standardized structure."""
    
    def __init__(self):
        self.base_dir = Path("datasets/real")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Mapping from common dataset names to our standard classes
        self.class_mappings = {
            # Standard mappings
            "healthy": "healthy",
            "normal": "healthy",
            
            # Pest mappings
            "aphid": "aphid",
            "aphids": "aphid", 
            "green_aphid": "aphid",
            
            "whitefly": "whitefly",
            "whiteflies": "whitefly",
            "white_fly": "whitefly",
            
            "spider_mites": "spider_mites",
            "spider_mite": "spider_mites",
            "two_spotted_spider_mite": "spider_mites",
            
            # Disease mappings
            "leaf_spot": "leaf_spot",
            "bacterial_spot": "leaf_spot",
            "septoria_leaf_spot": "leaf_spot",
            "target_spot": "leaf_spot",
            
            "powdery_mildew": "powdery_mildew",
            "powdery": "powdery_mildew",
            
            "early_blight": "leaf_spot",  # Group with leaf spot
            "late_blight": "leaf_spot",   # Group with leaf spot
            "leaf_mold": "leaf_spot",     # Group with leaf spot
        }
        
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    def organize_plantvillage_dataset(self, source_dir: str):
        """
        Organize PlantVillage dataset structure.
        
        Args:
            source_dir: Path to downloaded PlantVillage dataset
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"❌ Source directory not found: {source_dir}")
            return
        
        print(f"📂 Organizing PlantVillage dataset from: {source_dir}")
        
        # Create organized directory
        organized_dir = self.base_dir / "plantvillage_organized"
        organized_dir.mkdir(exist_ok=True)
        
        # PlantVillage typically has structure: crop/disease/images
        stats = {"total_images": 0, "classes": {}}
        
        # Scan for images and organize by disease type
        for root, dirs, files in os.walk(source_path):
            root_path = Path(root)
            
            for file in files:
                if Path(file).suffix.lower() in self.supported_extensions:
                    file_path = root_path / file
                    
                    # Extract disease class from path
                    disease_class = self._extract_disease_class(str(file_path))
                    
                    if disease_class:
                        # Create class directory
                        class_dir = organized_dir / disease_class
                        class_dir.mkdir(exist_ok=True)
                        
                        # Copy image
                        dest_path = class_dir / file
                        if not dest_path.exists():
                            shutil.copy2(file_path, dest_path)
                            stats["total_images"] += 1
                            
                            if disease_class not in stats["classes"]:
                                stats["classes"][disease_class] = 0
                            stats["classes"][disease_class] += 1
        
        # Save organization statistics
        stats_path = organized_dir / "organization_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Organized {stats['total_images']} images into {len(stats['classes'])} classes")
        for class_name, count in stats["classes"].items():
            print(f"  📁 {class_name}: {count} images")
        
        return str(organized_dir)
    
    def organize_custom_dataset(self, source_dir: str, target_name: str = "custom"):
        """
        Organize custom dataset with flexible structure.
        
        Args:
            source_dir: Path to custom dataset
            target_name: Name for organized dataset
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"❌ Source directory not found: {source_dir}")
            return
        
        print(f"📂 Organizing custom dataset: {source_dir}")
        
        # Create organized directory
        organized_dir = self.base_dir / f"{target_name}_organized"
        organized_dir.mkdir(exist_ok=True)
        
        stats = {"total_images": 0, "classes": {}, "unmapped": []}
        
        # If source has subdirectories, treat each as a class
        if any(item.is_dir() for item in source_path.iterdir()):
            for class_dir in source_path.iterdir():
                if class_dir.is_dir():
                    class_name = self._map_class_name(class_dir.name.lower())
                    if class_name:
                        self._copy_class_images(class_dir, organized_dir / class_name, stats, class_name)
                    else:
                        stats["unmapped"].append(class_dir.name)
        else:
            # All images in one directory - need manual classification
            print("⚠️  All images in single directory - manual classification needed")
            unknown_dir = organized_dir / "unknown"
            unknown_dir.mkdir(exist_ok=True)
            
            for file in source_path.iterdir():
                if file.is_file() and file.suffix.lower() in self.supported_extensions:
                    shutil.copy2(file, unknown_dir / file.name)
                    stats["total_images"] += 1
            
            stats["classes"]["unknown"] = stats["total_images"]
        
        # Save statistics
        stats_path = organized_dir / "organization_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Organized {stats['total_images']} images")
        for class_name, count in stats["classes"].items():
            print(f"  📁 {class_name}: {count} images")
        
        if stats["unmapped"]:
            print(f"⚠️  Unmapped classes: {stats['unmapped']}")
            print("   Please manually organize these or add mappings")
        
        return str(organized_dir)
    
    def _extract_disease_class(self, file_path: str) -> str:
        """Extract disease class from file path."""
        path_lower = file_path.lower()
        
        # Check path components for disease keywords
        path_parts = Path(file_path).parts
        
        for part in reversed(path_parts):  # Check from most specific to general
            part_lower = part.lower()
            
            # Direct mapping
            if part_lower in self.class_mappings:
                return self.class_mappings[part_lower]
            
            # Substring matching
            for keyword, mapped_class in self.class_mappings.items():
                if keyword in part_lower:
                    return mapped_class
        
        return None
    
    def _map_class_name(self, class_name: str) -> str:
        """Map class name to our standard classes."""
        class_name = class_name.lower().strip()
        
        # Direct mapping
        if class_name in self.class_mappings:
            return self.class_mappings[class_name]
        
        # Substring matching
        for keyword, mapped_class in self.class_mappings.items():
            if keyword in class_name:
                return mapped_class
        
        # If no mapping found, return original (user can manually fix)
        return class_name if class_name else None
    
    def _copy_class_images(self, source_dir: Path, target_dir: Path, stats: Dict, class_name: str):
        """Copy images from source to target directory."""
        target_dir.mkdir(exist_ok=True)
        
        count = 0
        for file_path in source_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                # Verify it's a valid image
                if self._is_valid_image(file_path):
                    dest_path = target_dir / file_path.name
                    if not dest_path.exists():
                        shutil.copy2(file_path, dest_path)
                        count += 1
        
        stats["total_images"] += count
        stats["classes"][class_name] = count
    
    def _is_valid_image(self, file_path: Path) -> bool:
        """Check if file is a valid image."""
        try:
            image = cv2.imread(str(file_path))
            return image is not None
        except:
            return False
    
    def create_train_test_split(self, dataset_dir: str, train_ratio: float = 0.8):
        """
        Create train/test split for a dataset.
        
        Args:
            dataset_dir: Path to organized dataset
            train_ratio: Ratio of images for training (default: 0.8)
        """
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            print(f"❌ Dataset directory not found: {dataset_dir}")
            return
        
        print(f"✂️  Creating train/test split ({train_ratio:.1%} train, {1-train_ratio:.1%} test)")
        
        # Create split directories
        train_dir = dataset_path.parent / f"{dataset_path.name}_train"
        test_dir = dataset_path.parent / f"{dataset_path.name}_test"
        
        train_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        split_stats = {"train": {}, "test": {}}
        
        # Process each class
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir() and not class_dir.name.startswith('.'):
                class_name = class_dir.name
                
                # Get all images in class
                images = list(class_dir.glob("*"))
                images = [img for img in images if img.suffix.lower() in self.supported_extensions]
                
                if not images:
                    continue
                
                # Calculate split
                num_train = int(len(images) * train_ratio)
                
                # Shuffle for random split
                import random
                random.shuffle(images)
                
                train_images = images[:num_train]
                test_images = images[num_train:]
                
                # Create class directories
                train_class_dir = train_dir / class_name
                test_class_dir = test_dir / class_name
                train_class_dir.mkdir(exist_ok=True)
                test_class_dir.mkdir(exist_ok=True)
                
                # Copy images
                for img in train_images:
                    shutil.copy2(img, train_class_dir / img.name)
                
                for img in test_images:
                    shutil.copy2(img, test_class_dir / img.name)
                
                split_stats["train"][class_name] = len(train_images)
                split_stats["test"][class_name] = len(test_images)
                
                print(f"  📁 {class_name}: {len(train_images)} train, {len(test_images)} test")
        
        # Save split statistics
        train_stats_path = train_dir / "split_stats.json"
        with open(train_stats_path, 'w') as f:
            json.dump(split_stats, f, indent=2)
        
        print(f"✅ Train/test split created:")
        print(f"  📁 Training: {train_dir}")
        print(f"  📁 Testing: {test_dir}")
        
        return str(train_dir), str(test_dir)
    
    def validate_dataset(self, dataset_dir: str):
        """Validate dataset structure and quality."""
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            print(f"❌ Dataset directory not found: {dataset_dir}")
            return False
        
        print(f"✅ Validating dataset: {dataset_dir}")
        
        issues = []
        stats = {"classes": {}, "total_images": 0}
        
        # Check each class directory
        class_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        if not class_dirs:
            issues.append("No class directories found")
            return False
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            
            # Count valid images
            valid_images = 0
            total_files = 0
            
            for file_path in class_dir.iterdir():
                if file_path.is_file():
                    total_files += 1
                    if file_path.suffix.lower() in self.supported_extensions:
                        if self._is_valid_image(file_path):
                            valid_images += 1
                        else:
                            issues.append(f"Invalid image: {file_path}")
            
            stats["classes"][class_name] = valid_images
            stats["total_images"] += valid_images
            
            print(f"  📁 {class_name}: {valid_images} valid images (of {total_files} files)")
            
            # Check for minimum images per class
            if valid_images < 5:
                issues.append(f"Class '{class_name}' has only {valid_images} images (minimum 5 recommended)")
        
        # Overall validation
        print(f"\n📊 Dataset Summary:")
        print(f"  Total classes: {len(stats['classes'])}")
        print(f"  Total images: {stats['total_images']}")
        
        if issues:
            print(f"\n⚠️  Issues found ({len(issues)}):")
            for issue in issues[:10]:  # Show first 10 issues
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
            return False
        else:
            print("✅ Dataset validation passed!")
            return True


def main():
    parser = argparse.ArgumentParser(description='Organize real pest/disease datasets')
    parser.add_argument('source', help='Source directory containing images')
    parser.add_argument('--type', choices=['plantvillage', 'custom'], default='custom',
                       help='Dataset type')
    parser.add_argument('--name', default='custom', help='Name for organized dataset')
    parser.add_argument('--split', action='store_true', help='Create train/test split')
    parser.add_argument('--validate', action='store_true', help='Validate dataset')
    
    args = parser.parse_args()
    
    organizer = RealDataOrganizer()
    
    print("📂 Real Data Organizer")
    print("=" * 40)
    
    # Organize dataset
    if args.type == 'plantvillage':
        organized_dir = organizer.organize_plantvillage_dataset(args.source)
    else:
        organized_dir = organizer.organize_custom_dataset(args.source, args.name)
    
    if organized_dir:
        # Validate if requested
        if args.validate:
            organizer.validate_dataset(organized_dir)
        
        # Create split if requested
        if args.split:
            organizer.create_train_test_split(organized_dir)
        
        print(f"\n✅ Organized dataset available at: {organized_dir}")
        print("🚀 Ready for training with: python algorithms/ml_classifier.py <dataset_path>")
    
    print("📋 Usage Instructions:")
    print("1. Place your downloaded images in appropriate class folders")
    print("2. Run: python tools/organize_real_data.py <source_dir> --validate --split")
    print("3. Train model: python algorithms/ml_classifier.py datasets/real/<organized_dir>")
    print("4. Test system: python tools/real_data_demo.py --dataset datasets/real/<organized_dir>")


if __name__ == "__main__":
    main()