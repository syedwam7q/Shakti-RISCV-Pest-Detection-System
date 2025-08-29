# Getting Started with Pest Detection System

## 🚀 Quick Start Guide

### 1. Environment Setup

```bash
# Navigate to project directory
cd /Users/navyamudgal/Works/ACAD/Pest-Detection

# Install dependencies
pip3 install -r requirements.txt

# Verify installation
python3 -c "import cv2, sklearn; print('✅ All dependencies installed')"
```

### 2. Quick Demo (30 seconds)

```bash
# Run instant demo
python3 demo.py

# Run comprehensive real-data demo  
python3 tools/real_data_demo.py --mode full
```

## 🔄 **NEW: Real Dataset Integration**

### Option 1: Use Our Demo Dataset
```bash
# Setup demo dataset with realistic test images
python3 tools/dataset_downloader.py setup

# Train ML model on demo data
python3 tools/real_data_demo.py --mode train

# Test with real data
python3 tools/real_data_demo.py --mode test
```

### Option 2: Download PlantVillage Dataset
```bash
# 1. Download from: https://www.kaggle.com/datasets/emmarex/plantdisease
# 2. Extract to: datasets/real/plantvillage/
# 3. Organize the data:
python3 tools/organize_real_data.py datasets/real/plantvillage --type plantvillage --validate --split

# 4. Train on real data:
python3 algorithms/ml_classifier.py datasets/real/plantvillage_organized_train

# 5. Run comprehensive demo:
python3 tools/real_data_demo.py --dataset datasets/real/plantvillage_organized --mode full
```

### Option 3: Use Your Own Images
```bash
# 1. Create directory structure:
mkdir -p datasets/real/my_images/{healthy,aphid,whitefly,leaf_spot,powdery_mildew}

# 2. Add your images to appropriate folders

# 3. Organize and validate:
python3 tools/organize_real_data.py datasets/real/my_images --validate --split

# 4. Train and test:
python3 tools/real_data_demo.py --dataset datasets/real/my_images --mode full
```

## 🎯 **Enhanced System Capabilities**

### ✅ **Production Ready Features**
- **Machine Learning**: Random Forest & SVM classifiers trained on real data
- **Ensemble Detection**: Combines ML + rule-based approaches for highest accuracy
- **Real-time Performance**: 30+ FPS with confidence scoring
- **Advanced Analytics**: Severity assessment, risk evaluation, treatment recommendations
- **Multiple Detection Methods**: Auto-select best method based on confidence
- **Comprehensive Testing**: Automated validation on real pest images

### 🧠 **Enhanced AI Detection**
```bash
# Test enhanced detector with real images
python3 algorithms/enhanced_pest_detector.py datasets/real/your_images

# Available detection methods:
# - "auto"     : Automatically selects best method
# - "ml"       : Machine learning classifier
# - "rule"     : Rule-based classifier  
# - "ensemble" : Combines ML + rules for best accuracy
```

### 📊 **Advanced Performance Analysis**
```bash
# Run comprehensive performance testing
python3 tools/real_data_demo.py --mode test --images datasets/real/test_images

# Interactive demo mode
python3 tools/real_data_demo.py --mode interactive
```

## 🎮 **Demo Modes**

### Quick Demo (10 seconds)
```bash
python3 demo.py
# Output: 10-frame pest detection demonstration
```

### Comprehensive Demo (2-3 minutes)
```bash
python3 tools/real_data_demo.py --mode full
# Includes: Setup → Training → Testing → Performance Analysis
```

### Interactive Mode
```bash
python3 tools/real_data_demo.py --mode interactive
# Commands: test, stats, info <class>, alert, quit
```

### Continuous Monitoring
```bash
python3 simulation/main_app.py --mode continuous
# Real-time pest detection with GUI
# Controls: 'q' quit, 's' save frame
```

## 🔍 **Understanding Enhanced Output**

### Standard Detection Result
```
🔴 PEST DETECTED: whitefly (85.2% confidence)
Method: ensemble (ML: 82%, Rules: 71%)
Severity: medium | Risk Level: moderate
Affected Regions: 3
Processing Time: 33.2ms (30.1 FPS)

Treatment Recommendations:
1. ⚠️ MODERATE: Treatment recommended within 24-48 hours  
2. Use yellow sticky traps
3. Apply insecticidal soap or oil
4. Remove heavily infested leaves
```

### Performance Statistics
```
📊 PERFORMANCE STATISTICS
====================================
Total Images Processed: 127
Average Processing Time: 31.4 ms
Processing Speed: 31.8 FPS
Healthy Detections: 89 (70.1%)
Pest Detections: 38 (29.9%)
⚡ Speed: EXCELLENT (Real-time capable)
```

## 📋 **System Architecture (Enhanced)**

```
📷 Camera Input → 🔍 Enhanced Processing → 🤖 ML/Ensemble Detection → 🚨 Smart Alerts
                        ↓                      ↓                         ↓
                   • Real image support    • ML classifier            • Risk assessment
                   • Advanced features     • Rule-based backup        • Severity analysis
                   • Quality validation    • Ensemble methods         • Treatment recommendations
```

## 🛠️ **Development Workflow (Updated)**

### For Real Data Integration:
1. **Get Dataset**: Download PlantVillage or use your images
2. **Organize**: Use `organize_real_data.py` to structure data
3. **Train**: Use `ml_classifier.py` to train on real data
4. **Test**: Use `real_data_demo.py` for comprehensive testing
5. **Deploy**: System ready for hardware integration

### For Algorithm Development:
1. **Enhance**: Modify enhanced algorithms in `algorithms/`
2. **Test**: Use real data demo tools for validation
3. **Optimize**: Performance testing with real images
4. **Validate**: Comprehensive accuracy testing


## 🔧 **Advanced Configuration**

### Enhanced Config File (`config.json`):
```json
{
  "detection": {
    "method": "auto",
    "confidence_threshold": 0.6,
    "ensemble_weights": {"ml": 0.7, "rule": 0.3}
  },
  "performance": {
    "target_fps": 30,
    "max_processing_time": 50
  },
  "alerts": {
    "severity_thresholds": {"low": 0.6, "medium": 0.75, "high": 0.9},
    "enable_iot": true,
    "enable_buzzer": true
  },
  "ml_model": {
    "model_type": "random_forest",
    "model_path": "models/enhanced_pest_detector.pkl"
  }
}
```
