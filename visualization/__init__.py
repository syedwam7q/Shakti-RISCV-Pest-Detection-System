"""
Enhanced Visualization System for Shakti RISC-V Pest Detection
================================================================

Complete visual representation system with:
- Bounding box detection and visualization
- Confidence heat maps and color coding
- Real-time performance dashboard
- Detection history and analytics
- Export capabilities for analysis

Optimized for Shakti E-class RISC-V processor deployment.
"""

from .detection_visualizer import DetectionVisualizer
from .dashboard_generator import DashboardGenerator
from .report_generator import ReportGenerator
from .bounding_box_detector import BoundingBoxDetector

__all__ = [
    'DetectionVisualizer',
    'DashboardGenerator', 
    'ReportGenerator',
    'BoundingBoxDetector'
]

__version__ = "1.0.0"
__author__ = "Shakti RISC-V Pest Detection Team"