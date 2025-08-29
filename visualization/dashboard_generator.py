"""
Real-time Dashboard Generator for Shakti RISC-V Pest Detection
=============================================================

Advanced dashboard system for real-time monitoring and analytics.
Optimized for embedded systems with memory and processing constraints.
"""

import cv2
import numpy as np
import time
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg


class DashboardGenerator:
    """
    Real-time dashboard generator for pest detection monitoring.
    Designed for efficient operation on Shakti E-class RISC-V systems.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        
        # Data storage for dashboard metrics (memory-optimized)
        self.metrics_history = deque(maxlen=self.config['max_history_points'])
        self.detection_events = deque(maxlen=self.config['max_detection_events'])
        self.performance_data = deque(maxlen=self.config['max_performance_points'])
        
        # Dashboard state
        self.start_time = time.time()
        self.last_update = time.time()
        
        # Visual settings
        self.colors = {
            'background': (30, 30, 30),
            'text': (255, 255, 255),
            'healthy': (0, 255, 0),
            'warning': (255, 165, 0),
            'critical': (255, 0, 0),
            'info': (100, 149, 237),
            'grid': (70, 70, 70)
        }
    
    def _get_default_config(self) -> Dict:
        """Get default configuration optimized for embedded systems."""
        return {
            'dashboard_resolution': (800, 600),
            'update_interval': 1.0,  # seconds
            'max_history_points': 100,  # Memory constraint
            'max_detection_events': 50,
            'max_performance_points': 60,
            'enable_charts': True,
            'enable_statistics': True,
            'enable_alerts': True,
            'chart_update_interval': 5.0,  # Less frequent for performance
            'save_dashboard_snapshots': False
        }
    
    def update_metrics(self, detection_results: Dict, performance_metrics: Dict):
        """
        Update dashboard metrics with new detection and performance data.
        
        Args:
            detection_results: Latest detection results
            performance_metrics: Performance metrics (FPS, processing time, etc.)
        """
        current_time = time.time()
        
        # Update metrics history
        metrics_entry = {
            'timestamp': current_time,
            'pest_detected': detection_results.get('pest_detected', False),
            'confidence': detection_results.get('confidence', 0.0),
            'pest_type': detection_results.get('class', 'none'),
            'severity': detection_results.get('severity', 'none')
        }
        self.metrics_history.append(metrics_entry)
        
        # Update detection events
        if detection_results.get('pest_detected', False):
            event_entry = {
                'timestamp': current_time,
                'pest_type': detection_results.get('class', 'unknown'),
                'confidence': detection_results.get('confidence', 0.0),
                'severity': detection_results.get('severity', 'low'),
                'location': detection_results.get('location', 'unknown')
            }
            self.detection_events.append(event_entry)
        
        # Update performance data
        perf_entry = {
            'timestamp': current_time,
            'fps': performance_metrics.get('current_fps', 0.0),
            'processing_time': performance_metrics.get('processing_time', 0.0),
            'memory_usage': performance_metrics.get('memory_usage', 0.0),
            'cpu_usage': performance_metrics.get('cpu_usage', 0.0)
        }
        self.performance_data.append(perf_entry)
        
        self.last_update = current_time
    
    def generate_dashboard(self) -> np.ndarray:
        """
        Generate complete dashboard visualization.
        
        Returns:
            Dashboard image as numpy array
        """
        width, height = self.config['dashboard_resolution']
        dashboard = np.full((height, width, 3), self.colors['background'], dtype=np.uint8)
        
        # Create dashboard sections
        header_height = 80
        stats_height = 120
        charts_height = 300
        alerts_height = height - header_height - stats_height - charts_height
        
        # 1. Header section
        dashboard = self._draw_header_section(dashboard, 0, header_height)
        
        # 2. Statistics section  
        dashboard = self._draw_statistics_section(
            dashboard, header_height, header_height + stats_height
        )
        
        # 3. Charts section
        if self.config['enable_charts']:
            dashboard = self._draw_charts_section(
                dashboard, header_height + stats_height, 
                header_height + stats_height + charts_height
            )
        
        # 4. Alerts section
        if self.config['enable_alerts']:
            dashboard = self._draw_alerts_section(
                dashboard, height - alerts_height, height
            )
        
        # Add timestamp
        timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(dashboard, timestamp_text, (width - 200, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        return dashboard
    
    def _draw_header_section(self, dashboard: np.ndarray, y_start: int, y_end: int) -> np.ndarray:
        """Draw dashboard header with title and system status."""
        width = dashboard.shape[1]
        
        # Draw header background
        cv2.rectangle(dashboard, (0, y_start), (width, y_end), (50, 50, 50), -1)
        cv2.line(dashboard, (0, y_end), (width, y_end), self.colors['grid'], 2)
        
        # Title
        title = "SHAKTI RISC-V PEST DETECTION SYSTEM"
        cv2.putText(dashboard, title, (20, y_start + 35),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors['text'], 2)
        
        # System status
        uptime = time.time() - self.start_time
        uptime_str = str(timedelta(seconds=int(uptime)))
        
        status_info = [
            f"Uptime: {uptime_str}",
            f"Status: {'MONITORING' if self.metrics_history else 'STANDBY'}",
            f"Last Update: {time.time() - self.last_update:.1f}s ago"
        ]
        
        for i, info in enumerate(status_info):
            cv2.putText(dashboard, info, (width - 300, y_start + 20 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['info'], 1)
        
        return dashboard
    
    def _draw_statistics_section(self, dashboard: np.ndarray, y_start: int, y_end: int) -> np.ndarray:
        """Draw key statistics section."""
        width = dashboard.shape[1]
        
        # Section background
        cv2.rectangle(dashboard, (0, y_start), (width, y_end), (40, 40, 40), -1)
        cv2.line(dashboard, (0, y_end), (width, y_end), self.colors['grid'], 1)
        
        # Calculate statistics
        stats = self._calculate_current_statistics()
        
        # Statistics layout (3 columns)
        col_width = width // 3
        
        # Column 1: Detection Statistics
        col1_stats = [
            f"Total Detections: {stats['total_detections']}",
            f"Detection Rate: {stats['detection_rate']:.1%}",
            f"Avg Confidence: {stats['avg_confidence']:.1%}"
        ]
        
        cv2.putText(dashboard, "DETECTION STATS", (20, y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        for i, stat in enumerate(col1_stats):
            cv2.putText(dashboard, stat, (20, y_start + 50 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['info'], 1)
        
        # Column 2: Pest Type Distribution
        cv2.putText(dashboard, "PEST TYPES", (col_width + 20, y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        pest_distribution = stats['pest_distribution']
        for i, (pest_type, count) in enumerate(list(pest_distribution.items())[:3]):
            text = f"{pest_type}: {count}"
            color = self._get_pest_color(pest_type)
            cv2.putText(dashboard, text, (col_width + 20, y_start + 50 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Column 3: Performance Statistics
        cv2.putText(dashboard, "PERFORMANCE", (2 * col_width + 20, y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        perf_stats = [
            f"Avg FPS: {stats['avg_fps']:.1f}",
            f"CPU Usage: {stats['avg_cpu']:.1f}%",
            f"Memory: {stats['avg_memory']:.1f}MB"
        ]
        
        for i, stat in enumerate(perf_stats):
            cv2.putText(dashboard, stat, (2 * col_width + 20, y_start + 50 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['info'], 1)
        
        return dashboard
    
    def _draw_charts_section(self, dashboard: np.ndarray, y_start: int, y_end: int) -> np.ndarray:
        """Draw charts and graphs section."""
        width = dashboard.shape[1]
        height = y_end - y_start
        
        # Section background
        cv2.rectangle(dashboard, (0, y_start), (width, y_end), (35, 35, 35), -1)
        cv2.line(dashboard, (0, y_end), (width, y_end), self.colors['grid'], 1)
        
        # Create charts (2 side by side)
        chart_width = width // 2 - 30
        chart_height = height - 40
        
        # Chart 1: Detection Timeline
        chart1 = self._create_detection_timeline_chart(chart_width, chart_height)
        if chart1 is not None:
            dashboard[y_start + 20:y_start + 20 + chart_height, 
                     20:20 + chart_width] = chart1
        
        # Chart 2: Performance Timeline
        chart2 = self._create_performance_chart(chart_width, chart_height)
        if chart2 is not None:
            dashboard[y_start + 20:y_start + 20 + chart_height,
                     width // 2 + 10:width // 2 + 10 + chart_width] = chart2
        
        return dashboard
    
    def _draw_alerts_section(self, dashboard: np.ndarray, y_start: int, y_end: int) -> np.ndarray:
        """Draw alerts and recent events section."""
        width = dashboard.shape[1]
        
        # Section background
        cv2.rectangle(dashboard, (0, y_start), (width, y_end), (45, 45, 45), -1)
        
        # Section title
        cv2.putText(dashboard, "RECENT ALERTS & EVENTS", (20, y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        # Recent detection events
        recent_events = list(self.detection_events)[-5:]  # Last 5 events
        
        for i, event in enumerate(recent_events):
            timestamp = datetime.fromtimestamp(event['timestamp']).strftime("%H:%M:%S")
            pest_type = event['pest_type']
            confidence = event['confidence']
            severity = event['severity']
            
            # Color based on severity
            if severity == 'high':
                color = self.colors['critical']
            elif severity == 'medium':
                color = self.colors['warning']
            else:
                color = self.colors['info']
            
            event_text = f"{timestamp} - {pest_type.upper()} detected ({confidence:.1%}, {severity})"
            cv2.putText(dashboard, event_text, (20, y_start + 50 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add alert summary
        alert_count = len([e for e in recent_events if e['severity'] in ['medium', 'high']])
        alert_text = f"Active Alerts: {alert_count}"
        alert_color = self.colors['critical'] if alert_count > 0 else self.colors['healthy']
        
        cv2.putText(dashboard, alert_text, (width - 200, y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)
        
        return dashboard
    
    def _create_detection_timeline_chart(self, width: int, height: int) -> Optional[np.ndarray]:
        """Create detection timeline chart."""
        try:
            if not self.metrics_history:
                return self._create_placeholder_chart(width, height, "No Detection Data")
            
            # Prepare data
            times = []
            detections = []
            confidences = []
            
            current_time = time.time()
            for entry in list(self.metrics_history)[-30:]:  # Last 30 points
                time_diff = (current_time - entry['timestamp']) / 60  # Minutes ago
                times.append(time_diff)
                detections.append(1 if entry['pest_detected'] else 0)
                confidences.append(entry['confidence'])
            
            # Create matplotlib figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width/100, height/100), 
                                          facecolor='black', tight_layout=True)
            
            # Detection events plot
            ax1.scatter(times, detections, c=['red' if d else 'green' for d in detections], 
                       s=20, alpha=0.7)
            ax1.set_ylabel('Detection', color='white', fontsize=8)
            ax1.set_ylim(-0.1, 1.1)
            ax1.set_facecolor('black')
            ax1.tick_params(colors='white', labelsize=6)
            ax1.grid(True, alpha=0.3)
            
            # Confidence plot
            ax2.plot(times, confidences, 'b-', linewidth=1, alpha=0.8)
            ax2.fill_between(times, confidences, alpha=0.3)
            ax2.set_xlabel('Minutes Ago', color='white', fontsize=8)
            ax2.set_ylabel('Confidence', color='white', fontsize=8)
            ax2.set_ylim(0, 1)
            ax2.set_facecolor('black')
            ax2.tick_params(colors='white', labelsize=6)
            ax2.grid(True, alpha=0.3)
            
            # Convert to image
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            
            # Resize to target dimensions
            resized = cv2.resize(buf, (width, height))
            return cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"Error creating detection chart: {e}")
            return self._create_placeholder_chart(width, height, "Chart Error")
    
    def _create_performance_chart(self, width: int, height: int) -> Optional[np.ndarray]:
        """Create performance metrics chart."""
        try:
            if not self.performance_data:
                return self._create_placeholder_chart(width, height, "No Performance Data")
            
            # Prepare data
            times = []
            fps_values = []
            cpu_values = []
            
            current_time = time.time()
            for entry in list(self.performance_data)[-30:]:  # Last 30 points
                time_diff = (current_time - entry['timestamp']) / 60  # Minutes ago
                times.append(time_diff)
                fps_values.append(entry['fps'])
                cpu_values.append(entry['cpu_usage'])
            
            # Create matplotlib figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width/100, height/100),
                                          facecolor='black', tight_layout=True)
            
            # FPS plot
            ax1.plot(times, fps_values, 'g-', linewidth=2, label='FPS')
            ax1.set_ylabel('FPS', color='white', fontsize=8)
            ax1.set_facecolor('black')
            ax1.tick_params(colors='white', labelsize=6)
            ax1.grid(True, alpha=0.3)
            
            # CPU usage plot
            ax2.plot(times, cpu_values, 'r-', linewidth=2, label='CPU %')
            ax2.set_xlabel('Minutes Ago', color='white', fontsize=8)
            ax2.set_ylabel('CPU %', color='white', fontsize=8)
            ax2.set_ylim(0, 100)
            ax2.set_facecolor('black')
            ax2.tick_params(colors='white', labelsize=6)
            ax2.grid(True, alpha=0.3)
            
            # Convert to image
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            
            # Resize to target dimensions
            resized = cv2.resize(buf, (width, height))
            return cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"Error creating performance chart: {e}")
            return self._create_placeholder_chart(width, height, "Chart Error")
    
    def _create_placeholder_chart(self, width: int, height: int, message: str) -> np.ndarray:
        """Create placeholder chart when data is not available."""
        chart = np.full((height, width, 3), (60, 60, 60), dtype=np.uint8)
        
        # Add border
        cv2.rectangle(chart, (0, 0), (width-1, height-1), self.colors['grid'], 1)
        
        # Add message
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        cv2.putText(chart, message, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
        
        return chart
    
    def _calculate_current_statistics(self) -> Dict:
        """Calculate current dashboard statistics."""
        stats = {
            'total_detections': len(self.detection_events),
            'detection_rate': 0.0,
            'avg_confidence': 0.0,
            'pest_distribution': {},
            'avg_fps': 0.0,
            'avg_cpu': 0.0,
            'avg_memory': 0.0
        }
        
        if self.metrics_history:
            # Detection rate
            detections = [m['pest_detected'] for m in self.metrics_history]
            stats['detection_rate'] = sum(detections) / len(detections)
            
            # Average confidence
            confidences = [m['confidence'] for m in self.metrics_history if m['pest_detected']]
            if confidences:
                stats['avg_confidence'] = sum(confidences) / len(confidences)
        
        if self.detection_events:
            # Pest distribution
            for event in self.detection_events:
                pest_type = event['pest_type']
                stats['pest_distribution'][pest_type] = stats['pest_distribution'].get(pest_type, 0) + 1
        
        if self.performance_data:
            # Performance averages
            recent_perf = list(self.performance_data)[-10:]  # Last 10 measurements
            stats['avg_fps'] = sum(p['fps'] for p in recent_perf) / len(recent_perf)
            stats['avg_cpu'] = sum(p['cpu_usage'] for p in recent_perf) / len(recent_perf)
            stats['avg_memory'] = sum(p['memory_usage'] for p in recent_perf) / len(recent_perf)
        
        return stats
    
    def _get_pest_color(self, pest_type: str) -> Tuple[int, int, int]:
        """Get color for pest type visualization."""
        color_map = {
            'healthy': self.colors['healthy'],
            'aphid': (0, 0, 255),
            'whitefly': (255, 255, 0),
            'leaf_spot': (128, 0, 128),
            'powdery_mildew': (255, 165, 0)
        }
        return color_map.get(pest_type.lower(), self.colors['info'])
    
    def save_dashboard_snapshot(self, dashboard: np.ndarray, filename: Optional[str] = None) -> bool:
        """Save dashboard snapshot to file."""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"dashboard_snapshot_{timestamp}.jpg"
            
            output_path = "/Users/navyamudgal/Works/ACAD/Pest-Detection/output/dashboards"
            os.makedirs(output_path, exist_ok=True)
            
            filepath = os.path.join(output_path, filename)
            cv2.imwrite(filepath, dashboard)
            
            return True
            
        except Exception as e:
            print(f"Error saving dashboard snapshot: {e}")
            return False
    
    def export_dashboard_data(self, filepath: str) -> bool:
        """Export dashboard data for analysis."""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'metrics_history': list(self.metrics_history),
                'detection_events': list(self.detection_events),
                'performance_data': list(self.performance_data),
                'statistics': self._calculate_current_statistics()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error exporting dashboard data: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Create dashboard generator
    dashboard = DashboardGenerator()
    
    # Simulate some data
    for i in range(10):
        test_detection = {
            'pest_detected': i % 3 == 0,
            'class': 'aphid' if i % 3 == 0 else 'healthy',
            'confidence': 0.8 if i % 3 == 0 else 0.1,
            'severity': 'medium' if i % 3 == 0 else 'none'
        }
        
        test_performance = {
            'current_fps': 25.0 + np.random.random() * 10,
            'processing_time': 0.04 + np.random.random() * 0.02,
            'memory_usage': 45.0 + np.random.random() * 10,
            'cpu_usage': 30.0 + np.random.random() * 20
        }
        
        dashboard.update_metrics(test_detection, test_performance)
        time.sleep(0.1)
    
    # Generate dashboard
    dashboard_image = dashboard.generate_dashboard()
    
    # Save snapshot
    success = dashboard.save_dashboard_snapshot(dashboard_image)
    print(f"Dashboard generated and saved: {success}")
    
    # Export data
    export_path = "/Users/navyamudgal/Works/ACAD/Pest-Detection/output/reports/dashboard_data.json"
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    dashboard.export_dashboard_data(export_path)