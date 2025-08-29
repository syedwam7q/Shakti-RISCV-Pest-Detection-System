"""
Comprehensive Report Generator for Shakti RISC-V Pest Detection
==============================================================

Advanced reporting system for analysis, documentation, and presentation.
Optimized for embedded systems deployment and agricultural research.
"""

import cv2
import numpy as np
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pathlib import Path


class ReportGenerator:
    """
    Comprehensive report generator for pest detection analysis.
    Creates detailed reports for research, deployment, and monitoring.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.report_data = {
            'session_info': {},
            'detection_summary': {},
            'performance_metrics': {},
            'analysis_results': {},
            'recommendations': []
        }
        
        # Ensure output directories exist
        self._ensure_output_directories()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for report generation."""
        return {
            'output_base_path': '/Users/navyamudgal/Works/ACAD/Pest-Detection/output',
            'report_formats': ['json', 'html', 'pdf'],
            'include_charts': True,
            'include_images': True,
            'include_statistics': True,
            'include_recommendations': True,
            'chart_resolution': (800, 600),
            'chart_dpi': 100,
            'max_sample_images': 20
        }
    
    def _ensure_output_directories(self):
        """Ensure all required output directories exist."""
        base_path = Path(self.config['output_base_path'])
        directories = ['reports', 'charts', 'analysis', 'exports']
        
        for directory in directories:
            (base_path / directory).mkdir(parents=True, exist_ok=True)
    
    def generate_comprehensive_report(self, 
                                    detection_history: List[Dict],
                                    performance_data: List[Dict],
                                    annotated_images: Optional[List[np.ndarray]] = None,
                                    session_metadata: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive analysis report.
        
        Args:
            detection_history: List of detection results over time
            performance_data: List of performance metrics
            annotated_images: Optional list of annotated images for analysis
            session_metadata: Optional metadata about the detection session
            
        Returns:
            Dictionary with report generation results and file paths
        """
        print("🔍 Generating comprehensive pest detection report...")
        start_time = time.time()
        
        # Initialize report data
        self._initialize_report_data(session_metadata)
        
        # Analyze detection data
        detection_analysis = self._analyze_detection_data(detection_history)
        self.report_data['detection_summary'] = detection_analysis
        
        # Analyze performance data
        performance_analysis = self._analyze_performance_data(performance_data)
        self.report_data['performance_metrics'] = performance_analysis
        
        # Generate advanced analytics
        advanced_analysis = self._generate_advanced_analytics(detection_history, performance_data)
        self.report_data['analysis_results'] = advanced_analysis
        
        # Generate recommendations
        recommendations = self._generate_recommendations(detection_analysis, performance_analysis)
        self.report_data['recommendations'] = recommendations
        
        # Create visualizations
        chart_paths = []
        if self.config['include_charts']:
            chart_paths = self._generate_analysis_charts(detection_history, performance_data)
        
        # Process sample images
        image_analysis = {}
        if annotated_images and self.config['include_images']:
            image_analysis = self._analyze_sample_images(annotated_images)
        
        # Generate report files
        report_files = self._generate_report_files(chart_paths, image_analysis)
        
        generation_time = time.time() - start_time
        print(f"✅ Report generated in {generation_time:.2f}s")
        
        return {
            'success': True,
            'generation_time': generation_time,
            'report_files': report_files,
            'chart_paths': chart_paths,
            'summary': self._create_report_summary()
        }
    
    def _initialize_report_data(self, session_metadata: Optional[Dict]):
        """Initialize report with session information."""
        self.report_data['session_info'] = {
            'generation_timestamp': datetime.now().isoformat(),
            'system_info': {
                'platform': 'Shakti E-class RISC-V',
                'board': 'Arty A7-35T',
                'software_version': '1.0.0',
                'optimization_level': 'embedded'
            },
            'session_metadata': session_metadata or {},
            'report_config': self.config.copy()
        }
    
    def _analyze_detection_data(self, detection_history: List[Dict]) -> Dict:
        """Analyze detection data for comprehensive insights."""
        if not detection_history:
            return {'error': 'No detection data available'}
        
        # Basic statistics
        total_frames = len(detection_history)
        pest_detections = [d for d in detection_history if d.get('pest_detected', False)]
        total_pest_detections = len(pest_detections)
        
        # Detection rate analysis
        detection_rate = total_pest_detections / total_frames if total_frames > 0 else 0
        
        # Confidence analysis
        confidences = [d.get('confidence', 0.0) for d in pest_detections]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        confidence_std = np.std(confidences) if confidences else 0.0
        
        # Pest type distribution
        pest_types = {}
        for detection in pest_detections:
            pest_type = detection.get('class', 'unknown')
            pest_types[pest_type] = pest_types.get(pest_type, 0) + 1
        
        # Severity analysis
        severity_distribution = {'low': 0, 'medium': 0, 'high': 0}
        for detection in pest_detections:
            severity = detection.get('severity', 'low')
            if severity in severity_distribution:
                severity_distribution[severity] += 1
        
        # Temporal analysis
        temporal_analysis = self._analyze_detection_temporal_patterns(detection_history)
        
        return {
            'total_frames': total_frames,
            'total_pest_detections': total_pest_detections,
            'detection_rate': detection_rate,
            'confidence_statistics': {
                'average': avg_confidence,
                'standard_deviation': confidence_std,
                'minimum': min(confidences) if confidences else 0.0,
                'maximum': max(confidences) if confidences else 0.0
            },
            'pest_type_distribution': pest_types,
            'severity_distribution': severity_distribution,
            'temporal_analysis': temporal_analysis
        }
    
    def _analyze_performance_data(self, performance_data: List[Dict]) -> Dict:
        """Analyze system performance metrics."""
        if not performance_data:
            return {'error': 'No performance data available'}
        
        # FPS analysis
        fps_values = [p.get('fps', 0.0) for p in performance_data]
        avg_fps = np.mean(fps_values)
        min_fps = min(fps_values)
        max_fps = max(fps_values)
        fps_stability = 1.0 - (np.std(fps_values) / avg_fps) if avg_fps > 0 else 0.0
        
        # Processing time analysis
        processing_times = [p.get('processing_time', 0.0) for p in performance_data]
        avg_processing_time = np.mean(processing_times)
        
        # Resource usage analysis
        memory_usage = [p.get('memory_usage', 0.0) for p in performance_data]
        cpu_usage = [p.get('cpu_usage', 0.0) for p in performance_data]
        
        avg_memory = np.mean(memory_usage) if memory_usage else 0.0
        avg_cpu = np.mean(cpu_usage) if cpu_usage else 0.0
        
        # Performance rating (for embedded systems)
        performance_rating = self._calculate_performance_rating(avg_fps, avg_processing_time, avg_cpu, avg_memory)
        
        return {
            'fps_statistics': {
                'average': avg_fps,
                'minimum': min_fps,
                'maximum': max_fps,
                'stability_score': fps_stability
            },
            'processing_time_statistics': {
                'average': avg_processing_time,
                'minimum': min(processing_times) if processing_times else 0.0,
                'maximum': max(processing_times) if processing_times else 0.0
            },
            'resource_usage': {
                'average_memory_mb': avg_memory,
                'average_cpu_percent': avg_cpu,
                'memory_efficiency': self._calculate_memory_efficiency(avg_memory),
                'cpu_efficiency': self._calculate_cpu_efficiency(avg_cpu)
            },
            'performance_rating': performance_rating,
            'embedded_optimization_score': self._calculate_embedded_optimization_score(performance_data)
        }
    
    def _generate_advanced_analytics(self, detection_history: List[Dict], 
                                   performance_data: List[Dict]) -> Dict:
        """Generate advanced analytics and insights."""
        analytics = {}
        
        # Detection accuracy trends
        if detection_history:
            analytics['accuracy_trends'] = self._analyze_accuracy_trends(detection_history)
        
        # Performance correlation analysis
        if detection_history and performance_data:
            analytics['performance_correlation'] = self._analyze_performance_correlation(
                detection_history, performance_data
            )
        
        # Anomaly detection
        analytics['anomalies'] = self._detect_anomalies(detection_history, performance_data)
        
        # Efficiency analysis for embedded deployment
        analytics['embedded_efficiency'] = self._analyze_embedded_efficiency(performance_data)
        
        # Predictive insights
        analytics['predictive_insights'] = self._generate_predictive_insights(detection_history)
        
        return analytics
    
    def _generate_recommendations(self, detection_analysis: Dict, 
                                performance_analysis: Dict) -> List[Dict]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Detection-based recommendations
        detection_rate = detection_analysis.get('detection_rate', 0.0)
        if detection_rate > 0.3:  # High pest activity
            recommendations.append({
                'category': 'Agricultural Action',
                'priority': 'high',
                'title': 'High Pest Activity Detected',
                'description': f'Detection rate of {detection_rate:.1%} indicates significant pest presence.',
                'actions': [
                    'Implement immediate pest control measures',
                    'Increase monitoring frequency',
                    'Consider targeted pesticide application',
                    'Consult agricultural specialist'
                ]
            })
        
        # Performance-based recommendations
        avg_fps = performance_analysis.get('fps_statistics', {}).get('average', 0.0)
        if avg_fps < 15.0:  # Low FPS
            recommendations.append({
                'category': 'System Optimization',
                'priority': 'medium',
                'title': 'Performance Optimization Needed',
                'description': f'Average FPS of {avg_fps:.1f} is below optimal threshold.',
                'actions': [
                    'Optimize image processing algorithms',
                    'Reduce image resolution if acceptable',
                    'Consider hardware upgrade',
                    'Profile memory usage patterns'
                ]
            })
        
        # Embedded system recommendations
        memory_usage = performance_analysis.get('resource_usage', {}).get('average_memory_mb', 0.0)
        if memory_usage > 200:  # High memory usage for embedded system
            recommendations.append({
                'category': 'Embedded Optimization',
                'priority': 'high',
                'title': 'Memory Usage Optimization',
                'description': f'Memory usage of {memory_usage:.1f}MB exceeds embedded system limits.',
                'actions': [
                    'Implement memory pooling',
                    'Optimize data structures',
                    'Reduce buffer sizes',
                    'Enable streaming processing'
                ]
            })
        
        # Confidence-based recommendations
        avg_confidence = detection_analysis.get('confidence_statistics', {}).get('average', 0.0)
        if avg_confidence < 0.7:  # Low confidence
            recommendations.append({
                'category': 'Algorithm Improvement',
                'priority': 'medium',
                'title': 'Detection Confidence Enhancement',
                'description': f'Average confidence of {avg_confidence:.1%} suggests model improvement needed.',
                'actions': [
                    'Retrain model with more diverse data',
                    'Improve image preprocessing',
                    'Adjust detection thresholds',
                    'Collect additional training samples'
                ]
            })
        
        return recommendations
    
    def _generate_analysis_charts(self, detection_history: List[Dict], 
                                performance_data: List[Dict]) -> List[str]:
        """Generate analysis charts and save them."""
        chart_paths = []
        chart_dir = Path(self.config['output_base_path']) / 'charts'
        
        try:
            # 1. Detection Timeline Chart
            if detection_history:
                chart_path = self._create_detection_timeline_chart(detection_history, chart_dir)
                if chart_path:
                    chart_paths.append(chart_path)
            
            # 2. Performance Metrics Chart
            if performance_data:
                chart_path = self._create_performance_metrics_chart(performance_data, chart_dir)
                if chart_path:
                    chart_paths.append(chart_path)
            
            # 3. Pest Distribution Chart
            if detection_history:
                chart_path = self._create_pest_distribution_chart(detection_history, chart_dir)
                if chart_path:
                    chart_paths.append(chart_path)
            
            # 4. Confidence Analysis Chart
            if detection_history:
                chart_path = self._create_confidence_analysis_chart(detection_history, chart_dir)
                if chart_path:
                    chart_paths.append(chart_path)
            
        except Exception as e:
            print(f"Error generating charts: {e}")
        
        return chart_paths
    
    def _create_detection_timeline_chart(self, detection_history: List[Dict], 
                                       output_dir: Path) -> Optional[str]:
        """Create detection timeline visualization."""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Prepare data
            timestamps = []
            detections = []
            confidences = []
            
            for i, detection in enumerate(detection_history):
                timestamps.append(i)
                detections.append(1 if detection.get('pest_detected', False) else 0)
                confidences.append(detection.get('confidence', 0.0))
            
            # Detection events plot
            ax1.scatter(timestamps, detections, 
                       c=['red' if d else 'green' for d in detections], 
                       s=30, alpha=0.7)
            ax1.set_ylabel('Pest Detected')
            ax1.set_title('Pest Detection Timeline')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(-0.1, 1.1)
            
            # Confidence plot
            ax2.plot(timestamps, confidences, 'b-', linewidth=2, alpha=0.8)
            ax2.fill_between(timestamps, confidences, alpha=0.3)
            ax2.set_xlabel('Frame Number')
            ax2.set_ylabel('Detection Confidence')
            ax2.set_title('Detection Confidence Over Time')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = output_dir / f"detection_timeline_{int(time.time())}.png"
            plt.savefig(chart_path, dpi=self.config['chart_dpi'], bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating detection timeline chart: {e}")
            return None
    
    def _create_performance_metrics_chart(self, performance_data: List[Dict], 
                                        output_dir: Path) -> Optional[str]:
        """Create performance metrics visualization."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Prepare data
            timestamps = list(range(len(performance_data)))
            fps_values = [p.get('fps', 0.0) for p in performance_data]
            processing_times = [p.get('processing_time', 0.0) * 1000 for p in performance_data]  # Convert to ms
            memory_usage = [p.get('memory_usage', 0.0) for p in performance_data]
            cpu_usage = [p.get('cpu_usage', 0.0) for p in performance_data]
            
            # FPS plot
            ax1.plot(timestamps, fps_values, 'g-', linewidth=2)
            ax1.set_ylabel('FPS')
            ax1.set_title('Frames Per Second')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='Target FPS')
            ax1.legend()
            
            # Processing time plot
            ax2.plot(timestamps, processing_times, 'b-', linewidth=2)
            ax2.set_ylabel('Processing Time (ms)')
            ax2.set_title('Processing Time per Frame')
            ax2.grid(True, alpha=0.3)
            
            # Memory usage plot
            ax3.plot(timestamps, memory_usage, 'r-', linewidth=2)
            ax3.set_xlabel('Frame Number')
            ax3.set_ylabel('Memory Usage (MB)')
            ax3.set_title('Memory Usage')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=256, color='r', linestyle='--', alpha=0.5, label='System Limit')
            ax3.legend()
            
            # CPU usage plot
            ax4.plot(timestamps, cpu_usage, 'orange', linewidth=2)
            ax4.set_xlabel('Frame Number')
            ax4.set_ylabel('CPU Usage (%)')
            ax4.set_title('CPU Usage')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='High Usage')
            ax4.legend()
            
            plt.tight_layout()
            
            # Save chart
            chart_path = output_dir / f"performance_metrics_{int(time.time())}.png"
            plt.savefig(chart_path, dpi=self.config['chart_dpi'], bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating performance metrics chart: {e}")
            return None
    
    def _create_pest_distribution_chart(self, detection_history: List[Dict], 
                                      output_dir: Path) -> Optional[str]:
        """Create pest type distribution chart."""
        try:
            # Count pest types
            pest_counts = {}
            for detection in detection_history:
                if detection.get('pest_detected', False):
                    pest_type = detection.get('class', 'unknown')
                    pest_counts[pest_type] = pest_counts.get(pest_type, 0) + 1
            
            if not pest_counts:
                return None
            
            # Create pie chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Pie chart
            labels = list(pest_counts.keys())
            sizes = list(pest_counts.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Pest Type Distribution')
            
            # Bar chart
            ax2.bar(labels, sizes, color=colors)
            ax2.set_ylabel('Detection Count')
            ax2.set_title('Pest Type Frequency')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = output_dir / f"pest_distribution_{int(time.time())}.png"
            plt.savefig(chart_path, dpi=self.config['chart_dpi'], bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating pest distribution chart: {e}")
            return None
    
    def _create_confidence_analysis_chart(self, detection_history: List[Dict], 
                                        output_dir: Path) -> Optional[str]:
        """Create confidence analysis visualization."""
        try:
            # Extract confidence data
            pest_detections = [d for d in detection_history if d.get('pest_detected', False)]
            if not pest_detections:
                return None
            
            confidences = [d.get('confidence', 0.0) for d in pest_detections]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            ax1.hist(confidences, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax1.set_xlabel('Confidence')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Confidence Distribution')
            ax1.grid(True, alpha=0.3)
            ax1.axvline(x=np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.2f}')
            ax1.legend()
            
            # Box plot
            ax2.boxplot(confidences, vert=True)
            ax2.set_ylabel('Confidence')
            ax2.set_title('Confidence Statistics')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = output_dir / f"confidence_analysis_{int(time.time())}.png"
            plt.savefig(chart_path, dpi=self.config['chart_dpi'], bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating confidence analysis chart: {e}")
            return None
    
    def _analyze_sample_images(self, annotated_images: List[np.ndarray]) -> Dict:
        """Analyze sample annotated images."""
        if not annotated_images:
            return {}
        
        analysis = {
            'total_images': len(annotated_images),
            'image_statistics': {
                'average_resolution': None,
                'color_statistics': {},
                'quality_metrics': {}
            }
        }
        
        # Analyze image properties
        resolutions = []
        for img in annotated_images[:self.config['max_sample_images']]:
            height, width = img.shape[:2]
            resolutions.append((width, height))
        
        if resolutions:
            avg_width = np.mean([r[0] for r in resolutions])
            avg_height = np.mean([r[1] for r in resolutions])
            analysis['image_statistics']['average_resolution'] = f"{avg_width:.0f}x{avg_height:.0f}"
        
        return analysis
    
    def _generate_report_files(self, chart_paths: List[str], 
                             image_analysis: Dict) -> Dict:
        """Generate report files in various formats."""
        report_files = {}
        
        # JSON Report
        if 'json' in self.config['report_formats']:
            json_path = self._generate_json_report()
            if json_path:
                report_files['json'] = json_path
        
        # HTML Report
        if 'html' in self.config['report_formats']:
            html_path = self._generate_html_report(chart_paths)
            if html_path:
                report_files['html'] = html_path
        
        return report_files
    
    def _generate_json_report(self) -> Optional[str]:
        """Generate JSON format report."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pest_detection_report_{timestamp}.json"
            
            report_dir = Path(self.config['output_base_path']) / 'reports'
            filepath = report_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(self.report_data, f, indent=2, default=str)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating JSON report: {e}")
            return None
    
    def _generate_html_report(self, chart_paths: List[str]) -> Optional[str]:
        """Generate HTML format report."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pest_detection_report_{timestamp}.html"
            
            report_dir = Path(self.config['output_base_path']) / 'reports'
            filepath = report_dir / filename
            
            html_content = self._create_html_content(chart_paths)
            
            with open(filepath, 'w') as f:
                f.write(html_content)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            return None
    
    def _create_html_content(self, chart_paths: List[str]) -> str:
        """Create HTML content for the report."""
        session_info = self.report_data.get('session_info', {})
        detection_summary = self.report_data.get('detection_summary', {})
        performance_metrics = self.report_data.get('performance_metrics', {})
        recommendations = self.report_data.get('recommendations', [])
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Shakti RISC-V Pest Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .metric {{ background-color: #f8f9fa; padding: 10px; margin: 5px 0; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }}
                .high-priority {{ border-left-color: #dc3545; }}
                .medium-priority {{ border-left-color: #ffc107; }}
                .low-priority {{ border-left-color: #28a745; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌱 Shakti RISC-V Pest Detection System</h1>
                <h2>Comprehensive Analysis Report</h2>
                <p>Generated: {session_info.get('generation_timestamp', 'Unknown')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Detection Summary</h2>
                <div class="metric">Total Frames Processed: {detection_summary.get('total_frames', 0)}</div>
                <div class="metric">Total Pest Detections: {detection_summary.get('total_pest_detections', 0)}</div>
                <div class="metric">Detection Rate: {detection_summary.get('detection_rate', 0):.1%}</div>
                <div class="metric">Average Confidence: {detection_summary.get('confidence_statistics', {}).get('average', 0):.1%}</div>
            </div>
            
            <div class="section">
                <h2>⚡ Performance Metrics</h2>
                <div class="metric">Average FPS: {performance_metrics.get('fps_statistics', {}).get('average', 0):.1f}</div>
                <div class="metric">Average Processing Time: {performance_metrics.get('processing_time_statistics', {}).get('average', 0)*1000:.1f}ms</div>
                <div class="metric">Memory Usage: {performance_metrics.get('resource_usage', {}).get('average_memory_mb', 0):.1f}MB</div>
                <div class="metric">CPU Usage: {performance_metrics.get('resource_usage', {}).get('average_cpu_percent', 0):.1f}%</div>
            </div>
        """
        
        # Add charts
        if chart_paths:
            html += """
            <div class="section">
                <h2>📈 Analysis Charts</h2>
            """
            for chart_path in chart_paths:
                chart_name = Path(chart_path).name
                html += f'<div class="chart"><img src="{chart_path}" alt="{chart_name}" style="max-width: 100%;"></div>'
            html += "</div>"
        
        # Add recommendations
        if recommendations:
            html += """
            <div class="section">
                <h2>💡 Recommendations</h2>
            """
            for rec in recommendations:
                priority_class = f"{rec.get('priority', 'low')}-priority"
                html += f"""
                <div class="recommendation {priority_class}">
                    <h3>{rec.get('title', 'Recommendation')}</h3>
                    <p><strong>Category:</strong> {rec.get('category', 'General')}</p>
                    <p><strong>Priority:</strong> {rec.get('priority', 'Low').upper()}</p>
                    <p>{rec.get('description', '')}</p>
                    <ul>
                """
                for action in rec.get('actions', []):
                    html += f"<li>{action}</li>"
                html += "</ul></div>"
            html += "</div>"
        
        html += """
            <div class="section">
                <h2>🔧 System Information</h2>
                <div class="metric">Platform: Shakti E-class RISC-V</div>
                <div class="metric">Board: Arty A7-35T FPGA</div>
                <div class="metric">Software Version: 1.0.0</div>
                <div class="metric">Optimization: Embedded Systems</div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_report_summary(self) -> Dict:
        """Create a summary of the generated report."""
        detection_summary = self.report_data.get('detection_summary', {})
        performance_metrics = self.report_data.get('performance_metrics', {})
        
        return {
            'total_detections': detection_summary.get('total_pest_detections', 0),
            'detection_rate': detection_summary.get('detection_rate', 0.0),
            'average_fps': performance_metrics.get('fps_statistics', {}).get('average', 0.0),
            'recommendations_count': len(self.report_data.get('recommendations', [])),
            'report_quality': 'high' if detection_summary.get('total_frames', 0) > 100 else 'medium'
        }
    
    # Helper methods for analysis
    def _analyze_detection_temporal_patterns(self, detection_history: List[Dict]) -> Dict:
        """Analyze temporal patterns in detections."""
        if len(detection_history) < 10:
            return {'insufficient_data': True}
        
        # Simple pattern analysis
        detection_sequence = [1 if d.get('pest_detected', False) else 0 for d in detection_history]
        
        # Calculate detection clusters
        clusters = []
        current_cluster = 0
        for detection in detection_sequence:
            if detection:
                current_cluster += 1
            else:
                if current_cluster > 0:
                    clusters.append(current_cluster)
                    current_cluster = 0
        
        if current_cluster > 0:
            clusters.append(current_cluster)
        
        return {
            'detection_clusters': len(clusters),
            'average_cluster_size': np.mean(clusters) if clusters else 0,
            'largest_cluster': max(clusters) if clusters else 0
        }
    
    def _analyze_accuracy_trends(self, detection_history: List[Dict]) -> Dict:
        """Analyze detection accuracy trends over time."""
        if len(detection_history) < 20:
            return {'insufficient_data': True}
        
        # Split into segments and analyze confidence trends
        segment_size = len(detection_history) // 4
        segments = [detection_history[i:i+segment_size] for i in range(0, len(detection_history), segment_size)]
        
        segment_confidences = []
        for segment in segments:
            pest_detections = [d for d in segment if d.get('pest_detected', False)]
            if pest_detections:
                avg_confidence = np.mean([d.get('confidence', 0.0) for d in pest_detections])
                segment_confidences.append(avg_confidence)
        
        if len(segment_confidences) > 1:
            trend = 'improving' if segment_confidences[-1] > segment_confidences[0] else 'declining'
        else:
            trend = 'stable'
        
        return {
            'confidence_trend': trend,
            'segment_confidences': segment_confidences
        }
    
    def _analyze_performance_correlation(self, detection_history: List[Dict], 
                                       performance_data: List[Dict]) -> Dict:
        """Analyze correlation between detection and performance."""
        if len(detection_history) != len(performance_data):
            return {'error': 'Mismatched data lengths'}
        
        # Calculate correlation between detection confidence and FPS
        confidences = []
        fps_values = []
        
        for i, detection in enumerate(detection_history):
            if detection.get('pest_detected', False) and i < len(performance_data):
                confidences.append(detection.get('confidence', 0.0))
                fps_values.append(performance_data[i].get('fps', 0.0))
        
        if len(confidences) > 5:
            correlation = np.corrcoef(confidences, fps_values)[0, 1]
            return {
                'confidence_fps_correlation': correlation,
                'interpretation': 'strong' if abs(correlation) > 0.7 else 'weak'
            }
        
        return {'insufficient_data': True}
    
    def _detect_anomalies(self, detection_history: List[Dict], 
                         performance_data: List[Dict]) -> Dict:
        """Detect anomalies in detection and performance data."""
        anomalies = {
            'detection_anomalies': [],
            'performance_anomalies': []
        }
        
        # Simple anomaly detection based on z-score
        if performance_data:
            fps_values = [p.get('fps', 0.0) for p in performance_data]
            fps_mean = np.mean(fps_values)
            fps_std = np.std(fps_values)
            
            for i, fps in enumerate(fps_values):
                if abs(fps - fps_mean) > 2 * fps_std:  # 2-sigma threshold
                    anomalies['performance_anomalies'].append({
                        'frame': i,
                        'metric': 'fps',
                        'value': fps,
                        'severity': 'high' if abs(fps - fps_mean) > 3 * fps_std else 'medium'
                    })
        
        return anomalies
    
    def _analyze_embedded_efficiency(self, performance_data: List[Dict]) -> Dict:
        """Analyze efficiency for embedded system deployment."""
        if not performance_data:
            return {}
        
        # Memory efficiency
        memory_values = [p.get('memory_usage', 0.0) for p in performance_data]
        memory_efficiency = 1.0 - (np.mean(memory_values) / 256.0)  # Assuming 256MB limit
        
        # Processing efficiency
        processing_times = [p.get('processing_time', 0.0) for p in performance_data]
        processing_efficiency = 1.0 - (np.mean(processing_times) / 0.1)  # Assuming 100ms target
        
        return {
            'memory_efficiency': max(0, memory_efficiency),
            'processing_efficiency': max(0, processing_efficiency),
            'overall_efficiency': (memory_efficiency + processing_efficiency) / 2
        }
    
    def _generate_predictive_insights(self, detection_history: List[Dict]) -> Dict:
        """Generate predictive insights based on detection patterns."""
        if len(detection_history) < 50:
            return {'insufficient_data': True}
        
        # Simple trend analysis
        recent_detections = detection_history[-20:]
        earlier_detections = detection_history[-40:-20]
        
        recent_rate = sum(1 for d in recent_detections if d.get('pest_detected', False)) / len(recent_detections)
        earlier_rate = sum(1 for d in earlier_detections if d.get('pest_detected', False)) / len(earlier_detections)
        
        trend = 'increasing' if recent_rate > earlier_rate * 1.2 else 'decreasing' if recent_rate < earlier_rate * 0.8 else 'stable'
        
        return {
            'detection_trend': trend,
            'recent_detection_rate': recent_rate,
            'earlier_detection_rate': earlier_rate,
            'prediction': f"Pest activity is {trend}"
        }
    
    def _calculate_performance_rating(self, avg_fps: float, avg_processing_time: float, 
                                    avg_cpu: float, avg_memory: float) -> str:
        """Calculate overall performance rating for embedded system."""
        score = 0
        
        # FPS scoring (target: 25-30 FPS)
        if avg_fps >= 25:
            score += 25
        elif avg_fps >= 15:
            score += 15
        elif avg_fps >= 10:
            score += 10
        
        # Processing time scoring (target: <40ms)
        if avg_processing_time < 0.04:
            score += 25
        elif avg_processing_time < 0.06:
            score += 20
        elif avg_processing_time < 0.1:
            score += 15
        
        # CPU usage scoring (target: <70%)
        if avg_cpu < 50:
            score += 25
        elif avg_cpu < 70:
            score += 20
        elif avg_cpu < 85:
            score += 10
        
        # Memory usage scoring (target: <200MB)
        if avg_memory < 150:
            score += 25
        elif avg_memory < 200:
            score += 20
        elif avg_memory < 250:
            score += 10
        
        if score >= 80:
            return 'excellent'
        elif score >= 60:
            return 'good'
        elif score >= 40:
            return 'fair'
        else:
            return 'poor'
    
    def _calculate_memory_efficiency(self, avg_memory: float) -> float:
        """Calculate memory efficiency score."""
        # Assuming 256MB total system memory
        return max(0, 1.0 - (avg_memory / 256.0))
    
    def _calculate_cpu_efficiency(self, avg_cpu: float) -> float:
        """Calculate CPU efficiency score."""
        return max(0, 1.0 - (avg_cpu / 100.0))
    
    def _calculate_embedded_optimization_score(self, performance_data: List[Dict]) -> float:
        """Calculate optimization score for embedded deployment."""
        if not performance_data:
            return 0.0
        
        # Multiple factors for embedded optimization
        factors = []
        
        # FPS stability
        fps_values = [p.get('fps', 0.0) for p in performance_data]
        fps_stability = 1.0 - (np.std(fps_values) / max(np.mean(fps_values), 1.0))
        factors.append(fps_stability)
        
        # Memory consistency
        memory_values = [p.get('memory_usage', 0.0) for p in performance_data]
        memory_consistency = 1.0 - (np.std(memory_values) / max(np.mean(memory_values), 1.0))
        factors.append(memory_consistency)
        
        # Processing time consistency
        proc_times = [p.get('processing_time', 0.0) for p in performance_data]
        proc_consistency = 1.0 - (np.std(proc_times) / max(np.mean(proc_times), 0.001))
        factors.append(proc_consistency)
        
        return np.mean(factors)


# Example usage
if __name__ == "__main__":
    # Create report generator
    report_gen = ReportGenerator()
    
    # Create sample data
    sample_detection_history = []
    sample_performance_data = []
    
    for i in range(100):
        # Sample detection data
        detection = {
            'pest_detected': i % 5 == 0,
            'class': 'aphid' if i % 5 == 0 else 'healthy',
            'confidence': 0.8 + np.random.random() * 0.2 if i % 5 == 0 else 0.1,
            'severity': 'medium' if i % 5 == 0 else 'none'
        }
        sample_detection_history.append(detection)
        
        # Sample performance data
        performance = {
            'fps': 25.0 + np.random.random() * 10,
            'processing_time': 0.04 + np.random.random() * 0.02,
            'memory_usage': 150.0 + np.random.random() * 50,
            'cpu_usage': 40.0 + np.random.random() * 30
        }
        sample_performance_data.append(performance)
    
    # Generate comprehensive report
    result = report_gen.generate_comprehensive_report(
        sample_detection_history, 
        sample_performance_data
    )
    
    print(f"Report generation result: {result}")
    print(f"Report files generated: {result['report_files']}")
    print(f"Charts created: {len(result['chart_paths'])}")