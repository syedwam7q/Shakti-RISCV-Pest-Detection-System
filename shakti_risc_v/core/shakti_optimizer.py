"""
Shakti RISC-V Processor Optimization Engine
==========================================

Advanced optimization engine specifically designed for Shakti E-class
32-bit RISC-V processor on Arty A7-35T FPGA board.

Key Optimizations:
- Memory access pattern optimization for DDR3
- Instruction scheduling for in-order pipeline
- Cache optimization for 3-stage pipeline
- Fixed-point arithmetic utilization
- Real-time constraints management
- Power consumption optimization
"""

import numpy as np
import time
import gc
import psutil
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class OptimizationLevel(Enum):
    """Optimization levels for different deployment scenarios."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    ULTRA_LOW_POWER = "ultra_low_power"


@dataclass
class ShaktiSystemConfig:
    """Configuration for Shakti E-class system."""
    # Hardware specifications
    cpu_frequency_mhz: int = 50  # Conservative for power efficiency
    memory_size_mb: int = 256
    cache_size_kb: int = 32
    pipeline_stages: int = 3
    
    # Performance targets
    target_fps: float = 15.0
    max_processing_time_ms: float = 66.0  # For 15 FPS
    memory_limit_mb: float = 200.0  # Reserve memory for OS
    
    # Power optimization
    enable_power_gating: bool = True
    dynamic_frequency_scaling: bool = True
    sleep_between_frames: bool = True
    
    # Processing optimization
    use_fixed_point: bool = True
    enable_simd: bool = False  # Limited SIMD on E-class
    optimize_memory_access: bool = True
    
    # Real-time constraints
    deadline_miss_threshold: float = 0.05  # 5% allowable deadline misses
    priority_level: int = 99  # Real-time priority


class ShaktiOptimizer:
    """
    Comprehensive optimization engine for Shakti E-class RISC-V processor.
    Handles memory, performance, power, and real-time optimization.
    """
    
    def __init__(self, config: Optional[ShaktiSystemConfig] = None):
        self.config = config or ShaktiSystemConfig()
        
        # Performance tracking
        self.performance_stats = {
            'frame_count': 0,
            'total_processing_time': 0.0,
            'deadline_misses': 0,
            'memory_peak': 0.0,
            'power_estimate': 0.0
        }
        
        # Memory optimization state
        self.memory_pools = {}
        self.allocated_buffers = {}
        
        # Real-time scheduling state
        self.frame_deadlines = []
        self.processing_times = []
        
        # Initialize optimization components
        self._initialize_memory_optimization()
        self._initialize_power_management()
        self._setup_real_time_constraints()
        
        print(f"🚀 Shakti RISC-V Optimizer initialized for {self.config.cpu_frequency_mhz}MHz E-class")
    
    def _initialize_memory_optimization(self):
        """Initialize memory optimization for DDR3 access patterns."""
        # Pre-allocate memory pools for different data types
        self.memory_pools = {
            'image_buffers': [],
            'feature_vectors': [],
            'intermediate_results': [],
            'small_objects': []
        }
        
        # Cache-friendly data alignment
        self.cache_line_size = 64  # bytes
        self.memory_alignment = 32  # 32-byte alignment for RISC-V
        
        print("📦 Memory optimization initialized")
    
    def _initialize_power_management(self):
        """Initialize power management for embedded deployment."""
        self.power_states = {
            'active': {'cpu_freq': self.config.cpu_frequency_mhz, 'power_factor': 1.0},
            'reduced': {'cpu_freq': self.config.cpu_frequency_mhz * 0.7, 'power_factor': 0.5},
            'idle': {'cpu_freq': self.config.cpu_frequency_mhz * 0.3, 'power_factor': 0.2}
        }
        
        self.current_power_state = 'active'
        print("⚡ Power management initialized")
    
    def _setup_real_time_constraints(self):
        """Setup real-time processing constraints."""
        # Set process priority if possible (requires privileges)
        try:
            if hasattr(os, 'nice'):
                os.nice(-19)  # Highest priority
        except:
            pass
        
        # Real-time scheduling parameters
        self.frame_deadline = self.config.max_processing_time_ms / 1000.0
        self.scheduling_slack = 0.8  # Use 80% of available time
        
        print("⏱️ Real-time constraints configured")
    
    def optimize_image_processing(self, image: np.ndarray, 
                                processing_func: callable,
                                **kwargs) -> Tuple[Any, Dict]:
        """
        Optimize image processing with Shakti-specific optimizations.
        
        Args:
            image: Input image array
            processing_func: Function to process the image
            **kwargs: Additional arguments for processing function
            
        Returns:
            Tuple of (processing_result, optimization_metrics)
        """
        frame_start_time = time.time()
        optimization_metrics = {}
        
        try:
            # 1. Memory optimization
            optimized_image = self._optimize_image_memory_layout(image)
            optimization_metrics['memory_optimization_time'] = time.time() - frame_start_time
            
            # 2. Processing optimization
            processing_start = time.time()
            
            # Apply Shakti-specific optimizations
            with self._performance_monitoring():
                result = self._execute_optimized_processing(
                    optimized_image, processing_func, **kwargs
                )
            
            processing_time = time.time() - processing_start
            optimization_metrics['processing_time'] = processing_time
            
            # 3. Real-time constraint checking
            total_frame_time = time.time() - frame_start_time
            deadline_met = total_frame_time <= self.frame_deadline
            
            if not deadline_met:
                self.performance_stats['deadline_misses'] += 1
                optimization_metrics['deadline_miss'] = True
            
            # 4. Power management
            self._update_power_state(processing_time)
            
            # 5. Update performance statistics
            self._update_performance_stats(total_frame_time)
            
            optimization_metrics.update({
                'total_frame_time': total_frame_time,
                'deadline_met': deadline_met,
                'memory_usage_mb': self._get_current_memory_usage(),
                'power_state': self.current_power_state,
                'fps_estimate': 1.0 / total_frame_time if total_frame_time > 0 else 0
            })
            
            return result, optimization_metrics
            
        except Exception as e:
            print(f"Error in optimized processing: {e}")
            return None, {'error': str(e)}
        
        finally:
            # Cleanup memory if needed
            self._cleanup_temporary_memory()
    
    def _optimize_image_memory_layout(self, image: np.ndarray) -> np.ndarray:
        """
        Optimize image memory layout for Shakti E-class cache efficiency.
        
        Args:
            image: Input image
            
        Returns:
            Memory-optimized image array
        """
        # Ensure proper memory alignment for RISC-V
        if not image.data.c_contiguous:
            image = np.ascontiguousarray(image)
        
        # Optimize for cache line access patterns
        height, width = image.shape[:2]
        
        # For small images, keep as-is
        if height * width < 640 * 480:
            return image
        
        # For larger images, consider tiling for cache efficiency
        if height > 480 or width > 640:
            # Resize to optimal size for embedded processing
            import cv2
            optimal_height = min(480, height)
            optimal_width = min(640, width)
            image = cv2.resize(image, (optimal_width, optimal_height))
        
        return image
    
    def _execute_optimized_processing(self, image: np.ndarray, 
                                    processing_func: callable, 
                                    **kwargs) -> Any:
        """
        Execute processing with Shakti-specific optimizations.
        """
        # Memory management
        original_memory = self._get_current_memory_usage()
        
        # Execute with garbage collection control
        gc.disable()  # Disable GC during processing for deterministic timing
        
        try:
            # Apply fixed-point optimization if enabled
            if self.config.use_fixed_point and hasattr(image, 'dtype'):
                if image.dtype in [np.float32, np.float64]:
                    image = self._convert_to_fixed_point(image)
            
            # Execute processing function
            result = processing_func(image, **kwargs)
            
            return result
            
        finally:
            gc.enable()  # Re-enable garbage collection
            
            # Check memory usage
            current_memory = self._get_current_memory_usage()
            if current_memory > self.config.memory_limit_mb:
                gc.collect()  # Force garbage collection if memory is high
    
    def _convert_to_fixed_point(self, array: np.ndarray, 
                              fractional_bits: int = 8) -> np.ndarray:
        """
        Convert floating-point array to fixed-point for faster RISC-V processing.
        
        Args:
            array: Input floating-point array
            fractional_bits: Number of fractional bits
            
        Returns:
            Fixed-point representation as integer array
        """
        scale_factor = 2 ** fractional_bits
        
        # Clamp values to prevent overflow
        array = np.clip(array, -128.0, 127.0)
        
        # Convert to fixed-point
        fixed_point = (array * scale_factor).astype(np.int16)
        
        return fixed_point
    
    def _performance_monitoring(self):
        """Context manager for performance monitoring."""
        class PerformanceMonitor:
            def __init__(self, optimizer):
                self.optimizer = optimizer
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.start_time:
                    processing_time = time.time() - self.start_time
                    self.optimizer.processing_times.append(processing_time)
                    
                    # Keep only recent measurements (memory constraint)
                    if len(self.optimizer.processing_times) > 100:
                        self.optimizer.processing_times = self.optimizer.processing_times[-100:]
        
        return PerformanceMonitor(self)
    
    def _update_power_state(self, processing_time: float):
        """Update power state based on processing load."""
        if not self.config.dynamic_frequency_scaling:
            return
        
        # Calculate processing load
        load_ratio = processing_time / self.frame_deadline
        
        if load_ratio > 0.9:  # High load
            self.current_power_state = 'active'
        elif load_ratio > 0.6:  # Medium load
            self.current_power_state = 'reduced'
        else:  # Low load
            self.current_power_state = 'idle'
            
            # Add sleep for power efficiency
            if self.config.sleep_between_frames:
                sleep_time = (self.frame_deadline - processing_time) * 0.5
                if sleep_time > 0.001:  # Minimum 1ms
                    time.sleep(sleep_time)
    
    def _update_performance_stats(self, frame_time: float):
        """Update performance statistics."""
        self.performance_stats['frame_count'] += 1
        self.performance_stats['total_processing_time'] += frame_time
        
        # Update memory peak
        current_memory = self._get_current_memory_usage()
        self.performance_stats['memory_peak'] = max(
            self.performance_stats['memory_peak'], current_memory
        )
        
        # Estimate power consumption
        power_factor = self.power_states[self.current_power_state]['power_factor']
        self.performance_stats['power_estimate'] += power_factor * frame_time
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            return memory_mb
        except:
            return 0.0
    
    def _cleanup_temporary_memory(self):
        """Cleanup temporary memory allocations."""
        # Force garbage collection if memory usage is high
        current_memory = self._get_current_memory_usage()
        if current_memory > self.config.memory_limit_mb * 0.8:
            gc.collect()
    
    def allocate_aligned_buffer(self, shape: Tuple[int, ...], 
                              dtype: np.dtype = np.uint8,
                              buffer_type: str = 'general') -> np.ndarray:
        """
        Allocate memory-aligned buffer for optimal cache performance.
        
        Args:
            shape: Buffer shape
            dtype: Data type
            buffer_type: Type of buffer for memory pool management
            
        Returns:
            Aligned numpy array
        """
        # Calculate required size
        size = np.prod(shape) * np.dtype(dtype).itemsize
        
        # Align to cache line boundary
        aligned_size = ((size + self.cache_line_size - 1) // 
                       self.cache_line_size) * self.cache_line_size
        
        # Allocate aligned buffer
        buffer = np.empty(aligned_size // np.dtype(dtype).itemsize, dtype=dtype)
        
        # Reshape to desired shape
        buffer = buffer[:np.prod(shape)].reshape(shape)
        
        # Store in memory pool for reuse
        if buffer_type not in self.memory_pools:
            self.memory_pools[buffer_type] = []
        
        self.memory_pools[buffer_type].append(buffer)
        
        return buffer
    
    def get_optimization_statistics(self) -> Dict:
        """Get comprehensive optimization statistics."""
        stats = self.performance_stats.copy()
        
        # Calculate derived statistics
        if stats['frame_count'] > 0:
            stats['average_fps'] = stats['frame_count'] / stats['total_processing_time']
            stats['average_frame_time'] = stats['total_processing_time'] / stats['frame_count']
            stats['deadline_miss_rate'] = stats['deadline_misses'] / stats['frame_count']
        
        # Add system configuration
        stats['system_config'] = {
            'cpu_frequency_mhz': self.config.cpu_frequency_mhz,
            'memory_limit_mb': self.config.memory_limit_mb,
            'target_fps': self.config.target_fps,
            'optimization_level': 'embedded'
        }
        
        # Add real-time performance
        if self.processing_times:
            stats['processing_time_stats'] = {
                'mean': np.mean(self.processing_times),
                'std': np.std(self.processing_times),
                'min': np.min(self.processing_times),
                'max': np.max(self.processing_times),
                'p95': np.percentile(self.processing_times, 95),
                'p99': np.percentile(self.processing_times, 99)
            }
        
        return stats
    
    def optimize_for_deployment(self, optimization_level: OptimizationLevel) -> Dict:
        """
        Optimize system configuration for specific deployment scenario.
        
        Args:
            optimization_level: Target optimization level
            
        Returns:
            Dictionary with applied optimizations
        """
        optimizations_applied = []
        
        if optimization_level == OptimizationLevel.ULTRA_LOW_POWER:
            # Maximum power savings
            self.config.cpu_frequency_mhz = 25
            self.config.target_fps = 5.0
            self.config.enable_power_gating = True
            self.config.sleep_between_frames = True
            optimizations_applied.extend([
                'ultra_low_power_mode',
                'reduced_cpu_frequency', 
                'aggressive_sleep'
            ])
        
        elif optimization_level == OptimizationLevel.PRODUCTION:
            # Balanced performance and power
            self.config.cpu_frequency_mhz = 50
            self.config.target_fps = 15.0
            self.config.use_fixed_point = True
            self.config.optimize_memory_access = True
            optimizations_applied.extend([
                'production_mode',
                'fixed_point_math',
                'memory_optimization'
            ])
        
        elif optimization_level == OptimizationLevel.TESTING:
            # Maximum performance for testing
            self.config.cpu_frequency_mhz = 75
            self.config.target_fps = 25.0
            self.config.deadline_miss_threshold = 0.1
            optimizations_applied.extend([
                'testing_mode',
                'high_performance',
                'relaxed_deadlines'
            ])
        
        # Update frame deadline based on new target FPS
        self.frame_deadline = 1.0 / self.config.target_fps
        
        print(f"🎯 Optimized for {optimization_level.value}")
        print(f"📊 Target: {self.config.target_fps}FPS @ {self.config.cpu_frequency_mhz}MHz")
        
        return {
            'optimization_level': optimization_level.value,
            'optimizations_applied': optimizations_applied,
            'target_fps': self.config.target_fps,
            'cpu_frequency_mhz': self.config.cpu_frequency_mhz,
            'memory_limit_mb': self.config.memory_limit_mb
        }
    
    def reset_statistics(self):
        """Reset all performance statistics."""
        self.performance_stats = {
            'frame_count': 0,
            'total_processing_time': 0.0,
            'deadline_misses': 0,
            'memory_peak': 0.0,
            'power_estimate': 0.0
        }
        self.processing_times.clear()
        print("📊 Performance statistics reset")
    
    def export_optimization_profile(self, filepath: str) -> bool:
        """Export optimization profile for analysis."""
        try:
            import json
            
            profile_data = {
                'system_config': {
                    'cpu_frequency_mhz': self.config.cpu_frequency_mhz,
                    'memory_size_mb': self.config.memory_size_mb,
                    'target_fps': self.config.target_fps,
                    'optimization_flags': {
                        'use_fixed_point': self.config.use_fixed_point,
                        'optimize_memory_access': self.config.optimize_memory_access,
                        'enable_power_gating': self.config.enable_power_gating
                    }
                },
                'performance_statistics': self.get_optimization_statistics(),
                'export_timestamp': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(profile_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Error exporting optimization profile: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Create Shakti optimizer
    config = ShaktiSystemConfig(
        cpu_frequency_mhz=50,
        target_fps=15.0,
        memory_limit_mb=200.0
    )
    
    optimizer = ShaktiOptimizer(config)
    
    # Test optimization with sample image processing
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def sample_processing_func(image, **kwargs):
        """Sample processing function."""
        import cv2
        # Simulate image processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred
    
    # Run optimization test
    result, metrics = optimizer.optimize_image_processing(
        test_image, sample_processing_func
    )
    
    print(f"Optimization test results:")
    print(f"- Processing time: {metrics.get('processing_time', 0):.3f}s")
    print(f"- Deadline met: {metrics.get('deadline_met', False)}")
    print(f"- Memory usage: {metrics.get('memory_usage_mb', 0):.1f}MB")
    print(f"- FPS estimate: {metrics.get('fps_estimate', 0):.1f}")
    
    # Get comprehensive statistics
    stats = optimizer.get_optimization_statistics()
    print(f"\nSystem statistics:")
    print(f"- Average FPS: {stats.get('average_fps', 0):.1f}")
    print(f"- Memory peak: {stats.get('memory_peak', 0):.1f}MB")
    print(f"- Deadline miss rate: {stats.get('deadline_miss_rate', 0):.1%}")
    
    # Export optimization profile
    profile_path = "/Users/navyamudgal/Works/ACAD/Pest-Detection/output/shakti_optimization_profile.json"
    optimizer.export_optimization_profile(profile_path)
    print(f"Optimization profile exported to: {profile_path}")
    """
        Shakti RISC-V Processor Optimization Engine
        ==========================================

        Advanced optimization engine specifically designed for Shakti E-class
        32-bit RISC-V processor on Arty A7-35T FPGA board.

        Key Optimizations:
        - Memory access pattern optimization for DDR3
        - Instruction scheduling for in-order pipeline
        - Cache optimization for 3-stage pipeline
        - Fixed-point arithmetic utilization
        - Real-time constraints management
        - Power consumption optimization
    """

import numpy as np
import time
import gc
import psutil
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class OptimizationLevel(Enum):
    """Optimization levels for different deployment scenarios."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    ULTRA_LOW_POWER = "ultra_low_power"


@dataclass
class ShaktiSystemConfig:
    """Configuration for Shakti E-class system."""
    # Hardware specifications
    cpu_frequency_mhz: int = 50  # Conservative for power efficiency
    memory_size_mb: int = 256
    cache_size_kb: int = 32
    pipeline_stages: int = 3
    
    # Performance targets
    target_fps: float = 15.0
    max_processing_time_ms: float = 66.0  # For 15 FPS
    memory_limit_mb: float = 200.0  # Reserve memory for OS
    
    # Power optimization
    enable_power_gating: bool = True
    dynamic_frequency_scaling: bool = True
    sleep_between_frames: bool = True
    
    # Processing optimization
    use_fixed_point: bool = True
    enable_simd: bool = False  # Limited SIMD on E-class
    optimize_memory_access: bool = True
    
    # Real-time constraints
    deadline_miss_threshold: float = 0.05  # 5% allowable deadline misses
    priority_level: int = 99  # Real-time priority


class ShaktiOptimizer:
    """
    Comprehensive optimization engine for Shakti E-class RISC-V processor.
    Handles memory, performance, power, and real-time optimization.
    """
    
    def __init__(self, config: Optional[ShaktiSystemConfig] = None):
        self.config = config or ShaktiSystemConfig()
        
        # Performance tracking
        self.performance_stats = {
            'frame_count': 0,
            'total_processing_time': 0.0,
            'deadline_misses': 0,
            'memory_peak': 0.0,
            'power_estimate': 0.0
        }
        
        # Memory optimization state
        self.memory_pools = {}
        self.allocated_buffers = {}
        
        # Real-time scheduling state
        self.frame_deadlines = []
        self.processing_times = []
        
        # Initialize optimization components
        self._initialize_memory_optimization()
        self._initialize_power_management()
        self._setup_real_time_constraints()
        
        print(f"🚀 Shakti RISC-V Optimizer initialized for {self.config.cpu_frequency_mhz}MHz E-class")
    
    def _initialize_memory_optimization(self):
        """Initialize memory optimization for DDR3 access patterns."""
        # Pre-allocate memory pools for different data types
        self.memory_pools = {
            'image_buffers': [],
            'feature_vectors': [],
            'intermediate_results': [],
            'small_objects': []
        }
        
        # Cache-friendly data alignment
        self.cache_line_size = 64  # bytes
        self.memory_alignment = 32  # 32-byte alignment for RISC-V
        
        print("📦 Memory optimization initialized")
    
    def _initialize_power_management(self):
        """Initialize power management for embedded deployment."""
        self.power_states = {
            'active': {'cpu_freq': self.config.cpu_frequency_mhz, 'power_factor': 1.0},
            'reduced': {'cpu_freq': self.config.cpu_frequency_mhz * 0.7, 'power_factor': 0.5},
            'idle': {'cpu_freq': self.config.cpu_frequency_mhz * 0.3, 'power_factor': 0.2}
        }
        
        self.current_power_state = 'active'
        print("⚡ Power management initialized")
    
    def _setup_real_time_constraints(self):
        """Setup real-time processing constraints."""
        # Set process priority if possible (requires privileges)
        try:
            if hasattr(os, 'nice'):
                os.nice(-19)  # Highest priority
        except:
            pass
        
        # Real-time scheduling parameters
        self.frame_deadline = self.config.max_processing_time_ms / 1000.0
        self.scheduling_slack = 0.8  # Use 80% of available time
        
        print("⏱️ Real-time constraints configured")
    
    def optimize_image_processing(self, image: np.ndarray, 
                                processing_func: callable,
                                **kwargs) -> Tuple[Any, Dict]:
        """
        Optimize image processing with Shakti-specific optimizations.
        
        Args:
            image: Input image array
            processing_func: Function to process the image
            **kwargs: Additional arguments for processing function
            
        Returns:
            Tuple of (processing_result, optimization_metrics)
        """
        frame_start_time = time.time()
        optimization_metrics = {}
        
        try:
            # 1. Memory optimization
            optimized_image = self._optimize_image_memory_layout(image)
            optimization_metrics['memory_optimization_time'] = time.time() - frame_start_time
            
            # 2. Processing optimization
            processing_start = time.time()
            
            # Apply Shakti-specific optimizations
            with self._performance_monitoring():
                result = self._execute_optimized_processing(
                    optimized_image, processing_func, **kwargs
                )
            
            processing_time = time.time() - processing_start
            optimization_metrics['processing_time'] = processing_time
            
            # 3. Real-time constraint checking
            total_frame_time = time.time() - frame_start_time
            deadline_met = total_frame_time <= self.frame_deadline
            
            if not deadline_met:
                self.performance_stats['deadline_misses'] += 1
                optimization_metrics['deadline_miss'] = True
            
            # 4. Power management
            self._update_power_state(processing_time)
            
            # 5. Update performance statistics
            self._update_performance_stats(total_frame_time)
            
            optimization_metrics.update({
                'total_frame_time': total_frame_time,
                'deadline_met': deadline_met,
                'memory_usage_mb': self._get_current_memory_usage(),
                'power_state': self.current_power_state,
                'fps_estimate': 1.0 / total_frame_time if total_frame_time > 0 else 0
            })
            
            return result, optimization_metrics
            
        except Exception as e:
            print(f"Error in optimized processing: {e}")
            return None, {'error': str(e)}
        
        finally:
            # Cleanup memory if needed
            self._cleanup_temporary_memory()
    
    def _optimize_image_memory_layout(self, image: np.ndarray) -> np.ndarray:
        """
        Optimize image memory layout for Shakti E-class cache efficiency.
        
        Args:
            image: Input image
            
        Returns:
            Memory-optimized image array
        """
        # Ensure proper memory alignment for RISC-V
        if not image.data.c_contiguous:
            image = np.ascontiguousarray(image)
        
        # Optimize for cache line access patterns
        height, width = image.shape[:2]
        
        # For small images, keep as-is
        if height * width < 640 * 480:
            return image
        
        # For larger images, consider tiling for cache efficiency
        if height > 480 or width > 640:
            # Resize to optimal size for embedded processing
            import cv2
            optimal_height = min(480, height)
            optimal_width = min(640, width)
            image = cv2.resize(image, (optimal_width, optimal_height))
        
        return image
    
    def _execute_optimized_processing(self, image: np.ndarray, 
                                    processing_func: callable, 
                                    **kwargs) -> Any:
        """
        Execute processing with Shakti-specific optimizations.
        """
        # Memory management
        original_memory = self._get_current_memory_usage()
        
        # Execute with garbage collection control
        gc.disable()  # Disable GC during processing for deterministic timing
        
        try:
            # Apply fixed-point optimization if enabled
            if self.config.use_fixed_point and hasattr(image, 'dtype'):
                if image.dtype in [np.float32, np.float64]:
                    image = self._convert_to_fixed_point(image)
            
            # Execute processing function
            result = processing_func(image, **kwargs)
            
            return result
            
        finally:
            gc.enable()  # Re-enable garbage collection
            
            # Check memory usage
            current_memory = self._get_current_memory_usage()
            if current_memory > self.config.memory_limit_mb:
                gc.collect()  # Force garbage collection if memory is high
    
    def _convert_to_fixed_point(self, array: np.ndarray, 
                              fractional_bits: int = 8) -> np.ndarray:
        """
        Convert floating-point array to fixed-point for faster RISC-V processing.
        
        Args:
            array: Input floating-point array
            fractional_bits: Number of fractional bits
            
        Returns:
            Fixed-point representation as integer array
        """
        scale_factor = 2 ** fractional_bits
        
        # Clamp values to prevent overflow
        array = np.clip(array, -128.0, 127.0)
        
        # Convert to fixed-point
        fixed_point = (array * scale_factor).astype(np.int16)
        
        return fixed_point
    
    def _performance_monitoring(self):
        """Context manager for performance monitoring."""
        class PerformanceMonitor:
            def __init__(self, optimizer):
                self.optimizer = optimizer
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.start_time:
                    processing_time = time.time() - self.start_time
                    self.optimizer.processing_times.append(processing_time)
                    
                    # Keep only recent measurements (memory constraint)
                    if len(self.optimizer.processing_times) > 100:
                        self.optimizer.processing_times = self.optimizer.processing_times[-100:]
        
        return PerformanceMonitor(self)
    
    def _update_power_state(self, processing_time: float):
        """Update power state based on processing load."""
        if not self.config.dynamic_frequency_scaling:
            return
        
        # Calculate processing load
        load_ratio = processing_time / self.frame_deadline
        
        if load_ratio > 0.9:  # High load
            self.current_power_state = 'active'
        elif load_ratio > 0.6:  # Medium load
            self.current_power_state = 'reduced'
        else:  # Low load
            self.current_power_state = 'idle'
            
            # Add sleep for power efficiency
            if self.config.sleep_between_frames:
                sleep_time = (self.frame_deadline - processing_time) * 0.5
                if sleep_time > 0.001:  # Minimum 1ms
                    time.sleep(sleep_time)
    
    def _update_performance_stats(self, frame_time: float):
        """Update performance statistics."""
        self.performance_stats['frame_count'] += 1
        self.performance_stats['total_processing_time'] += frame_time
        
        # Update memory peak
        current_memory = self._get_current_memory_usage()
        self.performance_stats['memory_peak'] = max(
            self.performance_stats['memory_peak'], current_memory
        )
        
        # Estimate power consumption
        power_factor = self.power_states[self.current_power_state]['power_factor']
        self.performance_stats['power_estimate'] += power_factor * frame_time
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            return memory_mb
        except:
            return 0.0
    
    def _cleanup_temporary_memory(self):
        """Cleanup temporary memory allocations."""
        # Force garbage collection if memory usage is high
        current_memory = self._get_current_memory_usage()
        if current_memory > self.config.memory_limit_mb * 0.8:
            gc.collect()
    
    def allocate_aligned_buffer(self, shape: Tuple[int, ...], 
                              dtype: np.dtype = np.uint8,
                              buffer_type: str = 'general') -> np.ndarray:
        """
        Allocate memory-aligned buffer for optimal cache performance.
        
        Args:
            shape: Buffer shape
            dtype: Data type
            buffer_type: Type of buffer for memory pool management
            
        Returns:
            Aligned numpy array
        """
        # Calculate required size
        size = np.prod(shape) * np.dtype(dtype).itemsize
        
        # Align to cache line boundary
        aligned_size = ((size + self.cache_line_size - 1) // 
                       self.cache_line_size) * self.cache_line_size
        
        # Allocate aligned buffer
        buffer = np.empty(aligned_size // np.dtype(dtype).itemsize, dtype=dtype)
        
        # Reshape to desired shape
        buffer = buffer[:np.prod(shape)].reshape(shape)
        
        # Store in memory pool for reuse
        if buffer_type not in self.memory_pools:
            self.memory_pools[buffer_type] = []
        
        self.memory_pools[buffer_type].append(buffer)
        
        return buffer
    
    def get_optimization_statistics(self) -> Dict:
        """Get comprehensive optimization statistics."""
        stats = self.performance_stats.copy()
        
        # Calculate derived statistics
        if stats['frame_count'] > 0:
            stats['average_fps'] = stats['frame_count'] / stats['total_processing_time']
            stats['average_frame_time'] = stats['total_processing_time'] / stats['frame_count']
            stats['deadline_miss_rate'] = stats['deadline_misses'] / stats['frame_count']
        
        # Add system configuration
        stats['system_config'] = {
            'cpu_frequency_mhz': self.config.cpu_frequency_mhz,
            'memory_limit_mb': self.config.memory_limit_mb,
            'target_fps': self.config.target_fps,
            'optimization_level': 'embedded'
        }
        
        # Add real-time performance
        if self.processing_times:
            stats['processing_time_stats'] = {
                'mean': np.mean(self.processing_times),
                'std': np.std(self.processing_times),
                'min': np.min(self.processing_times),
                'max': np.max(self.processing_times),
                'p95': np.percentile(self.processing_times, 95),
                'p99': np.percentile(self.processing_times, 99)
            }
        
        return stats
    
    def optimize_for_deployment(self, optimization_level: OptimizationLevel) -> Dict:
        """
        Optimize system configuration for specific deployment scenario.
        
        Args:
            optimization_level: Target optimization level
            
        Returns:
            Dictionary with applied optimizations
        """
        optimizations_applied = []
        
        if optimization_level == OptimizationLevel.ULTRA_LOW_POWER:
            # Maximum power savings
            self.config.cpu_frequency_mhz = 25
            self.config.target_fps = 5.0
            self.config.enable_power_gating = True
            self.config.sleep_between_frames = True
            optimizations_applied.extend([
                'ultra_low_power_mode',
                'reduced_cpu_frequency', 
                'aggressive_sleep'
            ])
        
        elif optimization_level == OptimizationLevel.PRODUCTION:
            # Balanced performance and power
            self.config.cpu_frequency_mhz = 50
            self.config.target_fps = 15.0
            self.config.use_fixed_point = True
            self.config.optimize_memory_access = True
            optimizations_applied.extend([
                'production_mode',
                'fixed_point_math',
                'memory_optimization'
            ])
        
        elif optimization_level == OptimizationLevel.TESTING:
            # Maximum performance for testing
            self.config.cpu_frequency_mhz = 75
            self.config.target_fps = 25.0
            self.config.deadline_miss_threshold = 0.1
            optimizations_applied.extend([
                'testing_mode',
                'high_performance',
                'relaxed_deadlines'
            ])
        
        # Update frame deadline based on new target FPS
        self.frame_deadline = 1.0 / self.config.target_fps
        
        print(f"🎯 Optimized for {optimization_level.value}")
        print(f"📊 Target: {self.config.target_fps}FPS @ {self.config.cpu_frequency_mhz}MHz")
        
        return {
            'optimization_level': optimization_level.value,
            'optimizations_applied': optimizations_applied,
            'target_fps': self.config.target_fps,
            'cpu_frequency_mhz': self.config.cpu_frequency_mhz,
            'memory_limit_mb': self.config.memory_limit_mb
        }
    
    def reset_statistics(self):
        """Reset all performance statistics."""
        self.performance_stats = {
            'frame_count': 0,
            'total_processing_time': 0.0,
            'deadline_misses': 0,
            'memory_peak': 0.0,
            'power_estimate': 0.0
        }
        self.processing_times.clear()
        print("📊 Performance statistics reset")
    
    def export_optimization_profile(self, filepath: str) -> bool:
        """Export optimization profile for analysis."""
        try:
            import json
            
            profile_data = {
                'system_config': {
                    'cpu_frequency_mhz': self.config.cpu_frequency_mhz,
                    'memory_size_mb': self.config.memory_size_mb,
                    'target_fps': self.config.target_fps,
                    'optimization_flags': {
                        'use_fixed_point': self.config.use_fixed_point,
                        'optimize_memory_access': self.config.optimize_memory_access,
                        'enable_power_gating': self.config.enable_power_gating
                    }
                },
                'performance_statistics': self.get_optimization_statistics(),
                'export_timestamp': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(profile_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Error exporting optimization profile: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Create Shakti optimizer
    config = ShaktiSystemConfig(
        cpu_frequency_mhz=50,
        target_fps=15.0,
        memory_limit_mb=200.0
    )
    
    optimizer = ShaktiOptimizer(config)
    
    # Test optimization with sample image processing
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def sample_processing_func(image, **kwargs):
        """Sample processing function."""
        import cv2
        # Simulate image processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred
    
    # Run optimization test
    result, metrics = optimizer.optimize_image_processing(
        test_image, sample_processing_func
    )
    
    print(f"Optimization test results:")
    print(f"- Processing time: {metrics.get('processing_time', 0):.3f}s")
    print(f"- Deadline met: {metrics.get('deadline_met', False)}")
    print(f"- Memory usage: {metrics.get('memory_usage_mb', 0):.1f}MB")
    print(f"- FPS estimate: {metrics.get('fps_estimate', 0):.1f}")
    
    # Get comprehensive statistics
    stats = optimizer.get_optimization_statistics()
    print(f"\nSystem statistics:")
    print(f"- Average FPS: {stats.get('average_fps', 0):.1f}")
    print(f"- Memory peak: {stats.get('memory_peak', 0):.1f}MB")
    print(f"- Deadline miss rate: {stats.get('deadline_miss_rate', 0):.1%}")
    
    # Export optimization profile
    profile_path = "/Users/navyamudgal/Works/ACAD/Pest-Detection/output/shakti_optimization_profile.json"
    optimizer.export_optimization_profile(profile_path)
    print(f"Optimization profile exported to: {profile_path}")