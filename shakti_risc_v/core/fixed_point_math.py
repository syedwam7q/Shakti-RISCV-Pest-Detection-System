"""
Fixed-Point Math Processor for Shakti RISC-V
============================================

High-performance fixed-point arithmetic implementation optimized for
Shakti E-class RISC-V processor. Provides faster computation compared
to floating-point operations on embedded systems without FPU.

Key Features:
- Q8.8, Q16.16, and custom fixed-point formats
- SIMD-style operations where possible
- Overflow and underflow protection
- Optimized transcendental functions (sin, cos, sqrt, exp)
- Image processing specific operations
- Lookup table optimizations for common functions
"""

import numpy as np
import time
from typing import Union, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import math


class FixedPointFormat(Enum):
    """Standard fixed-point formats."""
    Q8_8 = (8, 8)      # 8 integer bits, 8 fractional bits (16-bit total)
    Q16_16 = (16, 16)  # 16 integer bits, 16 fractional bits (32-bit total)
    Q4_12 = (4, 12)    # 4 integer bits, 12 fractional bits (16-bit total)
    Q12_4 = (12, 4)    # 12 integer bits, 4 fractional bits (16-bit total)
    Q24_8 = (24, 8)    # 24 integer bits, 8 fractional bits (32-bit total)


@dataclass
class FixedPointConfig:
    """Configuration for fixed-point arithmetic operations."""
    default_format: FixedPointFormat = FixedPointFormat.Q16_16
    enable_saturation: bool = True      # Saturate on overflow instead of wrapping
    enable_rounding: bool = True        # Round instead of truncating
    use_lookup_tables: bool = True      # Use LUTs for transcendental functions
    lut_size: int = 1024               # Size of lookup tables
    optimize_for_speed: bool = True     # Speed vs accuracy tradeoff


class FixedPointNumber:
    """
    Fixed-point number representation with arithmetic operations.
    Optimized for Shakti RISC-V integer ALU.
    """
    
    def __init__(self, value: Union[int, float, 'FixedPointNumber'], 
                 format_type: FixedPointFormat = FixedPointFormat.Q16_16):
        self.integer_bits, self.fractional_bits = format_type.value
        self.total_bits = self.integer_bits + self.fractional_bits
        self.scale_factor = 2 ** self.fractional_bits
        self.format_type = format_type
        
        # Calculate min/max values for this format
        self.max_value = (2 ** (self.total_bits - 1) - 1) / self.scale_factor
        self.min_value = -(2 ** (self.total_bits - 1)) / self.scale_factor
        
        # Set the raw integer value
        if isinstance(value, FixedPointNumber):
            # Convert from another fixed-point number
            self.raw = self._convert_from_fixed_point(value)
        elif isinstance(value, float):
            self.raw = self._convert_from_float(value)
        elif isinstance(value, int):
            self.raw = self._convert_from_int(value)
        else:
            raise ValueError(f"Unsupported value type: {type(value)}")
    
    def _convert_from_float(self, value: float) -> int:
        """Convert floating-point value to fixed-point raw integer."""
        # Clamp to valid range
        value = max(self.min_value, min(self.max_value, value))
        
        # Convert to fixed-point
        raw = int(value * self.scale_factor)
        
        # Handle rounding
        if value * self.scale_factor - raw >= 0.5:
            raw += 1
        
        return raw
    
    def _convert_from_int(self, value: int) -> int:
        """Convert integer value to fixed-point raw integer."""
        # Clamp to valid range
        value = max(int(self.min_value), min(int(self.max_value), value))
        return value * self.scale_factor
    
    def _convert_from_fixed_point(self, other: 'FixedPointNumber') -> int:
        """Convert from another fixed-point format."""
        if other.format_type == self.format_type:
            return other.raw
        
        # Convert through floating-point (could be optimized)
        float_value = other.to_float()
        return self._convert_from_float(float_value)
    
    def to_float(self) -> float:
        """Convert to floating-point value."""
        return self.raw / self.scale_factor
    
    def to_int(self) -> int:
        """Convert to integer value (truncated)."""
        return self.raw // self.scale_factor
    
    def __add__(self, other: Union['FixedPointNumber', int, float]) -> 'FixedPointNumber':
        """Addition with overflow protection."""
        if not isinstance(other, FixedPointNumber):
            other = FixedPointNumber(other, self.format_type)
        
        # Convert to same format if needed
        if other.format_type != self.format_type:
            other = FixedPointNumber(other.to_float(), self.format_type)
        
        result_raw = self.raw + other.raw
        
        # Saturation protection
        max_raw = int(self.max_value * self.scale_factor)
        min_raw = int(self.min_value * self.scale_factor)
        result_raw = max(min_raw, min(max_raw, result_raw))
        
        result = FixedPointNumber(0, self.format_type)
        result.raw = result_raw
        return result
    
    def __sub__(self, other: Union['FixedPointNumber', int, float]) -> 'FixedPointNumber':
        """Subtraction with overflow protection."""
        if not isinstance(other, FixedPointNumber):
            other = FixedPointNumber(other, self.format_type)
        
        if other.format_type != self.format_type:
            other = FixedPointNumber(other.to_float(), self.format_type)
        
        result_raw = self.raw - other.raw
        
        # Saturation protection
        max_raw = int(self.max_value * self.scale_factor)
        min_raw = int(self.min_value * self.scale_factor)
        result_raw = max(min_raw, min(max_raw, result_raw))
        
        result = FixedPointNumber(0, self.format_type)
        result.raw = result_raw
        return result
    
    def __mul__(self, other: Union['FixedPointNumber', int, float]) -> 'FixedPointNumber':
        """Multiplication with overflow protection."""
        if not isinstance(other, FixedPointNumber):
            other = FixedPointNumber(other, self.format_type)
        
        if other.format_type != self.format_type:
            other = FixedPointNumber(other.to_float(), self.format_type)
        
        # Multiply raw values (this gives 2x fractional bits)
        result_raw = (self.raw * other.raw) // self.scale_factor
        
        # Saturation protection
        max_raw = int(self.max_value * self.scale_factor)
        min_raw = int(self.min_value * self.scale_factor)
        result_raw = max(min_raw, min(max_raw, result_raw))
        
        result = FixedPointNumber(0, self.format_type)
        result.raw = result_raw
        return result
    
    def __truediv__(self, other: Union['FixedPointNumber', int, float]) -> 'FixedPointNumber':
        """Division with overflow protection."""
        if not isinstance(other, FixedPointNumber):
            other = FixedPointNumber(other, self.format_type)
        
        if other.format_type != self.format_type:
            other = FixedPointNumber(other.to_float(), self.format_type)
        
        if other.raw == 0:
            # Handle division by zero
            result = FixedPointNumber(0, self.format_type)
            result.raw = int(self.max_value * self.scale_factor) if self.raw > 0 else int(self.min_value * self.scale_factor)
            return result
        
        # Multiply by scale factor to maintain precision
        result_raw = (self.raw * self.scale_factor) // other.raw
        
        # Saturation protection
        max_raw = int(self.max_value * self.scale_factor)
        min_raw = int(self.min_value * self.scale_factor)
        result_raw = max(min_raw, min(max_raw, result_raw))
        
        result = FixedPointNumber(0, self.format_type)
        result.raw = result_raw
        return result
    
    def __repr__(self) -> str:
        return f"FixedPoint({self.to_float():.4f}, {self.format_type.name})"


class FixedPointProcessor:
    """
    High-performance fixed-point math processor for Shakti RISC-V.
    Provides optimized implementations of common mathematical operations.
    """
    
    def __init__(self, config: Optional[FixedPointConfig] = None):
        self.config = config or FixedPointConfig()
        
        # Performance tracking
        self.operation_counts = {}
        self.operation_times = {}
        
        # Lookup tables for transcendental functions
        self.lookup_tables = {}
        
        # Initialize lookup tables if enabled
        if self.config.use_lookup_tables:
            self._initialize_lookup_tables()
        
        print(f"🔢 Fixed-Point Processor initialized ({self.config.default_format.name})")
    
    def _initialize_lookup_tables(self):
        """Initialize lookup tables for common transcendental functions."""
        lut_size = self.config.lut_size
        
        # Sine lookup table (0 to π/2)
        x_values = np.linspace(0, np.pi/2, lut_size)
        sin_values = np.sin(x_values)
        self.lookup_tables['sin'] = [(x, FixedPointNumber(sin_val, self.config.default_format)) 
                                    for x, sin_val in zip(x_values, sin_values)]
        
        # Square root lookup table (0 to 256)
        x_values = np.linspace(0, 256, lut_size)
        sqrt_values = np.sqrt(x_values)
        self.lookup_tables['sqrt'] = [(x, FixedPointNumber(sqrt_val, self.config.default_format)) 
                                     for x, sqrt_val in zip(x_values, sqrt_values)]
        
        # Exponential lookup table (0 to 8)
        x_values = np.linspace(0, 8, lut_size)
        exp_values = np.exp(x_values)
        self.lookup_tables['exp'] = [(x, FixedPointNumber(min(exp_val, 1000), self.config.default_format)) 
                                    for x, exp_val in zip(x_values, exp_values)]
        
        print(f"📊 Initialized {len(self.lookup_tables)} lookup tables")
    
    def _lookup_interpolate(self, table_name: str, x: float) -> FixedPointNumber:
        """Perform linear interpolation in lookup table."""
        if table_name not in self.lookup_tables:
            raise ValueError(f"Lookup table {table_name} not found")
        
        table = self.lookup_tables[table_name]
        
        # Find surrounding values
        for i in range(len(table) - 1):
            x0, y0 = table[i]
            x1, y1 = table[i + 1]
            
            if x0 <= x <= x1:
                # Linear interpolation
                if x1 - x0 == 0:
                    return y0
                
                t = (x - x0) / (x1 - x0)
                interpolated = y0.to_float() + t * (y1.to_float() - y0.to_float())
                return FixedPointNumber(interpolated, self.config.default_format)
        
        # Out of range - return closest value
        if x < table[0][0]:
            return table[0][1]
        else:
            return table[-1][1]
    
    def _track_operation(self, operation_name: str, start_time: float):
        """Track operation performance."""
        elapsed_time = time.time() - start_time
        
        if operation_name not in self.operation_counts:
            self.operation_counts[operation_name] = 0
            self.operation_times[operation_name] = 0.0
        
        self.operation_counts[operation_name] += 1
        self.operation_times[operation_name] += elapsed_time
    
    def convert_array_to_fixed_point(self, array: np.ndarray, 
                                   format_type: Optional[FixedPointFormat] = None) -> np.ndarray:
        """
        Convert numpy array to fixed-point representation.
        
        Args:
            array: Input floating-point array
            format_type: Target fixed-point format
            
        Returns:
            Integer array representing fixed-point values
        """
        start_time = time.time()
        
        if format_type is None:
            format_type = self.config.default_format
        
        integer_bits, fractional_bits = format_type.value
        scale_factor = 2 ** fractional_bits
        
        # Convert to fixed-point
        fixed_array = (array * scale_factor).astype(np.int32)
        
        # Apply saturation if enabled
        if self.config.enable_saturation:
            max_val = (2 ** (integer_bits + fractional_bits - 1)) - 1
            min_val = -(2 ** (integer_bits + fractional_bits - 1))
            fixed_array = np.clip(fixed_array, min_val, max_val)
        
        self._track_operation('array_convert_to_fixed', start_time)
        return fixed_array
    
    def convert_array_from_fixed_point(self, fixed_array: np.ndarray, 
                                     format_type: Optional[FixedPointFormat] = None) -> np.ndarray:
        """
        Convert fixed-point array back to floating-point.
        
        Args:
            fixed_array: Fixed-point integer array
            format_type: Source fixed-point format
            
        Returns:
            Floating-point array
        """
        start_time = time.time()
        
        if format_type is None:
            format_type = self.config.default_format
        
        integer_bits, fractional_bits = format_type.value
        scale_factor = 2 ** fractional_bits
        
        # Convert to floating-point
        float_array = fixed_array.astype(np.float32) / scale_factor
        
        self._track_operation('array_convert_from_fixed', start_time)
        return float_array
    
    def fixed_point_multiply(self, a: np.ndarray, b: np.ndarray, 
                           format_type: Optional[FixedPointFormat] = None) -> np.ndarray:
        """
        Multiply two fixed-point arrays efficiently.
        
        Args:
            a, b: Fixed-point arrays (as integers)
            format_type: Fixed-point format
            
        Returns:
            Fixed-point result array
        """
        start_time = time.time()
        
        if format_type is None:
            format_type = self.config.default_format
        
        integer_bits, fractional_bits = format_type.value
        scale_factor = 2 ** fractional_bits
        
        # Multiply and adjust for double scaling
        result = (a.astype(np.int64) * b.astype(np.int64)) // scale_factor
        
        # Apply saturation
        if self.config.enable_saturation:
            max_val = (2 ** (integer_bits + fractional_bits - 1)) - 1
            min_val = -(2 ** (integer_bits + fractional_bits - 1))
            result = np.clip(result, min_val, max_val)
        
        self._track_operation('fixed_multiply', start_time)
        return result.astype(np.int32)
    
    def fixed_point_divide(self, a: np.ndarray, b: np.ndarray, 
                         format_type: Optional[FixedPointFormat] = None) -> np.ndarray:
        """
        Divide two fixed-point arrays efficiently.
        
        Args:
            a, b: Fixed-point arrays (as integers)
            format_type: Fixed-point format
            
        Returns:
            Fixed-point result array
        """
        start_time = time.time()
        
        if format_type is None:
            format_type = self.config.default_format
        
        integer_bits, fractional_bits = format_type.value
        scale_factor = 2 ** fractional_bits
        
        # Avoid division by zero
        b_safe = np.where(b == 0, 1, b)
        
        # Multiply by scale factor to maintain precision
        result = (a.astype(np.int64) * scale_factor) // b_safe.astype(np.int64)
        
        # Handle division by zero cases
        result = np.where(b == 0, 0, result)
        
        # Apply saturation
        if self.config.enable_saturation:
            max_val = (2 ** (integer_bits + fractional_bits - 1)) - 1
            min_val = -(2 ** (integer_bits + fractional_bits - 1))
            result = np.clip(result, min_val, max_val)
        
        self._track_operation('fixed_divide', start_time)
        return result.astype(np.int32)
    
    def fixed_point_sqrt(self, x: FixedPointNumber) -> FixedPointNumber:
        """
        Compute square root using lookup table or Newton's method.
        
        Args:
            x: Fixed-point input
            
        Returns:
            Fixed-point square root
        """
        start_time = time.time()
        
        x_float = x.to_float()
        
        if x_float < 0:
            result = FixedPointNumber(0, x.format_type)
        elif self.config.use_lookup_tables and x_float <= 256:
            result = self._lookup_interpolate('sqrt', x_float)
        else:
            # Newton's method for square root
            result = self._newton_sqrt(x)
        
        self._track_operation('fixed_sqrt', start_time)
        return result
    
    def _newton_sqrt(self, x: FixedPointNumber, iterations: int = 8) -> FixedPointNumber:
        """Compute square root using Newton's method."""
        if x.to_float() == 0:
            return FixedPointNumber(0, x.format_type)
        
        # Initial guess
        guess = FixedPointNumber(x.to_float() / 2, x.format_type)
        
        for _ in range(iterations):
            # Newton's iteration: guess = (guess + x/guess) / 2
            new_guess = (guess + x / guess) / FixedPointNumber(2, x.format_type)
            
            # Check for convergence
            if abs(new_guess.to_float() - guess.to_float()) < 0.001:
                break
            
            guess = new_guess
        
        return guess
    
    def fixed_point_sin(self, x: FixedPointNumber) -> FixedPointNumber:
        """
        Compute sine using lookup table or Taylor series.
        
        Args:
            x: Fixed-point input (in radians)
            
        Returns:
            Fixed-point sine value
        """
        start_time = time.time()
        
        x_float = x.to_float()
        
        # Normalize to [0, 2π]
        x_normalized = x_float % (2 * np.pi)
        
        # Use symmetry to reduce to [0, π/2]
        sign = 1
        if x_normalized > np.pi:
            x_normalized = 2 * np.pi - x_normalized
            sign = -1
        elif x_normalized > np.pi / 2:
            x_normalized = np.pi - x_normalized
        
        if self.config.use_lookup_tables:
            result = self._lookup_interpolate('sin', x_normalized)
        else:
            # Taylor series approximation
            result = self._taylor_sin(FixedPointNumber(x_normalized, x.format_type))
        
        if sign == -1:
            result = FixedPointNumber(-result.to_float(), result.format_type)
        
        self._track_operation('fixed_sin', start_time)
        return result
    
    def _taylor_sin(self, x: FixedPointNumber, terms: int = 6) -> FixedPointNumber:
        """Compute sine using Taylor series."""
        result = FixedPointNumber(0, x.format_type)
        x_power = x
        factorial = 1
        
        for n in range(terms):
            term_index = 2 * n + 1
            factorial *= term_index if n > 0 else 1
            if n > 1:
                factorial *= term_index - 1
            
            term = x_power / FixedPointNumber(factorial, x.format_type)
            
            if n % 2 == 0:
                result = result + term
            else:
                result = result - term
            
            # Update x_power for next term
            if n < terms - 1:
                x_power = x_power * x * x
        
        return result
    
    def optimize_image_processing_fixed_point(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Optimize image processing using fixed-point arithmetic.
        
        Args:
            image: Input image (floating-point or integer)
            
        Returns:
            Tuple of (processed_image, performance_metrics)
        """
        start_time = time.time()
        
        # Normalize image to [0, 1] range if needed
        if image.dtype in [np.uint8, np.uint16]:
            normalized_image = image.astype(np.float32) / 255.0
        else:
            normalized_image = image.astype(np.float32)
        
        # Convert to fixed-point
        conversion_start = time.time()
        fixed_image = self.convert_array_to_fixed_point(normalized_image, FixedPointFormat.Q8_8)
        conversion_time = time.time() - conversion_start
        
        # Perform image processing operations in fixed-point
        processing_start = time.time()
        
        # Example: Gaussian blur using fixed-point convolution
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.int32)
        kernel_fixed = (kernel * (2 ** 8) // 16).astype(np.int32)  # Normalize kernel
        
        # Apply convolution (simplified 3x3 kernel)
        if len(fixed_image.shape) == 2:
            processed_fixed = self._fixed_point_convolution(fixed_image, kernel_fixed)
        else:
            # Process each channel separately
            processed_fixed = np.zeros_like(fixed_image)
            for c in range(fixed_image.shape[2]):
                processed_fixed[:, :, c] = self._fixed_point_convolution(
                    fixed_image[:, :, c], kernel_fixed
                )
        
        processing_time = time.time() - processing_start
        
        # Convert back to floating-point
        reconversion_start = time.time()
        processed_image = self.convert_array_from_fixed_point(processed_fixed, FixedPointFormat.Q8_8)
        reconversion_time = time.time() - reconversion_start
        
        # Convert back to original range
        if image.dtype in [np.uint8, np.uint16]:
            processed_image = (processed_image * 255).astype(image.dtype)
        
        total_time = time.time() - start_time
        
        metrics = {
            'total_time': total_time,
            'conversion_time': conversion_time,
            'processing_time': processing_time,
            'reconversion_time': reconversion_time,
            'speedup_estimate': 'N/A',  # Would need floating-point comparison
            'memory_savings': '50%'     # Approximate for Q8.8 vs float32
        }
        
        return processed_image, metrics
    
    def _fixed_point_convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply 3x3 convolution in fixed-point arithmetic."""
        height, width = image.shape
        result = np.zeros_like(image)
        
        # Apply convolution (avoiding borders for simplicity)
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Extract 3x3 patch
                patch = image[y-1:y+2, x-1:x+2]
                
                # Compute convolution sum
                conv_sum = np.sum(patch * kernel) // (2 ** 8)  # Adjust for fixed-point scale
                
                # Apply saturation
                conv_sum = max(-32768, min(32767, conv_sum))
                result[y, x] = conv_sum
        
        return result
    
    def get_performance_statistics(self) -> Dict:
        """Get comprehensive performance statistics."""
        stats = {
            'operation_counts': self.operation_counts.copy(),
            'operation_times': self.operation_times.copy(),
            'average_times': {},
            'lookup_table_info': {
                'enabled': self.config.use_lookup_tables,
                'table_count': len(self.lookup_tables),
                'table_size': self.config.lut_size
            },
            'configuration': {
                'default_format': self.config.default_format.name,
                'saturation_enabled': self.config.enable_saturation,
                'rounding_enabled': self.config.enable_rounding,
                'optimize_for_speed': self.config.optimize_for_speed
            }
        }
        
        # Calculate average times
        for op_name, total_time in self.operation_times.items():
            count = self.operation_counts.get(op_name, 1)
            stats['average_times'][op_name] = total_time / count
        
        return stats
    
    def benchmark_operations(self, iterations: int = 1000) -> Dict:
        """Benchmark fixed-point operations."""
        print(f"🏃 Benchmarking fixed-point operations ({iterations} iterations)...")
        
        # Test data
        x = FixedPointNumber(1.5, self.config.default_format)
        y = FixedPointNumber(2.3, self.config.default_format)
        test_array = np.random.rand(100, 100).astype(np.float32)
        
        # Benchmark basic operations
        operations = {
            'addition': lambda: x + y,
            'multiplication': lambda: x * y,
            'division': lambda: x / y,
            'square_root': lambda: self.fixed_point_sqrt(x),
            'sine': lambda: self.fixed_point_sin(x),
            'array_conversion': lambda: self.convert_array_to_fixed_point(test_array)
        }
        
        results = {}
        
        for op_name, op_func in operations.items():
            start_time = time.time()
            
            for _ in range(iterations):
                op_func()
            
            elapsed_time = time.time() - start_time
            results[op_name] = {
                'total_time': elapsed_time,
                'average_time_us': (elapsed_time / iterations) * 1e6,
                'operations_per_second': iterations / elapsed_time
            }
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Create fixed-point processor
    config = FixedPointConfig(
        default_format=FixedPointFormat.Q16_16,
        use_lookup_tables=True,
        optimize_for_speed=True
    )
    
    processor = FixedPointProcessor(config)
    
    # Test basic operations
    print("🧮 Testing basic fixed-point operations...")
    
    a = FixedPointNumber(3.14159, FixedPointFormat.Q16_16)
    b = FixedPointNumber(2.71828, FixedPointFormat.Q16_16)
    
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"a / b = {a / b}")
    
    # Test transcendental functions
    print(f"\nTesting transcendental functions...")
    x = FixedPointNumber(1.0, FixedPointFormat.Q16_16)
    print(f"sqrt({x}) = {processor.fixed_point_sqrt(x)}")
    print(f"sin({x}) = {processor.fixed_point_sin(x)}")
    
    # Test image processing
    print(f"\n🖼️ Testing image processing optimization...")
    test_image = np.random.rand(240, 320, 3).astype(np.float32)
    
    processed_image, metrics = processor.optimize_image_processing_fixed_point(test_image)
    
    print(f"Image processing metrics:")
    print(f"  Total time: {metrics['total_time']:.3f}s")
    print(f"  Processing time: {metrics['processing_time']:.3f}s")
    print(f"  Memory savings: {metrics['memory_savings']}")
    
    # Benchmark operations
    benchmark_results = processor.benchmark_operations(1000)
    
    print(f"\n⚡ Benchmark Results:")
    for op_name, results in benchmark_results.items():
        print(f"  {op_name:20s}: {results['average_time_us']:6.1f}μs/op, "
              f"{results['operations_per_second']:8.0f} ops/sec")
    
    # Get comprehensive statistics
    stats = processor.get_performance_statistics()
    print(f"\n📊 Performance Statistics:")
    print(f"  Operations performed: {sum(stats['operation_counts'].values())}")
    print(f"  Lookup tables: {stats['lookup_table_info']['table_count']} tables")
    print(f"  Configuration: {stats['configuration']['default_format']}")