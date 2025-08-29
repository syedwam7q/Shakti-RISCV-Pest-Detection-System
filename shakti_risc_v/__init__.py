"""
Shakti RISC-V Optimization Module for Pest Detection
===================================================

Complete hardware optimization suite for Shakti E-class RISC-V processor
deployment on Arty A7-35T FPGA board.

Features:
- C language implementation bindings
- Memory management optimization
- Real-time processing constraints
- Fixed-point arithmetic implementation
- Hardware-specific performance tuning
- Embedded system power optimization

Target Platform: Shakti E-class 32-bit RISC-V on Arty A7-35T
Memory Constraint: 256MB DDR3
Processing Target: 10-25 FPS real-time pest detection
Power Budget: <5W
"""

from .core.shakti_optimizer import ShaktiOptimizer
from .core.memory_manager import EmbeddedMemoryManager
from .core.fixed_point_math import FixedPointProcessor
from .hardware.arty_a7_interface import ArtyA7Interface
from .c_implementation.pest_detector_c import CPestDetector

__all__ = [
    'ShaktiOptimizer',
    'EmbeddedMemoryManager', 
    'FixedPointProcessor',
    'ArtyA7Interface',
    'CPestDetector'
]

__version__ = "1.0.0"
__target_platform__ = "Shakti E-class RISC-V"
__board__ = "Arty A7-35T"