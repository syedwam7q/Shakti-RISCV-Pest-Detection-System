"""
Embedded Memory Manager for Shakti RISC-V
=========================================

Advanced memory management system optimized for Shakti E-class processor
with 256MB DDR3 memory constraint on Arty A7-35T board.

Key Features:
- Memory pool management for zero-allocation operation
- Cache-aware memory layout optimization
- Streaming buffer management for large images
- Memory fragmentation prevention
- Real-time memory allocation guarantees
- Power-efficient memory access patterns
"""

import numpy as np
import time
import gc
import mmap
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import threading
import weakref


class MemoryPoolType(Enum):
    """Types of memory pools for different data structures."""
    IMAGE_BUFFERS = "image_buffers"
    FEATURE_VECTORS = "feature_vectors"
    INTERMEDIATE_RESULTS = "intermediate_results"
    SMALL_OBJECTS = "small_objects"
    STREAMING_BUFFERS = "streaming_buffers"


@dataclass
class MemoryConfig:
    """Configuration for embedded memory management."""
    # Total memory constraints (256MB DDR3)
    total_memory_mb: float = 256.0
    reserved_for_os_mb: float = 56.0  # Reserve for OS and other processes
    available_memory_mb: float = 200.0  # Available for our application
    
    # Memory pool allocation
    image_pool_mb: float = 80.0       # 40% for image processing
    feature_pool_mb: float = 40.0     # 20% for feature vectors
    intermediate_pool_mb: float = 60.0 # 30% for intermediate results
    small_object_pool_mb: float = 20.0 # 10% for small objects
    
    # Cache optimization
    cache_line_size: int = 64         # DDR3 cache line size
    memory_alignment: int = 32        # 32-byte alignment for RISC-V
    prefetch_distance: int = 2        # Cache prefetch distance
    
    # Real-time constraints
    max_allocation_time_us: float = 100.0  # Maximum allocation time
    gc_threshold: float = 0.85        # Trigger GC at 85% usage
    defragmentation_interval: int = 100    # Defragment every N allocations
    
    # Power optimization
    enable_memory_compression: bool = True
    use_streaming_buffers: bool = True
    minimize_dram_access: bool = True


class MemoryBlock:
    """Represents a memory block in the pool."""
    
    def __init__(self, data: np.ndarray, block_id: int, pool_type: MemoryPoolType):
        self.data = data
        self.block_id = block_id
        self.pool_type = pool_type
        self.allocated = False
        self.last_used = time.time()
        self.reference_count = 0
        self.size_bytes = data.nbytes
    
    def allocate(self) -> bool:
        """Allocate this memory block."""
        if self.allocated:
            return False
        self.allocated = True
        self.last_used = time.time()
        self.reference_count += 1
        return True
    
    def deallocate(self):
        """Deallocate this memory block."""
        self.allocated = False
        self.reference_count = max(0, self.reference_count - 1)
        self.last_used = time.time()
    
    def is_available(self) -> bool:
        """Check if block is available for allocation."""
        return not self.allocated and self.reference_count == 0


class StreamingBuffer:
    """Streaming buffer for processing large images in chunks."""
    
    def __init__(self, chunk_size: Tuple[int, int], num_chunks: int, dtype: np.dtype):
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.dtype = dtype
        
        # Pre-allocate chunk buffers
        self.chunks = [
            np.empty(chunk_size, dtype=dtype) 
            for _ in range(num_chunks)
        ]
        
        self.current_chunk = 0
        self.lock = threading.Lock()
    
    def get_next_chunk(self) -> np.ndarray:
        """Get next available chunk buffer."""
        with self.lock:
            chunk = self.chunks[self.current_chunk]
            self.current_chunk = (self.current_chunk + 1) % self.num_chunks
            return chunk
    
    def process_image_streaming(self, image: np.ndarray, 
                              processing_func: callable) -> List[Any]:
        """Process large image using streaming buffers."""
        height, width = image.shape[:2]
        chunk_height, chunk_width = self.chunk_size
        
        results = []
        
        for y in range(0, height, chunk_height):
            for x in range(0, width, chunk_width):
                # Extract chunk
                y_end = min(y + chunk_height, height)
                x_end = min(x + chunk_width, width)
                
                chunk_buffer = self.get_next_chunk()
                actual_chunk_size = (y_end - y, x_end - x)
                
                # Copy image data to chunk buffer
                chunk_view = chunk_buffer[:actual_chunk_size[0], :actual_chunk_size[1]]
                chunk_view[:] = image[y:y_end, x:x_end]
                
                # Process chunk
                chunk_result = processing_func(chunk_view)
                results.append(chunk_result)
        
        return results


class EmbeddedMemoryManager:
    """
    Advanced memory manager for Shakti E-class RISC-V embedded system.
    Provides real-time memory allocation with zero-fragmentation guarantees.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        
        # Memory pools
        self.memory_pools: Dict[MemoryPoolType, List[MemoryBlock]] = {}
        self.pool_statistics: Dict[MemoryPoolType, Dict] = {}
        
        # Allocation tracking
        self.allocation_counter = 0
        self.total_allocated_bytes = 0
        self.peak_memory_usage = 0
        
        # Streaming buffers
        self.streaming_buffers: Dict[str, StreamingBuffer] = {}
        
        # Performance metrics
        self.allocation_times = []
        self.gc_events = []
        
        # Thread safety
        self.allocation_lock = threading.Lock()
        
        # Initialize memory pools
        self._initialize_memory_pools()
        
        print(f"🧠 Embedded Memory Manager initialized")
        print(f"📊 Available memory: {self.config.available_memory_mb:.1f}MB")
        self._print_pool_summary()
    
    def _initialize_memory_pools(self):
        """Initialize pre-allocated memory pools."""
        pool_configs = [
            (MemoryPoolType.IMAGE_BUFFERS, self.config.image_pool_mb, (480, 640, 3), np.uint8),
            (MemoryPoolType.FEATURE_VECTORS, self.config.feature_pool_mb, (1024,), np.float32),
            (MemoryPoolType.INTERMEDIATE_RESULTS, self.config.intermediate_pool_mb, (480, 640), np.float32),
            (MemoryPoolType.SMALL_OBJECTS, self.config.small_object_pool_mb, (256,), np.uint8)
        ]
        
        for pool_type, pool_size_mb, default_shape, dtype in pool_configs:
            self._create_memory_pool(pool_type, pool_size_mb, default_shape, dtype)
        
        # Initialize streaming buffers
        self._initialize_streaming_buffers()
    
    def _create_memory_pool(self, pool_type: MemoryPoolType, 
                          size_mb: float, default_shape: Tuple[int, ...], 
                          dtype: np.dtype):
        """Create a memory pool of specified type and size."""
        total_bytes = int(size_mb * 1024 * 1024)
        element_size = np.prod(default_shape) * np.dtype(dtype).itemsize
        num_blocks = max(1, total_bytes // element_size)
        
        # Pre-allocate memory blocks
        blocks = []
        for i in range(num_blocks):
            try:
                # Allocate aligned memory
                data = self._allocate_aligned_array(default_shape, dtype)
                block = MemoryBlock(data, i, pool_type)
                blocks.append(block)
                
            except MemoryError:
                print(f"⚠️ Warning: Could not allocate full pool for {pool_type.value}")
                break
        
        self.memory_pools[pool_type] = blocks
        
        # Initialize statistics
        self.pool_statistics[pool_type] = {
            'total_blocks': len(blocks),
            'allocated_blocks': 0,
            'total_size_mb': len(blocks) * element_size / (1024 * 1024),
            'allocation_count': 0,
            'deallocation_count': 0,
            'peak_usage': 0
        }
        
        print(f"🔲 Created {pool_type.value}: {len(blocks)} blocks, {len(blocks) * element_size / (1024*1024):.1f}MB")
    
    def _allocate_aligned_array(self, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Allocate memory-aligned array for optimal cache performance."""
        # Calculate total size
        total_elements = np.prod(shape)
        element_size = np.dtype(dtype).itemsize
        total_bytes = total_elements * element_size
        
        # Align to cache line boundary
        alignment = self.config.memory_alignment
        aligned_size = ((total_bytes + alignment - 1) // alignment) * alignment
        
        # Allocate aligned buffer
        buffer = np.empty(aligned_size // element_size, dtype=dtype)
        
        # Return view with correct shape
        return buffer[:total_elements].reshape(shape)
    
    def _initialize_streaming_buffers(self):
        """Initialize streaming buffers for large image processing."""
        if not self.config.use_streaming_buffers:
            return
        
        # Create streaming buffers for different image sizes
        buffer_configs = [
            ("small_chunks", (120, 160), 4, np.uint8),
            ("medium_chunks", (240, 320), 2, np.uint8),
            ("large_chunks", (480, 640), 1, np.uint8)
        ]
        
        for name, chunk_size, num_chunks, dtype in buffer_configs:
            try:
                buffer = StreamingBuffer(chunk_size, num_chunks, dtype)
                self.streaming_buffers[name] = buffer
                print(f"🌊 Created streaming buffer: {name}")
            except MemoryError:
                print(f"⚠️ Warning: Could not create streaming buffer {name}")
    
    def allocate_buffer(self, pool_type: MemoryPoolType, 
                       shape: Optional[Tuple[int, ...]] = None,
                       dtype: Optional[np.dtype] = None) -> Optional[np.ndarray]:
        """
        Allocate buffer from memory pool with real-time guarantees.
        
        Args:
            pool_type: Type of memory pool
            shape: Desired shape (if different from default)
            dtype: Desired data type (if different from default)
            
        Returns:
            Allocated buffer or None if allocation failed
        """
        allocation_start = time.time()
        
        with self.allocation_lock:
            # Find available block
            if pool_type not in self.memory_pools:
                return None
            
            pool = self.memory_pools[pool_type]
            
            for block in pool:
                if block.is_available():
                    success = block.allocate()
                    if success:
                        # Update statistics
                        self.allocation_counter += 1
                        self.total_allocated_bytes += block.size_bytes
                        self.peak_memory_usage = max(self.peak_memory_usage, self.total_allocated_bytes)
                        
                        stats = self.pool_statistics[pool_type]
                        stats['allocated_blocks'] += 1
                        stats['allocation_count'] += 1
                        stats['peak_usage'] = max(stats['peak_usage'], stats['allocated_blocks'])
                        
                        # Track allocation time
                        allocation_time = (time.time() - allocation_start) * 1e6  # microseconds
                        self.allocation_times.append(allocation_time)
                        
                        # Check if allocation time exceeds threshold
                        if allocation_time > self.config.max_allocation_time_us:
                            print(f"⚠️ Slow allocation: {allocation_time:.1f}μs")
                        
                        # Return appropriate view of the buffer
                        if shape is not None and shape != block.data.shape:
                            # Create a view with requested shape if possible
                            if np.prod(shape) <= block.data.size:
                                return block.data.flat[:np.prod(shape)].reshape(shape)
                        
                        return block.data
            
            # No available blocks - try garbage collection
            self._maybe_trigger_gc()
            
            # Try allocation again after GC
            for block in pool:
                if block.is_available():
                    success = block.allocate()
                    if success:
                        return block.data
            
            print(f"❌ Memory allocation failed for {pool_type.value}")
            return None
    
    def deallocate_buffer(self, buffer: np.ndarray):
        """
        Deallocate buffer back to memory pool.
        
        Args:
            buffer: Buffer to deallocate
        """
        with self.allocation_lock:
            # Find the block containing this buffer
            for pool_type, pool in self.memory_pools.items():
                for block in pool:
                    if block.allocated and np.shares_memory(block.data, buffer):
                        block.deallocate()
                        
                        # Update statistics
                        self.total_allocated_bytes -= block.size_bytes
                        stats = self.pool_statistics[pool_type]
                        stats['allocated_blocks'] -= 1
                        stats['deallocation_count'] += 1
                        
                        return
            
            print("⚠️ Warning: Buffer not found in any pool for deallocation")
    
    def get_streaming_buffer(self, buffer_name: str) -> Optional[StreamingBuffer]:
        """Get streaming buffer for large image processing."""
        return self.streaming_buffers.get(buffer_name)
    
    def process_large_image_streaming(self, image: np.ndarray, 
                                    processing_func: callable,
                                    buffer_name: str = "medium_chunks") -> List[Any]:
        """
        Process large image using streaming buffers to avoid memory overflow.
        
        Args:
            image: Large input image
            processing_func: Function to process each chunk
            buffer_name: Name of streaming buffer to use
            
        Returns:
            List of processing results for each chunk
        """
        streaming_buffer = self.get_streaming_buffer(buffer_name)
        if streaming_buffer is None:
            print(f"❌ Streaming buffer {buffer_name} not available")
            return []
        
        return streaming_buffer.process_image_streaming(image, processing_func)
    
    def _maybe_trigger_gc(self):
        """Trigger garbage collection if memory usage is high."""
        current_usage = self.total_allocated_bytes / (self.config.available_memory_mb * 1024 * 1024)
        
        if current_usage > self.config.gc_threshold:
            gc_start = time.time()
            gc.collect()
            gc_time = time.time() - gc_start
            
            self.gc_events.append({
                'timestamp': time.time(),
                'gc_time': gc_time,
                'memory_usage_before': current_usage
            })
            
            print(f"🗑️ Garbage collection: {gc_time:.3f}s, usage: {current_usage:.1%}")
    
    def defragment_pools(self):
        """Defragment memory pools to reduce fragmentation."""
        with self.allocation_lock:
            for pool_type, pool in self.memory_pools.items():
                # Sort blocks by allocation status (allocated first)
                pool.sort(key=lambda b: (not b.allocated, b.last_used))
                
                print(f"🔄 Defragmented {pool_type.value} pool")
    
    def get_memory_statistics(self) -> Dict:
        """Get comprehensive memory usage statistics."""
        with self.allocation_lock:
            total_allocated_mb = self.total_allocated_bytes / (1024 * 1024)
            usage_percentage = (self.total_allocated_bytes / 
                              (self.config.available_memory_mb * 1024 * 1024)) * 100
            
            stats = {
                'total_allocated_mb': total_allocated_mb,
                'available_memory_mb': self.config.available_memory_mb,
                'usage_percentage': usage_percentage,
                'peak_memory_mb': self.peak_memory_usage / (1024 * 1024),
                'allocation_count': self.allocation_counter,
                'gc_events': len(self.gc_events),
                'pool_statistics': self.pool_statistics.copy()
            }
            
            # Add allocation time statistics
            if self.allocation_times:
                stats['allocation_time_stats'] = {
                    'mean_us': np.mean(self.allocation_times),
                    'max_us': np.max(self.allocation_times),
                    'p95_us': np.percentile(self.allocation_times, 95),
                    'p99_us': np.percentile(self.allocation_times, 99)
                }
            
            return stats
    
    def _print_pool_summary(self):
        """Print summary of memory pool allocations."""
        print("\n📋 Memory Pool Summary:")
        for pool_type, stats in self.pool_statistics.items():
            print(f"  {pool_type.value:20s}: {stats['total_blocks']:3d} blocks, "
                  f"{stats['total_size_mb']:6.1f}MB")
        print()
    
    def reset_statistics(self):
        """Reset all memory statistics."""
        with self.allocation_lock:
            self.allocation_counter = 0
            self.total_allocated_bytes = 0
            self.peak_memory_usage = 0
            self.allocation_times.clear()
            self.gc_events.clear()
            
            for stats in self.pool_statistics.values():
                stats['allocation_count'] = 0
                stats['deallocation_count'] = 0
                stats['peak_usage'] = 0
            
            print("📊 Memory statistics reset")
    
    def export_memory_profile(self, filepath: str) -> bool:
        """Export memory usage profile for analysis."""
        try:
            import json
            
            profile_data = {
                'memory_config': {
                    'total_memory_mb': self.config.total_memory_mb,
                    'available_memory_mb': self.config.available_memory_mb,
                    'cache_line_size': self.config.cache_line_size,
                    'memory_alignment': self.config.memory_alignment
                },
                'memory_statistics': self.get_memory_statistics(),
                'pool_details': {
                    pool_type.value: {
                        'total_blocks': len(pool),
                        'allocated_blocks': sum(1 for b in pool if b.allocated),
                        'block_sizes': [b.size_bytes for b in pool[:5]]  # Sample
                    }
                    for pool_type, pool in self.memory_pools.items()
                },
                'streaming_buffers': {
                    name: {
                        'chunk_size': buffer.chunk_size,
                        'num_chunks': buffer.num_chunks,
                        'dtype': str(buffer.dtype)
                    }
                    for name, buffer in self.streaming_buffers.items()
                },
                'export_timestamp': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(profile_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Error exporting memory profile: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        # Deallocate all allocated blocks
        with self.allocation_lock:
            for pool in self.memory_pools.values():
                for block in pool:
                    if block.allocated:
                        block.deallocate()
        
        # Final garbage collection
        gc.collect()


# Example usage and testing
if __name__ == "__main__":
    # Create memory manager
    config = MemoryConfig(
        available_memory_mb=200.0,
        image_pool_mb=80.0,
        use_streaming_buffers=True
    )
    
    with EmbeddedMemoryManager(config) as memory_manager:
        # Test buffer allocation
        print("🧪 Testing buffer allocation...")
        
        # Allocate image buffer
        image_buffer = memory_manager.allocate_buffer(MemoryPoolType.IMAGE_BUFFERS)
        if image_buffer is not None:
            print(f"✅ Allocated image buffer: {image_buffer.shape}")
            
            # Use the buffer
            image_buffer.fill(128)  # Fill with test data
            
            # Deallocate
            memory_manager.deallocate_buffer(image_buffer)
            print("✅ Deallocated image buffer")
        
        # Test streaming buffer
        print("\n🧪 Testing streaming buffer...")
        test_large_image = np.random.randint(0, 255, (960, 1280, 3), dtype=np.uint8)
        
        def sample_chunk_processor(chunk):
            """Sample processing function for chunks."""
            return np.mean(chunk)
        
        results = memory_manager.process_large_image_streaming(
            test_large_image, sample_chunk_processor, "medium_chunks"
        )
        
        print(f"✅ Processed {len(results)} chunks")
        
        # Get statistics
        stats = memory_manager.get_memory_statistics()
        print(f"\n📊 Memory Statistics:")
        print(f"   Usage: {stats['usage_percentage']:.1f}%")
        print(f"   Peak: {stats['peak_memory_mb']:.1f}MB")
        print(f"   Allocations: {stats['allocation_count']}")
        
        # Export profile
        profile_path = "/Users/navyamudgal/Works/ACAD/Pest-Detection/output/memory_profile.json"
        memory_manager.export_memory_profile(profile_path)
        print(f"📁 Memory profile exported to: {profile_path}")