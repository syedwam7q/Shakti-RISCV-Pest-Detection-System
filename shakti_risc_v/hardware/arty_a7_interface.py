"""
Arty A7-35T Board Interface for Shakti RISC-V
============================================

Complete hardware abstraction layer for Arty A7-35T FPGA board
running Shakti E-class RISC-V processor.

Board Specifications:
- FPGA: Xilinx Artix-7 XC7A35T-1CSG324C
- Memory: 256MB DDR3L
- Clock: 100MHz external, configurable internal
- GPIO: 4 LEDs, 4 buttons, 4 switches
- Interfaces: USB-UART, Ethernet, GPIO pins

Key Features:
- GPIO control for LEDs, buttons, switches
- UART communication interface
- Clock management and frequency scaling
- Memory mapped I/O access
- Interrupt handling
- Power management
"""

import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import queue


class GPIOPin(Enum):
    """GPIO pin definitions for Arty A7."""
    LED0 = 0
    LED1 = 1
    LED2 = 2
    LED3 = 3
    BTN0 = 4
    BTN1 = 5
    BTN2 = 6
    BTN3 = 7
    SW0 = 8
    SW1 = 9
    SW2 = 10
    SW3 = 11
    
    # External GPIO pins
    JA1 = 16  # Pmod JA pin 1
    JA2 = 17  # Pmod JA pin 2
    JA3 = 18  # Pmod JA pin 3
    JA4 = 19  # Pmod JA pin 4
    
    # Camera interface pins
    CAM_VSYNC = 20
    CAM_HSYNC = 21
    CAM_PCLK = 22
    CAM_DATA0 = 24
    CAM_DATA1 = 25
    CAM_DATA2 = 26
    CAM_DATA3 = 27
    CAM_DATA4 = 28
    CAM_DATA5 = 29
    CAM_DATA6 = 30
    CAM_DATA7 = 31


class ClockDomain(Enum):
    """Clock domains available on Arty A7."""
    SYS_CLK = "sys_clk"      # System clock (100MHz external)
    CPU_CLK = "cpu_clk"      # CPU clock (configurable)
    MEM_CLK = "mem_clk"      # Memory controller clock
    AUX_CLK = "aux_clk"      # Auxiliary clock for peripherals


@dataclass
class BoardConfig:
    """Configuration for Arty A7 board."""
    # Clock configuration
    external_clock_mhz: float = 100.0
    cpu_clock_mhz: float = 50.0        # Conservative for power efficiency
    memory_clock_mhz: float = 200.0    # DDR3 clock
    
    # Memory configuration
    ddr3_size_mb: int = 256
    cache_size_kb: int = 32
    
    # GPIO configuration
    led_active_high: bool = True
    button_debounce_ms: int = 50
    
    # UART configuration
    uart_baudrate: int = 115200
    uart_buffer_size: int = 1024
    
    # Power management
    enable_clock_gating: bool = True
    enable_power_islands: bool = True
    idle_clock_divisor: int = 4
    
    # Interrupt configuration
    enable_timer_interrupts: bool = True
    enable_gpio_interrupts: bool = True
    timer_interval_ms: float = 10.0
    simulate_button_presses: bool = False


class ArtyA7Interface:
    """
    Complete hardware interface for Arty A7-35T board with Shakti RISC-V.
    Provides abstraction for all board peripherals and features.
    """
    
    def __init__(self, config: Optional[BoardConfig] = None):
        self.config = config or BoardConfig()
        
        # Hardware state
        self.gpio_states = {}
        self.clock_frequencies = {}
        self.interrupt_handlers = {}
        
        # Communication interfaces
        self.uart_rx_queue = queue.Queue(maxsize=self.config.uart_buffer_size)
        self.uart_tx_queue = queue.Queue(maxsize=self.config.uart_buffer_size)
        
        # Performance monitoring
        self.gpio_access_count = 0
        self.interrupt_count = 0
        self.power_state = "active"
        
        # Thread for interrupt simulation
        self.interrupt_thread = None
        self.running = False
        
        # Initialize hardware
        self._initialize_hardware()
        
        print(f"🔧 Arty A7 Interface initialized")
        print(f"   CPU Clock: {self.config.cpu_clock_mhz}MHz")
        print(f"   Memory: {self.config.ddr3_size_mb}MB DDR3")
        print(f"   UART: {self.config.uart_baudrate} baud")
    
    def _initialize_hardware(self):
        """Initialize all hardware components."""
        # Initialize GPIO states
        for pin in GPIOPin:
            if pin.value < 16:  # On-board pins
                self.gpio_states[pin] = 0
        
        # Initialize clock domains
        self.clock_frequencies = {
            ClockDomain.SYS_CLK: self.config.external_clock_mhz,
            ClockDomain.CPU_CLK: self.config.cpu_clock_mhz,
            ClockDomain.MEM_CLK: self.config.memory_clock_mhz,
            ClockDomain.AUX_CLK: self.config.external_clock_mhz / 4
        }
        
        # Start interrupt handling thread only if interrupts are enabled
        self.running = True
        if self.config.enable_timer_interrupts or self.config.enable_gpio_interrupts:
            self.interrupt_thread = threading.Thread(target=self._interrupt_handler_thread)
            self.interrupt_thread.daemon = True
            self.interrupt_thread.start()
        else:
            self.interrupt_thread = None
        
        print("✅ Hardware initialization complete")
    
    def gpio_write(self, pin: GPIOPin, value: int) -> bool:
        """
        Write value to GPIO pin.
        
        Args:
            pin: GPIO pin to write
            value: Value to write (0 or 1)
            
        Returns:
            True if successful
        """
        try:
            if pin not in self.gpio_states:
                print(f"❌ GPIO pin {pin.name} not available")
                return False
            
            # Validate value
            value = 1 if value else 0
            
            # Handle active low logic for some pins
            if pin in [GPIOPin.LED0, GPIOPin.LED1, GPIOPin.LED2, GPIOPin.LED3]:
                if not self.config.led_active_high:
                    value = 1 - value
            
            # Simulate memory-mapped I/O access
            self._simulate_mmio_write(pin.value, value)
            
            self.gpio_states[pin] = value
            self.gpio_access_count += 1
            
            # Print status for LEDs (visible feedback)
            if pin.name.startswith('LED'):
                status = "ON" if value else "OFF"
                print(f"💡 {pin.name}: {status}")
            
            return True
            
        except Exception as e:
            print(f"❌ GPIO write error: {e}")
            return False
    
    def gpio_read(self, pin: GPIOPin) -> Optional[int]:
        """
        Read value from GPIO pin.
        
        Args:
            pin: GPIO pin to read
            
        Returns:
            Pin value (0 or 1) or None if error
        """
        try:
            if pin not in self.gpio_states:
                print(f"❌ GPIO pin {pin.name} not available")
                return None
            
            # Simulate memory-mapped I/O access
            value = self._simulate_mmio_read(pin.value)
            
            self.gpio_states[pin] = value
            self.gpio_access_count += 1
            
            return value
            
        except Exception as e:
            print(f"❌ GPIO read error: {e}")
            return None
    
    def _simulate_mmio_write(self, address: int, value: int):
        """Simulate memory-mapped I/O write access."""
        # Simulate the delay of memory-mapped I/O
        # In real hardware, this would be a direct memory write
        time.sleep(0.000001)  # 1μs delay to simulate hardware access
    
    def _simulate_mmio_read(self, address: int) -> int:
        """Simulate memory-mapped I/O read access."""
        # Simulate button presses and switch states
        if address in [4, 5, 6, 7]:  # Buttons
            # Simulate random button presses for testing
            import random
            return random.randint(0, 1) if random.random() < 0.01 else 0
        elif address in [8, 9, 10, 11]:  # Switches
            # Simulate switch states
            return self.gpio_states.get(GPIOPin(address), 0)
        else:
            return self.gpio_states.get(GPIOPin(address), 0)
    
    def set_clock_frequency(self, domain: ClockDomain, frequency_mhz: float) -> bool:
        """
        Set clock frequency for specified domain.
        
        Args:
            domain: Clock domain to configure
            frequency_mhz: Target frequency in MHz
            
        Returns:
            True if successful
        """
        try:
            # Validate frequency range
            if domain == ClockDomain.CPU_CLK:
                if not (10.0 <= frequency_mhz <= 100.0):
                    print(f"❌ CPU clock frequency {frequency_mhz}MHz out of range (10-100MHz)")
                    return False
            
            # Simulate PLL reconfiguration delay
            time.sleep(0.001)  # 1ms PLL lock time
            
            self.clock_frequencies[domain] = frequency_mhz
            
            print(f"⏰ {domain.value}: {frequency_mhz}MHz")
            return True
            
        except Exception as e:
            print(f"❌ Clock configuration error: {e}")
            return False
    
    def get_clock_frequency(self, domain: ClockDomain) -> float:
        """Get current clock frequency for domain."""
        return self.clock_frequencies.get(domain, 0.0)
    
    def uart_send(self, data: bytes) -> bool:
        """
        Send data via UART interface.
        
        Args:
            data: Data bytes to send
            
        Returns:
            True if successful
        """
        try:
            if self.uart_tx_queue.full():
                print("⚠️ UART TX buffer full")
                return False
            
            self.uart_tx_queue.put(data)
            
            # Simulate UART transmission
            transmission_time = len(data) * 8 / self.config.uart_baudrate
            time.sleep(transmission_time)
            
            return True
            
        except Exception as e:
            print(f"❌ UART send error: {e}")
            return False
    
    def uart_receive(self, timeout_ms: float = 100) -> Optional[bytes]:
        """
        Receive data from UART interface.
        
        Args:
            timeout_ms: Timeout in milliseconds
            
        Returns:
            Received data bytes or None if timeout
        """
        try:
            timeout_seconds = timeout_ms / 1000.0
            data = self.uart_rx_queue.get(timeout=timeout_seconds)
            return data
            
        except queue.Empty:
            return None
        except Exception as e:
            print(f"❌ UART receive error: {e}")
            return None
    
    def register_interrupt_handler(self, interrupt_type: str, 
                                 handler: Callable[[str], None]):
        """
        Register interrupt handler.
        
        Args:
            interrupt_type: Type of interrupt (timer, gpio, uart, etc.)
            handler: Handler function
        """
        self.interrupt_handlers[interrupt_type] = handler
        print(f"🔔 Registered interrupt handler for {interrupt_type}")
    
    def _interrupt_handler_thread(self):
        """Thread function for handling interrupts."""
        last_timer = time.time()
        
        while self.running:
            current_time = time.time()
            
            # Timer interrupt
            if (self.config.enable_timer_interrupts and 
                current_time - last_timer >= self.config.timer_interval_ms / 1000.0):
                
                if "timer" in self.interrupt_handlers:
                    try:
                        self.interrupt_handlers["timer"]("timer")
                        self.interrupt_count += 1
                    except Exception as e:
                        print(f"❌ Timer interrupt handler error: {e}")
                
                last_timer = current_time
            
            # GPIO interrupt (button presses) - only if enabled and simulating
            if (self.config.enable_gpio_interrupts and self.config.simulate_button_presses):
                for pin in [GPIOPin.BTN0, GPIOPin.BTN1, GPIOPin.BTN2, GPIOPin.BTN3]:
                    current_value = self._simulate_mmio_read(pin.value)
                    previous_value = self.gpio_states.get(pin, 0)
                    
                    if current_value != previous_value and current_value == 1:
                        # Button press detected
                        if "gpio" in self.interrupt_handlers:
                            try:
                                self.interrupt_handlers["gpio"](f"button_{pin.value - 4}")
                                self.interrupt_count += 1
                            except Exception as e:
                                print(f"❌ GPIO interrupt handler error: {e}")
            
            # Sleep to avoid busy waiting
            time.sleep(0.001)  # 1ms
    
    def enter_power_saving_mode(self, mode: str = "idle"):
        """
        Enter power saving mode.
        
        Args:
            mode: Power saving mode ("idle", "sleep", "deep_sleep")
        """
        if mode == "idle":
            # Reduce CPU clock frequency
            original_freq = self.clock_frequencies[ClockDomain.CPU_CLK]
            idle_freq = original_freq / self.config.idle_clock_divisor
            self.set_clock_frequency(ClockDomain.CPU_CLK, idle_freq)
            self.power_state = "idle"
            
        elif mode == "sleep":
            # Stop non-essential clocks
            self.set_clock_frequency(ClockDomain.AUX_CLK, 0)
            self.power_state = "sleep"
            
        elif mode == "deep_sleep":
            # Minimal power mode
            self.set_clock_frequency(ClockDomain.CPU_CLK, 1.0)  # Minimum frequency
            self.power_state = "deep_sleep"
        
        print(f"💤 Entered {mode} power mode")
    
    def exit_power_saving_mode(self):
        """Exit power saving mode and restore normal operation."""
        # Restore normal clock frequencies
        self.set_clock_frequency(ClockDomain.CPU_CLK, self.config.cpu_clock_mhz)
        self.set_clock_frequency(ClockDomain.AUX_CLK, self.config.external_clock_mhz / 4)
        self.power_state = "active"
        
        print("🔋 Exited power saving mode")
    
    def read_system_temperature(self) -> float:
        """
        Read FPGA die temperature from built-in sensor.
        
        Returns:
            Temperature in Celsius
        """
        # Simulate temperature reading
        # In real hardware, this would read from XADC
        import random
        base_temp = 45.0  # Base temperature
        variation = random.uniform(-5.0, 15.0)
        return base_temp + variation
    
    def read_system_voltages(self) -> Dict[str, float]:
        """
        Read system voltages from built-in monitors.
        
        Returns:
            Dictionary of voltage rail measurements
        """
        # Simulate voltage readings
        # In real hardware, this would read from XADC
        return {
            'vccint': 1.0 + (random.uniform(-0.05, 0.05) if hasattr(self, 'random') else 0),
            'vccaux': 1.8 + (random.uniform(-0.1, 0.1) if hasattr(self, 'random') else 0),
            'vccbram': 1.0 + (random.uniform(-0.05, 0.05) if hasattr(self, 'random') else 0)
        }
    
    def get_board_statistics(self) -> Dict:
        """Get comprehensive board statistics."""
        return {
            'gpio_access_count': self.gpio_access_count,
            'interrupt_count': self.interrupt_count,
            'power_state': self.power_state,
            'clock_frequencies': self.clock_frequencies.copy(),
            'temperature_c': self.read_system_temperature(),
            'voltages': self.read_system_voltages(),
            'uart_buffer_usage': {
                'tx_queue_size': self.uart_tx_queue.qsize(),
                'rx_queue_size': self.uart_rx_queue.qsize()
            }
        }
    
    def reset_statistics(self):
        """Reset all board statistics."""
        self.gpio_access_count = 0
        self.interrupt_count = 0
        print("📊 Board statistics reset")
    
    def shutdown(self):
        """Shutdown the board interface cleanly."""
        self.running = False
        
        if self.interrupt_thread and self.interrupt_thread.is_alive():
            self.interrupt_thread.join(timeout=1.0)
        
        # Turn off all LEDs
        for led in [GPIOPin.LED0, GPIOPin.LED1, GPIOPin.LED2, GPIOPin.LED3]:
            self.gpio_write(led, 0)
        
        print("🔌 Arty A7 interface shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()


# Example usage and testing
if __name__ == "__main__":
    import random
    
    # Create board interface
    config = BoardConfig(
        cpu_clock_mhz=50.0,
        enable_timer_interrupts=True,
        timer_interval_ms=100.0
    )
    
    def timer_interrupt_handler(interrupt_type: str):
        """Example timer interrupt handler."""
        print(f"⏰ Timer interrupt: {interrupt_type}")
    
    def gpio_interrupt_handler(interrupt_type: str):
        """Example GPIO interrupt handler."""
        print(f"🔘 Button pressed: {interrupt_type}")
    
    with ArtyA7Interface(config) as board:
        # Register interrupt handlers
        board.register_interrupt_handler("timer", timer_interrupt_handler)
        board.register_interrupt_handler("gpio", gpio_interrupt_handler)
        
        print("🧪 Testing Arty A7 interface...")
        
        # Test LED control
        print("\n💡 Testing LED control...")
        for i in range(4):
            led = GPIOPin(i)  # LED0 to LED3
            board.gpio_write(led, 1)
            time.sleep(0.2)
            board.gpio_write(led, 0)
        
        # Test clock frequency control
        print("\n⏰ Testing clock control...")
        board.set_clock_frequency(ClockDomain.CPU_CLK, 25.0)
        time.sleep(0.5)
        board.set_clock_frequency(ClockDomain.CPU_CLK, 50.0)
        
        # Test UART communication
        print("\n📡 Testing UART...")
        test_data = b"Hello Shakti RISC-V!"
        board.uart_send(test_data)
        
        # Test power management
        print("\n💤 Testing power management...")
        board.enter_power_saving_mode("idle")
        time.sleep(1.0)
        board.exit_power_saving_mode()
        
        # Wait for some interrupts
        print("\n⏳ Waiting for interrupts...")
        time.sleep(2.0)
        
        # Get statistics
        stats = board.get_board_statistics()
        print(f"\n📊 Board Statistics:")
        print(f"   GPIO accesses: {stats['gpio_access_count']}")
        print(f"   Interrupts: {stats['interrupt_count']}")
        print(f"   Temperature: {stats['temperature_c']:.1f}°C")
        print(f"   Power state: {stats['power_state']}")
        print(f"   CPU frequency: {stats['clock_frequencies'][ClockDomain.CPU_CLK]}MHz")
        
        print("\n✅ Arty A7 interface test complete")