"""
Decoherence Module for Quantum-Accelerated Bitcoin Miner

This module implements quantum decoherence channels to simulate the effects
of environmental noise on the quantum state. It uses the Environment-Assisted
Quantum Transport (ENAQT) principle to optimize mining performance.
"""

import numpy as np
import logging
import time
from threading import Thread
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('decoherence')

class DecoherenceChannel:
    """Base class for decoherence channels."""
    
    def __init__(self, num_qubits):
        """
        Initialize the decoherence channel.
        
        Args:
            num_qubits (int): Number of qubits in the system
        """
        self.num_qubits = num_qubits
        
    def apply(self, state_vector, p):
        """
        Apply the decoherence channel to the state vector.
        
        Args:
            state_vector (numpy.ndarray): The quantum state vector
            p (float): Decoherence probability (0 to 1)
            
        Returns:
            numpy.ndarray: The state vector after decoherence
        """
        raise NotImplementedError("Subclasses must implement this method")


class DephasingChannel(DecoherenceChannel):
    """
    Implements a dephasing (phase damping) channel.
    
    This channel causes qubits to lose phase coherence without energy loss.
    It's equivalent to randomly applying Z gates with probability p.
    """
    
    def apply(self, state_vector, p):
        """
        Apply dephasing to the state vector.
        
        Args:
            state_vector (numpy.ndarray): The quantum state vector
            p (float): Dephasing probability (0 to 1)
            
        Returns:
            numpy.ndarray: The state vector after dephasing
        """
        if p <= 0:
            return state_vector
            
        if p >= 1:
            p = 0.999  # Avoid complete decoherence
            
        new_state = np.copy(state_vector)
        
        # Apply dephasing to each qubit
        for q in range(self.num_qubits):
            # Probability of applying Z gate
            if np.random.random() < p:
                # Apply Z gate to qubit q
                for i in range(2**self.num_qubits):
                    # Check if the q-th bit is 1
                    if (i >> (self.num_qubits - 1 - q)) & 1:
                        # Apply phase flip
                        new_state[i] *= -1
                        
        logger.debug(f"Applied dephasing with p={p}")
        return new_state


class AmplitudeDampingChannel(DecoherenceChannel):
    """
    Implements an amplitude damping channel.
    
    This channel models energy dissipation, causing qubits to decay from |1⟩ to |0⟩.
    """
    
    def apply(self, state_vector, p):
        """
        Apply amplitude damping to the state vector.
        
        Args:
            state_vector (numpy.ndarray): The quantum state vector
            p (float): Damping probability (0 to 1)
            
        Returns:
            numpy.ndarray: The state vector after amplitude damping
        """
        if p <= 0:
            return state_vector
            
        if p >= 1:
            p = 0.999  # Avoid complete decoherence
            
        new_state = np.copy(state_vector)
        
        # Apply amplitude damping to each qubit
        for q in range(self.num_qubits):
            # For each basis state
            for i in range(2**self.num_qubits):
                # Check if the q-th bit is 1
                if (i >> (self.num_qubits - 1 - q)) & 1:
                    # Compute the corresponding state with q-th bit flipped to 0
                    j = i & ~(1 << (self.num_qubits - 1 - q))
                    
                    # Apply damping
                    decay_amplitude = np.sqrt(p) * new_state[i]
                    new_state[j] += decay_amplitude
                    new_state[i] *= np.sqrt(1 - p)
                    
        # Normalize the state
        norm = np.linalg.norm(new_state)
        if norm > 0:
            new_state /= norm
            
        logger.debug(f"Applied amplitude damping with p={p}")
        return new_state


class DepolarizingChannel(DecoherenceChannel):
    """
    Implements a depolarizing channel.
    
    This channel models general noise, randomly applying X, Y, or Z gates.
    """
    
    def apply(self, state_vector, p):
        """
        Apply depolarizing noise to the state vector.
        
        Args:
            state_vector (numpy.ndarray): The quantum state vector
            p (float): Depolarizing probability (0 to 1)
            
        Returns:
            numpy.ndarray: The state vector after depolarizing
        """
        if p <= 0:
            return state_vector
            
        if p >= 1:
            p = 0.999  # Avoid complete decoherence
            
        new_state = np.copy(state_vector)
        
        # Apply depolarizing to each qubit
        for q in range(self.num_qubits):
            # Probability of applying a Pauli gate
            if np.random.random() < p:
                # Randomly choose X, Y, or Z gate
                gate = np.random.choice(['X', 'Y', 'Z'])
                
                if gate == 'X':
                    # Apply X gate (bit flip)
                    for i in range(2**self.num_qubits):
                        # Flip the q-th bit
                        j = i ^ (1 << (self.num_qubits - 1 - q))
                        new_state[i], new_state[j] = new_state[j], new_state[i]
                        
                elif gate == 'Y':
                    # Apply Y gate (bit and phase flip)
                    for i in range(2**self.num_qubits):
                        # Flip the q-th bit
                        j = i ^ (1 << (self.num_qubits - 1 - q))
                        # Swap with phase change
                        temp = new_state[i]
                        new_state[i] = -1j * new_state[j]
                        new_state[j] = 1j * temp
                        
                else:  # Z gate
                    # Apply Z gate (phase flip)
                    for i in range(2**self.num_qubits):
                        # Check if the q-th bit is 1
                        if (i >> (self.num_qubits - 1 - q)) & 1:
                            # Apply phase flip
                            new_state[i] *= -1
                            
        logger.debug(f"Applied depolarizing with p={p}, gate={gate}")
        return new_state


class HardwareNoiseMonitor:
    """
    Monitors hardware noise sources to dynamically adjust decoherence rates.
    """
    
    def __init__(self, update_interval=1.0):
        """
        Initialize the hardware noise monitor.
        
        Args:
            update_interval (float): Update interval in seconds
        """
        self.update_interval = update_interval
        self.running = False
        self.thread = None
        
        # Noise metrics
        self.cpu_load = 0.0
        self.cpu_temp = 0.0
        self.disk_activity = 0.0
        self.memory_usage = 0.0
        
        # Baseline values
        self.baseline_disk_io = None
        
    def start(self):
        """Start the hardware monitoring thread."""
        if self.thread is not None and self.thread.is_alive():
            return
            
        self.running = True
        self.thread = Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Hardware noise monitor started")
        
    def stop(self):
        """Stop the hardware monitoring thread."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            logger.info("Hardware noise monitor stopped")
            
    def _monitor_loop(self):
        """Main monitoring loop."""
        # Initialize baseline values
        disk_io_before = psutil.disk_io_counters()
        time_before = time.time()
        
        while self.running:
            try:
                # CPU metrics
                self.cpu_load = psutil.cpu_percent(interval=None) / 100.0
                
                # Try to get CPU temperature (platform-dependent)
                try:
                    temps = psutil.sensors_temperatures()
                    if temps and 'coretemp' in temps:
                        self.cpu_temp = sum(t.current for t in temps['coretemp']) / len(temps['coretemp'])
                    else:
                        self.cpu_temp = 0.0
                except:
                    self.cpu_temp = 0.0
                    
                # Disk activity
                disk_io_after = psutil.disk_io_counters()
                time_after = time.time()
                
                time_delta = time_after - time_before
                if time_delta > 0:
                    read_bytes = disk_io_after.read_bytes - disk_io_before.read_bytes
                    write_bytes = disk_io_after.write_bytes - disk_io_before.write_bytes
                    
                    # Calculate disk activity in MB/s
                    disk_activity_mbs = (read_bytes + write_bytes) / (1024 * 1024) / time_delta
                    
                    # Normalize to 0-1 range (assuming max 100 MB/s)
                    self.disk_activity = min(1.0, disk_activity_mbs / 100.0)
                    
                disk_io_before = disk_io_after
                time_before = time_after
                
                # Memory usage
                self.memory_usage = psutil.virtual_memory().percent / 100.0
                
                logger.debug(f"Hardware metrics - CPU: {self.cpu_load:.2f}, Temp: {self.cpu_temp:.1f}°C, "
                           f"Disk: {self.disk_activity:.2f}, Mem: {self.memory_usage:.2f}")
                           
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in hardware monitoring: {e}")
                time.sleep(self.update_interval)
                
    def get_noise_metrics(self):
        """
        Get the current noise metrics.
        
        Returns:
            dict: Dictionary of noise metrics
        """
        return {
            'cpu_load': self.cpu_load,
            'cpu_temp': self.cpu_temp,
            'disk_activity': self.disk_activity,
            'memory_usage': self.memory_usage
        }
        
    def get_optimal_decoherence_rate(self):
        """
        Calculate the optimal decoherence rate based on hardware metrics.
        
        Returns:
            float: Optimal decoherence rate (0 to 1)
        """
        # ENAQT principle: moderate noise can enhance transport
        # We use disk activity as the primary source of "natural" decoherence
        natural_decoherence = self.disk_activity * 0.2  # Scale factor
        
        # Add a baseline decoherence rate for optimal ENAQT (around 0.1)
        optimal_rate = 0.1 + natural_decoherence
        
        # Ensure the rate is within bounds
        return max(0.05, min(0.3, optimal_rate))


class DecoherenceManager:
    """
    Manages decoherence channels and applies them to quantum states.
    """
    
    def __init__(self, num_qubits, enable_hardware_monitoring=True):
        """
        Initialize the decoherence manager.
        
        Args:
            num_qubits (int): Number of qubits in the system
            enable_hardware_monitoring (bool): Whether to enable hardware monitoring
        """
        self.num_qubits = num_qubits
        
        # Create decoherence channels
        self.dephasing = DephasingChannel(num_qubits)
        self.amplitude_damping = AmplitudeDampingChannel(num_qubits)
        self.depolarizing = DepolarizingChannel(num_qubits)
        
        # Default decoherence rates
        self.dephasing_rate = 0.1
        self.amplitude_damping_rate = 0.05
        self.depolarizing_rate = 0.02
        
        # Hardware noise monitor
        self.hardware_monitor = None
        if enable_hardware_monitoring:
            self.hardware_monitor = HardwareNoiseMonitor()
            self.hardware_monitor.start()
            
        logger.info(f"Decoherence manager initialized with {num_qubits} qubits")
        
    def set_dephasing_rate(self, rate):
        """Set the dephasing rate."""
        self.dephasing_rate = max(0.0, min(1.0, rate))
        logger.info(f"Dephasing rate set to {self.dephasing_rate}")
        
    def set_amplitude_damping_rate(self, rate):
        """Set the amplitude damping rate."""
        self.amplitude_damping_rate = max(0.0, min(1.0, rate))
        logger.info(f"Amplitude damping rate set to {self.amplitude_damping_rate}")
        
    def set_depolarizing_rate(self, rate):
        """Set the depolarizing rate."""
        self.depolarizing_rate = max(0.0, min(1.0, rate))
        logger.info(f"Depolarizing rate set to {self.depolarizing_rate}")
        
    def apply_decoherence(self, state_vector, auto_adjust=True):
        """
        Apply all decoherence channels to the state vector.
        
        Args:
            state_vector (numpy.ndarray): The quantum state vector
            auto_adjust (bool): Whether to auto-adjust rates based on hardware metrics
            
        Returns:
            numpy.ndarray: The state vector after decoherence
        """
        # Auto-adjust rates based on hardware metrics if enabled
        if auto_adjust and self.hardware_monitor:
            optimal_rate = self.hardware_monitor.get_optimal_decoherence_rate()
            self.dephasing_rate = optimal_rate
            self.amplitude_damping_rate = optimal_rate * 0.5
            self.depolarizing_rate = optimal_rate * 0.2
            
        # Apply decoherence channels
        state = self.dephasing.apply(state_vector, self.dephasing_rate)
        state = self.amplitude_damping.apply(state, self.amplitude_damping_rate)
        state = self.depolarizing.apply(state, self.depolarizing_rate)
        
        # Normalize the state
        norm = np.linalg.norm(state)
        if norm > 0:
            state /= norm
            
        logger.info(f"Applied decoherence with rates: dephasing={self.dephasing_rate:.3f}, "
                   f"amplitude={self.amplitude_damping_rate:.3f}, depolarizing={self.depolarizing_rate:.3f}")
                   
        return state
        
    def get_hardware_metrics(self):
        """
        Get the current hardware metrics.
        
        Returns:
            dict: Dictionary of hardware metrics, or None if monitoring is disabled
        """
        if self.hardware_monitor:
            return self.hardware_monitor.get_noise_metrics()
        return None
        
    def close(self):
        """Clean up resources."""
        if self.hardware_monitor:
            self.hardware_monitor.stop()
            logger.info("Decoherence manager closed")


# Example usage
if __name__ == "__main__":
    # Test the decoherence module
    num_qubits = 2
    
    # Create a simple state vector (|00⟩ + |11⟩)/√2
    state = np.zeros(2**num_qubits, dtype=complex)
    state[0] = 1/np.sqrt(2)
    state[3] = 1/np.sqrt(2)
    
    print("Initial state:", state)
    
    # Create decoherence manager
    manager = DecoherenceManager(num_qubits)
    
    # Apply decoherence
    new_state = manager.apply_decoherence(state)
    print("After decoherence:", new_state)
    
    # Wait to see hardware metrics
    print("Monitoring hardware for 5 seconds...")
    time.sleep(5)
    
    # Get hardware metrics
    metrics = manager.get_hardware_metrics()
    print("Hardware metrics:", metrics)
    
    # Apply auto-adjusted decoherence
    new_state = manager.apply_decoherence(state, auto_adjust=True)
    print("After auto-adjusted decoherence:", new_state)
    
    manager.close()