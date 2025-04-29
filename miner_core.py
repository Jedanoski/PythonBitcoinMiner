"""
Miner Core Module for Quantum-Accelerated Bitcoin Miner

This module implements the core mining functionality, leveraging quantum
decoherence to optimize the mining process using the Environment-Assisted
Quantum Transport (ENAQT) principle.
"""

import hashlib
import struct
import time
import multiprocessing
import logging
import json
import socket
import numpy as np
import psutil
from threading import Thread, Event
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

# Try to import quantum components, but don't fail if they're not available
try:
    from scsi_qubits import SCIQubits
    from decoherence import DecoherenceManager
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    logging.warning("Quantum components not available. Some functionality will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('miner_core')

class QuantumMiner:
    """
    Quantum-accelerated Bitcoin miner that uses decoherence to enhance mining performance.
    """
    
    def __init__(self, config_path='config.json', num_qubits=4, start_lba=1000):
        """
        Initialize the quantum miner.
        
        Args:
            config_path (str): Path to the configuration file
            num_qubits (int): Number of qubits to use
            start_lba (int): Starting LBA for the qubit register
        """
        self.config_path = config_path
        self.num_qubits = num_qubits
        self.start_lba = start_lba
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize statistics
        self.stats = {
            'hash_rate': 0.0,
            'shares_submitted': 0,
            'shares_accepted': 0,
            'shares_rejected': 0,
            'best_difficulty': 0.0,
            'uptime': 0,
            'quantum_collapses': 0,
            'decoherence_rate': 0.1,
            'worker_count': 0,
            'cpu_temp': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_activity': 0.0
        }
        
        # Initialize components
        self.qubits = None
        self.decoherence = None
        
        # Mining state
        self.start_time = time.time()
        self.hash_count = 0
        self.last_hash_time = time.time()
        self.last_hash_count = 0
        
        # Control flags
        self.running = False
        self.stop_event = Event()
        
        # Communication queues
        self.result_queue = Queue()
        self.job_queue = Queue()
        
        # Pool connection
        self.sock = None
        self.extranonce1 = None
        self.extranonce2_size = None
        
        # Worker management
        self.max_workers = multiprocessing.cpu_count()
        self.active_workers = []
        
        logger.info(f"Quantum Miner initialized with {num_qubits} qubits")
        
    def _load_config(self):
        """
        Load configuration from the config file.
        
        Returns:
            dict: Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as file:
                config = json.load(file)
                
            # Ensure required fields are present
            required_fields = ['pool_address', 'pool_port', 'user_name', 'password', 'min_diff']
            for field in required_fields:
                if field not in config:
                    logger.error(f"Missing required field in config: {field}")
                    raise ValueError(f"Missing required field in config: {field}")
                    
            # Add quantum-specific configuration with defaults
            if 'quantum' not in config:
                config['quantum'] = {
                    'num_qubits': self.num_qubits,
                    'start_lba': self.start_lba,
                    'decoherence_rate': 0.1,
                    'auto_adjust': True,
                    'measurement_interval': 5.0
                }
                
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
            
    def _save_config(self):
        """Save the current configuration to the config file."""
        try:
            with open(self.config_path, 'w') as file:
                json.dump(self.config, file, indent=4)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            
    def _connect_to_pool(self):
        """
        Connect to the mining pool.
        
        Returns:
            socket.socket: Connected socket
        """
        pool_address = self.config['pool_address']
        pool_port = self.config['pool_port']
        
        # Remove stratum+tcp:// prefix if present
        if pool_address.startswith("stratum+tcp://"):
            pool_address = pool_address[len("stratum+tcp://"):]
            
        retries = 5
        timeout = 30
        
        for attempt in range(retries):
            try:
                logger.info(f"Connecting to pool {pool_address}:{pool_port} (Attempt {attempt + 1}/{retries})...")
                sock = socket.create_connection((pool_address, pool_port), timeout)
                logger.info("Connected to pool!")
                return sock
            except socket.gaierror as e:
                logger.error(f"Address-related error connecting to server: {e}")
            except socket.timeout as e:
                logger.error(f"Connection timed out: {e}")
            except socket.error as e:
                logger.error(f"Socket error: {e}")
                
            logger.info(f"Retrying in 5 seconds...")
            time.sleep(5)
            
        raise Exception("Failed to connect to the pool after multiple attempts")
        
    def _send_message(self, message):
        """
        Send a message to the pool.
        
        Args:
            message (dict): Message to send
        """
        if not self.sock:
            logger.error("Socket not connected")
            return
            
        try:
            logger.debug(f"Sending message: {message}")
            self.sock.sendall((json.dumps(message) + '\n').encode('utf-8'))
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise
            
    def _receive_messages(self):
        """
        Receive and process messages from the pool.
        """
        if not self.sock:
            logger.error("Socket not connected")
            return
            
        buffer = b''
        self.sock.settimeout(1.0)  # Short timeout for responsive handling
        
        while not self.stop_event.is_set():
            try:
                chunk = self.sock.recv(4096)
                if not chunk:
                    logger.warning("Connection closed by pool")
                    break
                    
                buffer += chunk
                
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    message = json.loads(line.decode('utf-8'))
                    self._process_message(message)
                    
            except socket.timeout:
                # This is expected due to the short timeout
                continue
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}, data: {buffer[:100]}")
                buffer = b''
            except Exception as e:
                logger.error(f"Error receiving messages: {e}")
                time.sleep(1)
                
    def _process_message(self, message):
        """
        Process a message from the pool.
        
        Args:
            message (dict): Message from the pool
        """
        logger.debug(f"Received message: {message}")
        
        # Handle different message types
        if 'method' in message:
            # Server-to-client notification
            if message['method'] == 'mining.notify':
                # New job notification
                self.job_queue.put(message['params'])
                logger.info("Received new mining job")
                
            elif message['method'] == 'mining.set_difficulty':
                # Difficulty change
                new_diff = message['params'][0]
                logger.info(f"Pool set new difficulty: {new_diff}")
                
        elif 'id' in message:
            # Response to a client request
            if message['id'] == 1:  # mining.subscribe response
                if 'result' in message and message['result']:
                    self.extranonce1 = message['result'][1]
                    self.extranonce2_size = message['result'][2]
                    logger.info(f"Subscribed to pool, extranonce1: {self.extranonce1}, extranonce2_size: {self.extranonce2_size}")
                else:
                    logger.error(f"Subscribe failed: {message}")
                    
            elif message['id'] == 2:  # mining.authorize response
                if 'result' in message and message['result']:
                    logger.info("Worker authorized")
                else:
                    error = message.get('error', 'Unknown error')
                    logger.error(f"Authorization failed: {error}")
                    
            elif message['id'] == 4:  # mining.submit response
                if 'result' in message and message['result']:
                    self.stats['shares_accepted'] += 1
                    logger.info("Share accepted!")
                else:
                    self.stats['shares_rejected'] += 1
                    error = message.get('error', 'Unknown error')
                    logger.error(f"Share rejected: {error}")
                    
    def _subscribe(self):
        """
        Subscribe to the mining pool.
        """
        message = {
            "id": 1,
            "method": "mining.subscribe",
            "params": []
        }
        self._send_message(message)
        
    def _authorize(self):
        """
        Authorize with the mining pool.
        """
        username = self.config['user_name']
        password = self.config['password']
        
        message = {
            "id": 2,
            "method": "mining.authorize",
            "params": [username, password]
        }
        self._send_message(message)
        
    def _submit_solution(self, job_id, extranonce2, ntime, nonce):
        """
        Submit a solution to the pool.
        
        Args:
            job_id (str): Job ID
            extranonce2 (bytes): Extranonce2
            ntime (str): nTime
            nonce (int): Nonce
        """
        username = self.config['user_name']
        
        message = {
            "id": 4,
            "method": "mining.submit",
            "params": [username, job_id, extranonce2.hex(), ntime, struct.pack('<I', nonce).hex()]
        }
        self._send_message(message)
        self.stats['shares_submitted'] += 1
        
    def _calculate_difficulty(self, hash_result):
        """
        Calculate the difficulty of a hash result.
        
        Args:
            hash_result (bytes): Hash result
            
        Returns:
            float: Difficulty
        """
        hash_int = int.from_bytes(hash_result[::-1], byteorder='big')
        max_target = 0xffff * (2**208)
        difficulty = max_target / hash_int
        
        # Update best difficulty if higher
        if difficulty > self.stats['best_difficulty']:
            self.stats['best_difficulty'] = difficulty
            
        return difficulty
        
    def _mine_worker(self, job, target, extranonce1, extranonce2_size, nonce_start, nonce_end, result_queue, stop_event):
        """
        Worker function for mining.
        
        Args:
            job (list): Mining job parameters
            target (str): Target difficulty
            extranonce1 (str): Extranonce1
            extranonce2_size (int): Extranonce2 size
            nonce_start (int): Starting nonce
            nonce_end (int): Ending nonce
            result_queue (Queue): Queue for results
            stop_event (Event): Stop event
        """
        job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean_jobs = job
        
        extranonce2 = struct.pack('<Q', nonce_start)[:extranonce2_size]
        coinbase = (coinb1 + extranonce1 + extranonce2.hex() + coinb2).encode('utf-8')
        coinbase_hash_bin = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
        
        merkle_root = coinbase_hash_bin
        for branch in merkle_branch:
            merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + bytes.fromhex(branch)).digest()).digest()
            
        block_header = (version + prevhash + merkle_root[::-1].hex() + ntime + nbits).encode('utf-8')
        target_bin = bytes.fromhex(target)[::-1]
        
        min_diff = self.config['min_diff']
        hashes = 0
        start_time = time.time()
        
        for nonce in range(nonce_start, nonce_end):
            if stop_event.is_set():
                break
                
            nonce_bin = struct.pack('<I', nonce)
            hash_result = hashlib.sha256(hashlib.sha256(block_header + nonce_bin).digest()).digest()
            
            hashes += 1
            
            # Update hash count periodically
            if hashes % 1000 == 0:
                with self.hash_count_lock:
                    self.hash_count += 1000
                    
            if hash_result[::-1] < target_bin:
                difficulty = self._calculate_difficulty(hash_result)
                if difficulty > min_diff:
                    logger.info(f"Nonce found: {nonce}, Difficulty: {difficulty}")
                    logger.info(f"Hash: {hash_result[::-1].hex()}")
                    result_queue.put((job_id, extranonce2, ntime, nonce))
                    stop_event.set()
                    break
                    
        # Final hash count update
        with self.hash_count_lock:
            self.hash_count += (hashes % 1000)
            
        elapsed = time.time() - start_time
        if elapsed > 0:
            hash_rate = hashes / elapsed
            logger.debug(f"Worker finished: {hashes} hashes in {elapsed:.2f}s ({hash_rate:.2f} H/s)")
            
    def _quantum_mine(self, job):
        """
        Mine using quantum-accelerated approach.
        
        Args:
            job (list): Mining job parameters
            
        Returns:
            tuple: (job_id, extranonce2, ntime, nonce) if successful, None otherwise
        """
        if not job:
            logger.error("No job provided")
            return None
            
        job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean_jobs = job
        target = nbits
        
        # Get the list of available devices from the GUI's HDDInterface
        try:
            from gui import QuantumMinerGUI
            gui_app = QApplication.instance()
            if gui_app is None:
                logger.error("GUI application instance not found.")
                return None
            
            main_window = None
            for widget in QApplication.topLevelWidgets():
                if isinstance(widget, QuantumMinerGUI):
                    main_window = widget
                    break
            
            if main_window is None:
                logger.error("QuantumMinerGUI instance not found.")
                return None
            
            devices = main_window.hdd.devices
            if not devices:
                logger.warning("No HDDs detected. Mining will proceed without HDD I/O.")
                devices = [None]  # Use None to indicate no HDD
        
        except Exception as e:
            logger.error(f"Error accessing GUI for HDD devices: {e}")
            devices = [None]  # Fallback to no HDD
        
        for device in devices:
            # Check if quantum components are available
            if self.qubits and self.decoherence:
                # Prepare qubits in superposition
                self.qubits.prepare_superposition(device=device)
                
                # Apply HDD I/O gate to introduce real hardware noise
                self.qubits.apply_hdd_io_gate(device=device)
                
                # Apply decoherence to enhance transport
                state_vector = self.qubits.get_state_vector()
                state_vector = self.decoherence.apply_decoherence(state_vector, auto_adjust=self.config['quantum']['auto_adjust'])
                
                # Update qubit state
                self.qubits.state_vector = state_vector
                
                # Measure the quantum state
                measured_state, probability = self.qubits.measure()
                self.stats['quantum_collapses'] += 1
                
                # Use the measured state to determine the number of workers and nonce ranges
                worker_count = max(1, int(self.max_workers * probability))
                self.stats['worker_count'] = worker_count
                
                # Use the measured state as a seed for nonce ranges
                nonce_seed = measured_state * (2**24)  # Spread across the nonce space
                nonce_range_size = 2**32 // worker_count
                
                logger.info(f"Quantum collapse to state {measured_state} with probability {probability:.4f} on device {device}")
                logger.info(f"Spawning {worker_count} workers with nonce seed {nonce_seed}")
            else:
                # Classical mode - use random values
                import random
                worker_count = max(1, int(self.max_workers * 0.75))  # Use 75% of max workers
                self.stats['worker_count'] = worker_count
                
                nonce_seed = random.randint(0, 2**32 - 1)
                nonce_range_size = 2**32 // worker_count
                
                logger.info(f"Classical mode: Spawning {worker_count} workers with random nonce seed {nonce_seed} on device {device}")
            
            # Create a local stop event for this mining round
            local_stop_event = Event()
            
            # Create and start worker processes
            workers = []
            for i in range(worker_count):
                nonce_start = (nonce_seed + i * nonce_range_size) % (2**32)
                nonce_end = (nonce_start + nonce_range_size) % (2**32)
                
                worker = Thread(
                    target=self._mine_worker,
                    args=(job, target, self.extranonce1, self.extranonce2_size, 
                          nonce_start, nonce_end, self.result_queue, local_stop_event)
                )
                worker.daemon = True
                workers.append(worker)
                worker.start()
                
            # Store active workers
            self.active_workers = workers
            
            # Wait for a result or timeout
            timeout = self.config['quantum'].get('measurement_interval', 5.0)
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    result = self.result_queue.get(block=False)
                    local_stop_event.set()
                    
                    # Wait for workers to finish
                    for worker in workers:
                        worker.join(timeout=1.0)
                        
                    return result
                except Empty:
                    if self.stop_event.is_set():
                        break
                    time.sleep(0.1)
                    
            # No result found within timeout, stop workers
            local_stop_event.set()
            
            # Wait for workers to finish
            for worker in workers:
                worker.join(timeout=1.0)
            
        return None
        
    def _update_stats(self):
        """Update mining statistics."""
        current_time = time.time()
        elapsed = current_time - self.last_hash_time
        
        if elapsed >= 1.0:
            with self.hash_count_lock:
                hash_diff = self.hash_count - self.last_hash_count
                self.stats['hash_rate'] = hash_diff / elapsed
                self.last_hash_count = self.hash_count
                
            self.last_hash_time = current_time
            
        # Update uptime
        self.stats['uptime'] = int(current_time - self.start_time)
        
        # Update decoherence rate
        if self.decoherence:
            self.stats['decoherence_rate'] = self.decoherence.dephasing_rate
            
        # Update hardware metrics
        if self.decoherence:
            metrics = self.decoherence.get_hardware_metrics()
            if metrics:
                self.stats['cpu_usage'] = metrics['cpu_load']
                self.stats['cpu_temp'] = metrics['cpu_temp']
                self.stats['memory_usage'] = metrics['memory_usage']
                self.stats['disk_activity'] = metrics['disk_activity']
                
    def _stats_reporter(self):
        """Report mining statistics periodically."""
        while not self.stop_event.is_set():
            self._update_stats()
            
            # Log statistics
            logger.info(f"Hash rate: {self.stats['hash_rate']:.2f} H/s, "
                       f"Shares: {self.stats['shares_accepted']}/{self.stats['shares_submitted']}, "
                       f"Uptime: {self.stats['uptime']}s, "
                       f"Workers: {self.stats['worker_count']}, "
                       f"Decoherence: {self.stats['decoherence_rate']:.3f}")
                       
            time.sleep(5.0)
            
    def start(self):
        """Start the quantum miner."""
        if self.running:
            logger.warning("Miner is already running")
            return
            
        self.running = True
        self.stop_event.clear()
        self.start_time = time.time()
        self.hash_count = 0
        self.last_hash_time = time.time()
        self.last_hash_count = 0
        self.hash_count_lock = multiprocessing.Lock()
        
        # Initialize quantum components
        try:
            if not QUANTUM_AVAILABLE:
                logger.warning("Quantum components not available. Running in classical mode.")
                self.qubits = None
                self.decoherence = None
            else:
                self.qubits = SCIQubits(
                    num_qubits=self.config['quantum']['num_qubits'],
                    start_lba=self.config['quantum']['start_lba']
                )
                
                self.decoherence = DecoherenceManager(
                    num_qubits=self.config['quantum']['num_qubits'],
                    enable_hardware_monitoring=True
                )
                
                # Set initial decoherence rate
                self.decoherence.set_dephasing_rate(self.config['quantum']['decoherence_rate'])
            
        except Exception as e:
            logger.error(f"Error initializing quantum components: {e}")
            logger.warning("Running in classical mode.")
            self.qubits = None
            self.decoherence = None
            
        # Connect to pool
        try:
            self.sock = self._connect_to_pool()
            self._subscribe()
            self._authorize()
        except Exception as e:
            logger.error(f"Error connecting to pool: {e}")
            self.running = False
            return
            
        # Start message receiver thread
        self.receiver_thread = Thread(target=self._receive_messages)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()
        
        # Start stats reporter thread
        self.stats_thread = Thread(target=self._stats_reporter)
        self.stats_thread.daemon = True
        self.stats_thread.start()
        
        # Start mining loop
        self.mining_thread = Thread(target=self._mining_loop)
        self.mining_thread.daemon = True
        self.mining_thread.start()
        
        logger.info("Quantum miner started")
        
    def _mining_loop(self):
        """Main mining loop."""
        while not self.stop_event.is_set():
            try:
                # Get a job from the queue
                try:
                    job = self.job_queue.get(timeout=1.0)
                except Empty:
                    continue
                    
                # Mine the job
                result = self._quantum_mine(job)
                
                # Submit the result if found
                if result:
                    self._submit_solution(*result)
                    
            except Exception as e:
                logger.error(f"Error in mining loop: {e}")
                time.sleep(1.0)
                
    def stop(self):
        """Stop the quantum miner."""
        if not self.running:
            logger.warning("Miner is not running")
            return
            
        logger.info("Stopping quantum miner...")
        self.stop_event.set()
        
        # Wait for threads to finish
        if hasattr(self, 'mining_thread') and self.mining_thread.is_alive():
            self.mining_thread.join(timeout=5.0)
            
        if hasattr(self, 'receiver_thread') and self.receiver_thread.is_alive():
            self.receiver_thread.join(timeout=5.0)
            
        if hasattr(self, 'stats_thread') and self.stats_thread.is_alive():
            self.stats_thread.join(timeout=5.0)
            
        # Close socket
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
            
        # Clean up quantum components
        if self.qubits:
            self.qubits.close()
            self.qubits = None
            
        if self.decoherence:
            self.decoherence.close()
            self.decoherence = None
            
        self.running = False
        logger.info("Quantum miner stopped")
        
    def get_stats(self):
        """
        Get current mining statistics.
        
        Returns:
            dict: Mining statistics
        """
        self._update_stats()
        return self.stats
        
    def set_config(self, key, value):
        """
        Set a configuration value.
        
        Args:
            key (str): Configuration key
            value: Configuration value
        """
        # Handle nested keys
        if '.' in key:
            section, subkey = key.split('.', 1)
            if section not in self.config:
                self.config[section] = {}
            self.config[section][subkey] = value
        else:
            self.config[key] = value
            
        # Save the updated configuration
        self._save_config()
        logger.info(f"Configuration updated: {key} = {value}")
        
        # Apply changes if needed
        if key == 'quantum.decoherence_rate' and self.decoherence:
            self.decoherence.set_dephasing_rate(value)
            
    def get_config(self):
        """
        Get the current configuration.
        
        Returns:
            dict: Current configuration
        """
        return self.config


# Example usage
if __name__ == "__main__":
    # Test the quantum miner
    miner = QuantumMiner()
    
    try:
        miner.start()
        
        # Run for 60 seconds
        time.sleep(60)
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        miner.stop()
        
    # Print final statistics
    stats = miner.get_stats()
    print("Final statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")