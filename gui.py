"""
GUI Module for Quantum-Accelerated Bitcoin Miner

This module implements a graphical user interface for the quantum miner
using PySide6 and PyQtGraph.
"""

import sys
import time
import logging
import multiprocessing
import numpy as np
from threading import Thread, Event
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QTabWidget, QLineEdit,
    QFormLayout, QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QComboBox, QTextEdit, QSplitter
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread
from PySide6.QtGui import QFont, QColor
import pyqtgraph as pg

from miner_core import QuantumMiner
from hdd_interface import HDDInterface
from scsi_qubits import SCIQubits
from decoherence import DecoherenceManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('gui')

# Set PyQtGraph configuration
pg.setConfigOptions(antialias=True, background='w', foreground='k')

class HDDMonitorThread(QThread):
    """Thread for monitoring HDD I/O operations."""
    
    update_signal = Signal(dict)
    
    def __init__(self, hdd_interface):
        super().__init__()
        self.hdd = hdd_interface
        self.running = True
        
    def run(self):
        """Main thread loop."""
        while self.running:
            try:
                # Get HDD information
                info = self.hdd.inquiry() if self.hdd else None
                
                # Read from a few LBAs to monitor activity
                read_results = []
                if self.hdd:
                    for lba in range(1000, 1010):
                        data = self.hdd.scsi_read_lba(lba)
                        if data:
                            read_results.append((lba, len(data), data[:16].hex()))
                            
                # Emit update signal
                self.update_signal.emit({
                    'info': info,
                    'read_results': read_results,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                logger.error(f"Error in HDD monitor thread: {e}")
                
            time.sleep(1.0)
            
    def stop(self):
        """Stop the thread."""
        self.running = False
        self.wait()


class QubitStateThread(QThread):
    """Thread for monitoring qubit states."""
    
    update_signal = Signal(dict)
    
    def __init__(self, qubits):
        super().__init__()
        self.qubits = qubits
        self.running = True
        
    def run(self):
        """Main thread loop."""
        while self.running:
            try:
                if self.qubits:
                    # Get state vector and probabilities
                    state_vector = self.qubits.get_state_vector()
                    probabilities = self.qubits.get_probabilities()
                    
                    # Emit update signal
                    self.update_signal.emit({
                        'state_vector': state_vector,
                        'probabilities': probabilities,
                        'timestamp': time.time()
                    })
                    
            except Exception as e:
                logger.error(f"Error in qubit state thread: {e}")
                
            time.sleep(0.5)
            
    def stop(self):
        """Stop the thread."""
        self.running = False
        self.wait()


class DecoherenceThread(QThread):
    """Thread for monitoring decoherence events."""
    
    update_signal = Signal(dict)
    
    def __init__(self, decoherence_manager):
        super().__init__()
        self.decoherence = decoherence_manager
        self.running = True
        
    def run(self):
        """Main thread loop."""
        while self.running:
            try:
                if self.decoherence:
                    # Get hardware metrics
                    metrics = self.decoherence.get_hardware_metrics()
                    
                    # Get decoherence rates
                    rates = {
                        'dephasing': self.decoherence.dephasing_rate,
                        'amplitude_damping': self.decoherence.amplitude_damping_rate,
                        'depolarizing': self.decoherence.depolarizing_rate
                    }
                    
                    # Emit update signal
                    self.update_signal.emit({
                        'metrics': metrics,
                        'rates': rates,
                        'timestamp': time.time()
                    })
                    
            except Exception as e:
                logger.error(f"Error in decoherence thread: {e}")
                
            time.sleep(0.5)
            
    def stop(self):
        """Stop the thread."""
        self.running = False
        self.wait()


class MinerStatsThread(QThread):
    """Thread for monitoring miner statistics."""
    
    update_signal = Signal(dict)
    
    def __init__(self, miner):
        super().__init__()
        self.miner = miner
        self.running = True
        
    def run(self):
        """Main thread loop."""
        while self.running:
            try:
                if self.miner:
                    # Get miner statistics
                    stats = self.miner.get_stats()
                    
                    # Emit update signal
                    self.update_signal.emit({
                        'stats': stats,
                        'timestamp': time.time()
                    })
                    
            except Exception as e:
                logger.error(f"Error in miner stats thread: {e}")
                
            time.sleep(1.0)
            
    def stop(self):
        """Stop the thread."""
        self.running = False
        self.wait()


class BlochSphereWidget(pg.PlotWidget):
    """Widget for displaying a Bloch sphere representation of a qubit."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up 3D plot
        self.setAspectLocked(True)
        self.setLabel('left', 'Z')
        self.setLabel('bottom', 'X')
        self.setLabel('right', 'Y')
        self.setRange(xRange=(-1.1, 1.1), yRange=(-1.1, 1.1))
        
        # Draw Bloch sphere
        self._draw_sphere()
        
        # Initialize state vector
        self.state_vector = np.array([1.0, 0.0], dtype=complex)
        
        # Create point for state
        self.state_point = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255))
        self.addItem(self.state_point)
        
        # Update state display
        self.update_state(self.state_vector)
        
    def _draw_sphere(self):
        """Draw the Bloch sphere."""
        # Draw circle
        theta = np.linspace(0, 2*np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        self.addItem(pg.PlotCurveItem(x, y, pen=pg.mkPen(color='k', width=1)))
        
        # Draw X, Y, Z axes
        self.addItem(pg.PlotCurveItem([0, 1], [0, 0], pen=pg.mkPen(color='r', width=2)))  # X-axis
        self.addItem(pg.PlotCurveItem([0, 0], [0, 1], pen=pg.mkPen(color='g', width=2)))  # Y-axis
        self.addItem(pg.PlotCurveItem([0, 0], [0, 0], pen=pg.mkPen(color='b', width=2)))  # Z-axis (into the screen)
        
        # Add labels
        text_0 = pg.TextItem(text='|0⟩', color='k', anchor=(0.5, 0.5))
        text_1 = pg.TextItem(text='|1⟩', color='k', anchor=(0.5, 0.5))
        text_plus = pg.TextItem(text='|+⟩', color='k', anchor=(0.5, 0.5))
        text_minus = pg.TextItem(text='|-⟩', color='k', anchor=(0.5, 0.5))
        
        self.addItem(text_0)
        self.addItem(text_1)
        self.addItem(text_plus)
        self.addItem(text_minus)
        
        text_0.setPos(0, 1)
        text_1.setPos(0, -1)
        text_plus.setPos(1, 0)
        text_minus.setPos(-1, 0)
        
    def update_state(self, state_vector):
        """
        Update the state display.
        
        Args:
            state_vector (numpy.ndarray): 2-element complex array representing the qubit state
        """
        self.state_vector = state_vector
        
        # Calculate Bloch sphere coordinates
        alpha = state_vector[0]
        beta = state_vector[1]
        
        # Convert to Bloch sphere coordinates
        theta = 2 * np.angle(beta * np.conj(alpha))
        phi = 2 * np.arccos(np.abs(alpha))
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        # Update state point
        self.state_point.setData([x], [y])


class HDDIOTab(QWidget):
    """Tab for displaying HDD I/O operations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # HDD information group
        info_group = QGroupBox("HDD Information")
        info_layout = QFormLayout(info_group)
        
        self.vendor_label = QLabel("N/A")
        self.product_label = QLabel("N/A")
        self.version_label = QLabel("N/A")
        
        info_layout.addRow("Vendor:", self.vendor_label)
        info_layout.addRow("Product:", self.product_label)
        info_layout.addRow("Version:", self.version_label)
        
        layout.addWidget(info_group)
        
        # LBA queue group
        lba_group = QGroupBox("LBA Queue")
        lba_layout = QVBoxLayout(lba_group)
        
        self.lba_table = QTableWidget(0, 3)
        self.lba_table.setHorizontalHeaderLabels(["LBA", "Size", "Data (first 16 bytes)"])
        self.lba_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        lba_layout.addWidget(self.lba_table)
        
        layout.addWidget(lba_group)
        
        # Throughput graph
        throughput_group = QGroupBox("Throughput")
        throughput_layout = QVBoxLayout(throughput_group)
        
        self.throughput_plot = pg.PlotWidget()
        self.throughput_plot.setLabel('left', 'Throughput', 'MB/s')
        self.throughput_plot.setLabel('bottom', 'Time', 's')
        self.throughput_curve = self.throughput_plot.plot(pen=pg.mkPen(color='b', width=2))
        
        throughput_layout.addWidget(self.throughput_plot)
        
        layout.addWidget(throughput_group)
        
        # Initialize data
        self.throughput_data = {'x': [], 'y': []}
        self.start_time = time.time()
        
    @Slot(dict)
    def update_hdd_info(self, data):
        """
        Update HDD information.
        
        Args:
            data (dict): HDD information data
        """
        info = data.get('info')
        if info:
            self.vendor_label.setText(info.get('vendor', 'N/A'))
            self.product_label.setText(info.get('product', 'N/A'))
            self.version_label.setText(info.get('version', 'N/A'))
            
        read_results = data.get('read_results', [])
        
        # Update LBA table
        self.lba_table.setRowCount(len(read_results))
        for i, (lba, size, data_hex) in enumerate(read_results):
            self.lba_table.setItem(i, 0, QTableWidgetItem(str(lba)))
            self.lba_table.setItem(i, 1, QTableWidgetItem(str(size)))
            self.lba_table.setItem(i, 2, QTableWidgetItem(data_hex))
            
        # Update throughput graph
        timestamp = data.get('timestamp', time.time())
        elapsed = timestamp - self.start_time
        
        # Calculate throughput (MB/s)
        throughput = sum(size for _, size, _ in read_results) / (1024 * 1024)
        
        # Add data point
        self.throughput_data['x'].append(elapsed)
        self.throughput_data['y'].append(throughput)
        
        # Keep only the last 100 points
        if len(self.throughput_data['x']) > 100:
            self.throughput_data['x'] = self.throughput_data['x'][-100:]
            self.throughput_data['y'] = self.throughput_data['y'][-100:]
            
        # Update plot
        self.throughput_curve.setData(self.throughput_data['x'], self.throughput_data['y'])


class QubitRegisterTab(QWidget):
    """Tab for displaying qubit register state."""
    
    def __init__(self, num_qubits=4, parent=None):
        super().__init__(parent)
        
        self.num_qubits = num_qubits
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Bloch spheres group
        bloch_group = QGroupBox("Qubit States (Bloch Spheres)")
        bloch_layout = QHBoxLayout(bloch_group)
        
        self.bloch_spheres = []
        for i in range(num_qubits):
            sphere = BlochSphereWidget()
            sphere.setTitle(f"Qubit {i}")
            self.bloch_spheres.append(sphere)
            bloch_layout.addWidget(sphere)
            
        layout.addWidget(bloch_group)
        
        # State vector group
        state_group = QGroupBox("State Vector")
        state_layout = QVBoxLayout(state_group)
        
        self.state_table = QTableWidget(2**num_qubits, 3)
        self.state_table.setHorizontalHeaderLabels(["State", "Amplitude", "Probability"])
        self.state_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        state_layout.addWidget(self.state_table)
        
        layout.addWidget(state_group)
        
        # Initialize state table
        for i in range(2**num_qubits):
            binary = format(i, f'0{num_qubits}b')
            self.state_table.setItem(i, 0, QTableWidgetItem(f"|{binary}⟩"))
            self.state_table.setItem(i, 1, QTableWidgetItem("0"))
            self.state_table.setItem(i, 2, QTableWidgetItem("0"))
            
    @Slot(dict)
    def update_qubit_state(self, data):
        """
        Update qubit state display.
        
        Args:
            data (dict): Qubit state data
        """
        state_vector = data.get('state_vector')
        probabilities = data.get('probabilities')
        
        if state_vector is None or probabilities is None:
            return
            
        # Update Bloch spheres for individual qubits
        for i in range(self.num_qubits):
            # Calculate reduced density matrix for this qubit
            reduced_state = self._calculate_reduced_state(state_vector, i)
            self.bloch_spheres[i].update_state(reduced_state)
            
        # Update state table
        for i in range(2**self.num_qubits):
            amplitude = state_vector[i]
            probability = probabilities[i]
            
            self.state_table.setItem(i, 1, QTableWidgetItem(f"{amplitude.real:.4f} + {amplitude.imag:.4f}j"))
            self.state_table.setItem(i, 2, QTableWidgetItem(f"{probability:.6f}"))
            
            # Highlight the most probable state
            if probability == max(probabilities):
                for j in range(3):
                    self.state_table.item(i, j).setBackground(QColor(200, 255, 200))
            else:
                for j in range(3):
                    self.state_table.item(i, j).setBackground(QColor(255, 255, 255))
                    
    def _calculate_reduced_state(self, state_vector, qubit_idx):
        """
        Calculate the reduced state for a single qubit.
        
        Args:
            state_vector (numpy.ndarray): Full state vector
            qubit_idx (int): Index of the qubit
            
        Returns:
            numpy.ndarray: 2-element state vector for the qubit
        """
        # For a pure state, we can calculate the reduced state by tracing out other qubits
        # This is a simplified approach for visualization purposes
        
        # Initialize reduced state
        reduced_state = np.zeros(2, dtype=complex)
        
        # For each basis state
        for i in range(2**self.num_qubits):
            # Convert to binary
            binary = format(i, f'0{self.num_qubits}b')
            
            # Get the bit value for this qubit
            bit = int(binary[qubit_idx])
            
            # Add contribution to the reduced state
            reduced_state[bit] += np.abs(state_vector[i])**2
            
        # Normalize
        norm = np.sqrt(np.sum(np.abs(reduced_state)**2))
        if norm > 0:
            reduced_state /= norm
            
        return reduced_state


class DecoherenceTab(QWidget):
    """Tab for controlling and monitoring decoherence."""
    
    decoherence_changed = Signal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Decoherence rate control
        rate_group = QGroupBox("Decoherence Rate")
        rate_layout = QVBoxLayout(rate_group)
        
        slider_layout = QHBoxLayout()
        self.rate_slider = QSlider(Qt.Horizontal)
        self.rate_slider.setMinimum(0)
        self.rate_slider.setMaximum(100)
        self.rate_slider.setValue(10)  # Default 0.1
        self.rate_slider.setTickPosition(QSlider.TicksBelow)
        self.rate_slider.setTickInterval(10)
        
        self.rate_label = QLabel("0.10")
        slider_layout.addWidget(self.rate_slider)
        slider_layout.addWidget(self.rate_label)
        
        rate_layout.addLayout(slider_layout)
        
        self.auto_adjust_checkbox = QCheckBox("Auto-adjust based on hardware noise")
        self.auto_adjust_checkbox.setChecked(True)
        rate_layout.addWidget(self.auto_adjust_checkbox)
        
        layout.addWidget(rate_group)
        
        # Decoherence channels group
        channels_group = QGroupBox("Decoherence Channels")
        channels_layout = QFormLayout(channels_group)
        
        self.dephasing_label = QLabel("0.10")
        self.amplitude_label = QLabel("0.05")
        self.depolarizing_label = QLabel("0.02")
        
        channels_layout.addRow("Dephasing:", self.dephasing_label)
        channels_layout.addRow("Amplitude Damping:", self.amplitude_label)
        channels_layout.addRow("Depolarizing:", self.depolarizing_label)
        
        layout.addWidget(channels_group)
        
        # Hardware noise group
        noise_group = QGroupBox("Hardware Noise Sources")
        noise_layout = QFormLayout(noise_group)
        
        self.cpu_load_bar = QProgressBar()
        self.cpu_temp_bar = QProgressBar()
        self.disk_activity_bar = QProgressBar()
        self.memory_usage_bar = QProgressBar()
        
        noise_layout.addRow("CPU Load:", self.cpu_load_bar)
        noise_layout.addRow("CPU Temperature:", self.cpu_temp_bar)
        noise_layout.addRow("Disk Activity:", self.disk_activity_bar)
        noise_layout.addRow("Memory Usage:", self.memory_usage_bar)
        
        layout.addWidget(noise_group)
        
        # Noise events graph
        events_group = QGroupBox("Decoherence Events")
        events_layout = QVBoxLayout(events_group)
        
        self.events_plot = pg.PlotWidget()
        self.events_plot.setLabel('left', 'Rate')
        self.events_plot.setLabel('bottom', 'Time', 's')
        
        self.dephasing_curve = self.events_plot.plot(pen=pg.mkPen(color='r', width=2), name="Dephasing")
        self.amplitude_curve = self.events_plot.plot(pen=pg.mkPen(color='g', width=2), name="Amplitude")
        self.depolarizing_curve = self.events_plot.plot(pen=pg.mkPen(color='b', width=2), name="Depolarizing")
        
        self.events_plot.addLegend()
        
        events_layout.addWidget(self.events_plot)
        
        layout.addWidget(events_group)
        
        # Connect signals
        self.rate_slider.valueChanged.connect(self._on_rate_changed)
        self.auto_adjust_checkbox.stateChanged.connect(self._on_auto_adjust_changed)
        
        # Initialize data
        self.decoherence_data = {
            'x': [],
            'dephasing': [],
            'amplitude': [],
            'depolarizing': []
        }
        self.start_time = time.time()
        
    def _on_rate_changed(self, value):
        """Handle rate slider change."""
        rate = value / 100.0
        self.rate_label.setText(f"{rate:.2f}")
        self.decoherence_changed.emit(rate)
        
    def _on_auto_adjust_changed(self, state):
        """Handle auto-adjust checkbox change."""
        self.rate_slider.setEnabled(not state)
        
    @Slot(dict)
    def update_decoherence(self, data):
        """
        Update decoherence display.
        
        Args:
            data (dict): Decoherence data
        """
        metrics = data.get('metrics')
        rates = data.get('rates')
        timestamp = data.get('timestamp', time.time())
        
        if metrics:
            # Update hardware noise bars
            self.cpu_load_bar.setValue(int(metrics['cpu_load'] * 100))
            self.cpu_temp_bar.setValue(int(metrics['cpu_temp']))
            self.disk_activity_bar.setValue(int(metrics['disk_activity'] * 100))
            self.memory_usage_bar.setValue(int(metrics['memory_usage'] * 100))
            
        if rates:
            # Update decoherence channel labels
            self.dephasing_label.setText(f"{rates['dephasing']:.3f}")
            self.amplitude_label.setText(f"{rates['amplitude_damping']:.3f}")
            self.depolarizing_label.setText(f"{rates['depolarizing']:.3f}")
            
            # Update rate slider if auto-adjust is enabled
            if self.auto_adjust_checkbox.isChecked():
                self.rate_slider.setValue(int(rates['dephasing'] * 100))
                self.rate_label.setText(f"{rates['dephasing']:.2f}")
                
            # Update events graph
            elapsed = timestamp - self.start_time
            
            self.decoherence_data['x'].append(elapsed)
            self.decoherence_data['dephasing'].append(rates['dephasing'])
            self.decoherence_data['amplitude'].append(rates['amplitude_damping'])
            self.decoherence_data['depolarizing'].append(rates['depolarizing'])
            
            # Keep only the last 100 points
            if len(self.decoherence_data['x']) > 100:
                self.decoherence_data['x'] = self.decoherence_data['x'][-100:]
                self.decoherence_data['dephasing'] = self.decoherence_data['dephasing'][-100:]
                self.decoherence_data['amplitude'] = self.decoherence_data['amplitude'][-100:]
                self.decoherence_data['depolarizing'] = self.decoherence_data['depolarizing'][-100:]
                
            # Update plots
            self.dephasing_curve.setData(self.decoherence_data['x'], self.decoherence_data['dephasing'])
            self.amplitude_curve.setData(self.decoherence_data['x'], self.decoherence_data['amplitude'])
            self.depolarizing_curve.setData(self.decoherence_data['x'], self.decoherence_data['depolarizing'])


class MiningConfigTab(QWidget):
    """Tab for configuring mining parameters."""
    
    config_changed = Signal(str, object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Pool configuration group
        pool_group = QGroupBox("Pool Configuration")
        pool_layout = QFormLayout(pool_group)
        
        self.pool_address_edit = QLineEdit()
        self.pool_port_spin = QSpinBox()
        self.pool_port_spin.setRange(1, 65535)
        self.pool_port_spin.setValue(3333)
        
        self.username_edit = QLineEdit()
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        
        pool_layout.addRow("Pool Address:", self.pool_address_edit)
        pool_layout.addRow("Pool Port:", self.pool_port_spin)
        pool_layout.addRow("Username:", self.username_edit)
        pool_layout.addRow("Password:", self.password_edit)
        
        layout.addWidget(pool_group)
        
        # Quantum configuration group
        quantum_group = QGroupBox("Quantum Configuration")
        quantum_layout = QFormLayout(quantum_group)
        
        self.num_qubits_spin = QSpinBox()
        self.num_qubits_spin.setRange(2, 8)
        self.num_qubits_spin.setValue(4)
        
        self.start_lba_spin = QSpinBox()
        self.start_lba_spin.setRange(1, 1000000)
        self.start_lba_spin.setValue(1000)
        
        self.measurement_interval_spin = QDoubleSpinBox()
        self.measurement_interval_spin.setRange(1.0, 60.0)
        self.measurement_interval_spin.setValue(5.0)
        self.measurement_interval_spin.setSuffix(" seconds")
        
        quantum_layout.addRow("Number of Qubits:", self.num_qubits_spin)
        quantum_layout.addRow("Start LBA:", self.start_lba_spin)
        quantum_layout.addRow("Measurement Interval:", self.measurement_interval_spin)
        
        layout.addWidget(quantum_group)
        
        # Mining parameters group
        mining_group = QGroupBox("Mining Parameters")
        mining_layout = QFormLayout(mining_group)
        
        self.min_diff_spin = QDoubleSpinBox()
        self.min_diff_spin.setRange(0.0001, 100.0)
        self.min_diff_spin.setValue(0.01)
        self.min_diff_spin.setDecimals(4)
        
        self.max_workers_spin = QSpinBox()
        self.max_workers_spin.setRange(1, 32)
        self.max_workers_spin.setValue(multiprocessing.cpu_count())
        
        mining_layout.addRow("Minimum Difficulty:", self.min_diff_spin)
        mining_layout.addRow("Maximum Workers:", self.max_workers_spin)
        
        layout.addWidget(mining_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save Configuration")
        self.reset_button = QPushButton("Reset to Defaults")
        
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.reset_button)
        
        layout.addLayout(button_layout)
        
        # Connect signals
        self.save_button.clicked.connect(self._on_save_clicked)
        self.reset_button.clicked.connect(self._on_reset_clicked)
        
    def set_config(self, config):
        """
        Set the configuration values.
        
        Args:
            config (dict): Configuration dictionary
        """
        # Pool configuration
        self.pool_address_edit.setText(config.get('pool_address', ''))
        self.pool_port_spin.setValue(config.get('pool_port', 3333))
        self.username_edit.setText(config.get('user_name', ''))
        self.password_edit.setText(config.get('password', ''))
        
        # Quantum configuration
        quantum_config = config.get('quantum', {})
        self.num_qubits_spin.setValue(quantum_config.get('num_qubits', 4))
        self.start_lba_spin.setValue(quantum_config.get('start_lba', 1000))
        self.measurement_interval_spin.setValue(quantum_config.get('measurement_interval', 5.0))
        
        # Mining parameters
        self.min_diff_spin.setValue(config.get('min_diff', 0.01))
        self.max_workers_spin.setValue(config.get('max_workers', multiprocessing.cpu_count()))
        
    def _on_save_clicked(self):
        """Handle save button click."""
        # Emit config changed signals
        self.config_changed.emit('pool_address', self.pool_address_edit.text())
        self.config_changed.emit('pool_port', self.pool_port_spin.value())
        self.config_changed.emit('user_name', self.username_edit.text())
        self.config_changed.emit('password', self.password_edit.text())
        
        self.config_changed.emit('quantum.num_qubits', self.num_qubits_spin.value())
        self.config_changed.emit('quantum.start_lba', self.start_lba_spin.value())
        self.config_changed.emit('quantum.measurement_interval', self.measurement_interval_spin.value())
        
        self.config_changed.emit('min_diff', self.min_diff_spin.value())
        self.config_changed.emit('max_workers', self.max_workers_spin.value())
        
    def _on_reset_clicked(self):
        """Handle reset button click."""
        # Reset to default values
        self.pool_address_edit.setText('public-pool.io')
        self.pool_port_spin.setValue(21496)
        self.username_edit.setText('')
        self.password_edit.setText('x')
        
        self.num_qubits_spin.setValue(4)
        self.start_lba_spin.setValue(1000)
        self.measurement_interval_spin.setValue(5.0)
        
        self.min_diff_spin.setValue(0.01)
        self.max_workers_spin.setValue(multiprocessing.cpu_count())


class PerformanceTab(QWidget):
    """Tab for displaying mining performance."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Hash rate graph
        hashrate_group = QGroupBox("Hash Rate")
        hashrate_layout = QVBoxLayout(hashrate_group)
        
        self.hashrate_plot = pg.PlotWidget()
        self.hashrate_plot.setLabel('left', 'Hash Rate', 'H/s')
        self.hashrate_plot.setLabel('bottom', 'Time', 's')
        self.hashrate_curve = self.hashrate_plot.plot(pen=pg.mkPen(color='g', width=2))
        
        hashrate_layout.addWidget(self.hashrate_plot)
        
        layout.addWidget(hashrate_group)
        
        # Statistics group
        stats_group = QGroupBox("Mining Statistics")
        stats_layout = QFormLayout(stats_group)
        
        self.hashrate_label = QLabel("0 H/s")
        self.shares_label = QLabel("0/0/0")
        self.uptime_label = QLabel("0s")
        self.workers_label = QLabel("0")
        self.collapses_label = QLabel("0")
        self.best_diff_label = QLabel("0")
        
        stats_layout.addRow("Hash Rate:", self.hashrate_label)
        stats_layout.addRow("Shares (Accepted/Submitted/Rejected):", self.shares_label)
        stats_layout.addRow("Uptime:", self.uptime_label)
        stats_layout.addRow("Active Workers:", self.workers_label)
        stats_layout.addRow("Quantum Collapses:", self.collapses_label)
        stats_layout.addRow("Best Difficulty:", self.best_diff_label)
        
        layout.addWidget(stats_group)
        
        # Hardware monitoring group
        hardware_group = QGroupBox("Hardware Monitoring")
        hardware_layout = QFormLayout(hardware_group)
        
        self.cpu_usage_bar = QProgressBar()
        self.cpu_temp_bar = QProgressBar()
        self.memory_usage_bar = QProgressBar()
        
        hardware_layout.addRow("CPU Usage:", self.cpu_usage_bar)
        hardware_layout.addRow("CPU Temperature:", self.cpu_temp_bar)
        hardware_layout.addRow("Memory Usage:", self.memory_usage_bar)
        
        layout.addWidget(hardware_group)
        
        # Initialize data
        self.hashrate_data = {'x': [], 'y': []}
        self.start_time = time.time()
        
    @Slot(dict)
    def update_performance(self, data):
        """
        Update performance display.
        
        Args:
            data (dict): Performance data
        """
        stats = data.get('stats')
        timestamp = data.get('timestamp', time.time())
        
        if not stats:
            return
            
        # Update statistics labels
        self.hashrate_label.setText(f"{stats['hash_rate']:.2f} H/s")
        self.shares_label.setText(f"{stats['shares_accepted']}/{stats['shares_submitted']}/{stats['shares_rejected']}")
        
        # Format uptime
        uptime = stats['uptime']
        hours, remainder = divmod(uptime, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        self.uptime_label.setText(uptime_str)
        
        self.workers_label.setText(str(stats['worker_count']))
        self.collapses_label.setText(str(stats['quantum_collapses']))
        self.best_diff_label.setText(f"{stats['best_difficulty']:.6f}")
        
        # Update hardware monitoring
        self.cpu_usage_bar.setValue(int(stats['cpu_usage'] * 100))
        self.cpu_temp_bar.setValue(int(stats['cpu_temp']))
        self.memory_usage_bar.setValue(int(stats['memory_usage'] * 100))
        
        # Update hash rate graph
        elapsed = timestamp - self.start_time
        
        self.hashrate_data['x'].append(elapsed)
        self.hashrate_data['y'].append(stats['hash_rate'])
        
        # Keep only the last 100 points
        if len(self.hashrate_data['x']) > 100:
            self.hashrate_data['x'] = self.hashrate_data['x'][-100:]
            self.hashrate_data['y'] = self.hashrate_data['y'][-100:]
            
        # Update plot
        self.hashrate_curve.setData(self.hashrate_data['x'], self.hashrate_data['y'])


class LogTab(QWidget):
    """Tab for displaying log messages."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier New", 9))
        
        layout.addWidget(self.log_text)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.clear_button = QPushButton("Clear Log")
        self.save_button = QPushButton("Save Log")
        
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.save_button)
        
        layout.addLayout(button_layout)
        
        # Connect signals
        self.clear_button.clicked.connect(self.log_text.clear)
        self.save_button.clicked.connect(self._on_save_clicked)
        
        # Set up log handler
        self.log_handler = QTextEditLogger(self.log_text)
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(self.log_handler)
        
    def _on_save_clicked(self):
        """Handle save button click."""
        # Save log to file
        try:
            with open("quantum_miner.log", "w") as f:
                f.write(self.log_text.toPlainText())
            logging.info("Log saved to quantum_miner.log")
        except Exception as e:
            logging.error(f"Error saving log: {e}")


class QTextEditLogger(logging.Handler):
    """Logger handler that outputs to a QTextEdit."""
    
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit
        
    def emit(self, record):
        """Emit a log record."""
        msg = self.format(record)
        self.text_edit.append(msg)


class QuantumMinerGUI(QMainWindow):
    """Main window for the Quantum Miner GUI."""
    
    def __init__(self):
        super().__init__()
        
        # Set up window
        self.setWindowTitle("Quantum-Accelerated Bitcoin Miner")
        self.setMinimumSize(800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.hdd_tab = HDDIOTab()
        self.qubit_tab = QubitRegisterTab(num_qubits=4)
        self.decoherence_tab = DecoherenceTab()
        self.config_tab = MiningConfigTab()
        self.performance_tab = PerformanceTab()
        self.log_tab = LogTab()
        
        # Add tabs
        self.tabs.addTab(self.performance_tab, "Performance")
        self.tabs.addTab(self.qubit_tab, "Qubit Register")
        self.tabs.addTab(self.decoherence_tab, "Decoherence")
        self.tabs.addTab(self.hdd_tab, "HDD I/O")
        self.tabs.addTab(self.config_tab, "Configuration")
        self.tabs.addTab(self.log_tab, "Log")
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Mining")
        self.stop_button = QPushButton("Stop Mining")
        self.stop_button.setEnabled(False)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        
        main_layout.addLayout(button_layout)
        
        # Initialize components
        self.miner = None
        self.hdd = None
        self.qubits = None
        self.decoherence = None
        
        # Initialize threads
        self.hdd_thread = None
        self.qubit_thread = None
        self.decoherence_thread = None
        self.miner_thread = None
        
        # Connect signals
        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.decoherence_tab.decoherence_changed.connect(self._on_decoherence_changed)
        self.config_tab.config_changed.connect(self._on_config_changed)
        
        # Load initial configuration
        self._load_config()
        
    def _load_config(self):
        """Load configuration from file."""
        try:
            with open('config.json', 'r') as file:
                config = json.load(file)
            self.config_tab.set_config(config)
            logging.info("Configuration loaded from config.json")
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            
    def _on_start_clicked(self):
        """Handle start button click."""
        try:
            # Initialize components
            self.hdd = HDDInterface(auto_detect=True)
            
            num_qubits = self.config_tab.num_qubits_spin.value()
            start_lba = self.config_tab.start_lba_spin.value()
            
            self.qubits = SCIQubits(num_qubits=num_qubits, start_lba=start_lba, hdd_interface=self.hdd)
            self.decoherence = DecoherenceManager(num_qubits=num_qubits, enable_hardware_monitoring=True)
            
            # Set initial decoherence rate
            rate = self.decoherence_tab.rate_slider.value() / 100.0
            self.decoherence.set_dephasing_rate(rate)
            
            # Initialize miner
            self.miner = QuantumMiner(num_qubits=num_qubits, start_lba=start_lba)
            
            # Start monitoring threads
            self.hdd_thread = HDDMonitorThread(self.hdd)
            self.hdd_thread.update_signal.connect(self.hdd_tab.update_hdd_info)
            self.hdd_thread.start()
            
            self.qubit_thread = QubitStateThread(self.qubits)
            self.qubit_thread.update_signal.connect(self.qubit_tab.update_qubit_state)
            self.qubit_thread.start()
            
            self.decoherence_thread = DecoherenceThread(self.decoherence)
            self.decoherence_thread.update_signal.connect(self.decoherence_tab.update_decoherence)
            self.decoherence_thread.start()
            
            self.miner_thread = MinerStatsThread(self.miner)
            self.miner_thread.update_signal.connect(self.performance_tab.update_performance)
            self.miner_thread.start()
            
            # Start miner
            self.miner.start()
            
            # Update UI
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            
            logging.info("Quantum miner started")
            
        except Exception as e:
            logging.error(f"Error starting miner: {e}")
            self._cleanup()
            
    def _on_stop_clicked(self):
        """Handle stop button click."""
        try:
            # Stop miner
            if self.miner:
                self.miner.stop()
                
            # Clean up
            self._cleanup()
            
            # Update UI
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            
            logging.info("Quantum miner stopped")
            
        except Exception as e:
            logging.error(f"Error stopping miner: {e}")
            
    def _cleanup(self):
        """Clean up resources."""
        # Stop threads
        if self.hdd_thread:
            self.hdd_thread.stop()
            self.hdd_thread = None
            
        if self.qubit_thread:
            self.qubit_thread.stop()
            self.qubit_thread = None
            
        if self.decoherence_thread:
            self.decoherence_thread.stop()
            self.decoherence_thread = None
            
        if self.miner_thread:
            self.miner_thread.stop()
            self.miner_thread = None
            
        # Clean up components
        if self.miner:
            self.miner = None
            
        if self.decoherence:
            self.decoherence.close()
            self.decoherence = None
            
        if self.qubits:
            self.qubits.close()
            self.qubits = None
            
        if self.hdd:
            self.hdd.close()
            self.hdd = None
            
    def _on_decoherence_changed(self, rate):
        """Handle decoherence rate change."""
        if self.decoherence:
            self.decoherence.set_dephasing_rate(rate)
            
        if self.miner:
            self.miner.set_config('quantum.decoherence_rate', rate)
            
    def _on_config_changed(self, key, value):
        """Handle configuration change."""
        if self.miner:
            self.miner.set_config(key, value)
            
    def closeEvent(self, event):
        """Handle window close event."""
        self._on_stop_clicked()
        event.accept()


def main():
    """Main function."""
    app = QApplication(sys.argv)
    window = QuantumMinerGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()