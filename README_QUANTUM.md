# Quantum-Accelerated Bitcoin Miner

This project extends the PythonBitcoinMiner with quantum-inspired algorithms that harness decoherence to enhance mining performance. By consciously controlling decoherence events—both software-simulated and hardware-induced—we can boost hash rate through the Environment-Assisted Quantum Transport (ENAQT) principle.

## Key Concepts

### Environment-Assisted Quantum Transport (ENAQT)

Decoherence is typically viewed as a source of errors in quantum systems. However, research has shown that moderate levels of decoherence can actually enhance quantum transport phenomena. This is known as Environment-Assisted Quantum Transport (ENAQT).

In our mining application, we leverage this principle by:

1. Treating the mining process as a quantum transport problem
2. Introducing controlled decoherence to prevent localization in the search space
3. Using quantum collapse events to dynamically allocate computational resources

### Hardware-Induced Decoherence

We use an external USB 3.0 HDD as both:

1. A physical qubit register (through SCSI read/write operations)
2. A natural source of decoherence (through magnetic domain noise)

The HDD's magnetic platter noise provides a genuine source of environmental decoherence that we harness rather than fight against.

## Architecture

The quantum-accelerated miner consists of several key components:

### 1. HDD Interface (hdd_interface.py)

- Interfaces with an external USB 3.0 HDD using libusb-win32 and PyUSB
- Implements SCSI commands for reading and writing to the HDD
- Handles error recovery and endpoint management

### 2. SCSI Qubits (scsi_qubits.py)

- Maps SCSI read/write operations to qubit states
- Implements quantum gates (X, H) and measurement operations
- Manages the quantum state vector

### 3. Decoherence Module (decoherence.py)

- Implements various decoherence channels (dephasing, amplitude damping, depolarizing)
- Monitors hardware noise sources to dynamically adjust decoherence rates
- Applies the ENAQT principle to optimize mining performance

### 4. Miner Core (miner_core.py)

- Integrates quantum components with the Bitcoin mining process
- Uses quantum measurement outcomes to dynamically allocate CPU resources
- Implements the stratum protocol for pool mining

### 5. GUI (gui.py)

- Provides a comprehensive interface for monitoring and controlling the miner
- Visualizes qubit states, decoherence events, and mining performance
- Allows configuration of mining and quantum parameters

## Installation

### Prerequisites

- Windows with libusb-win32 drivers installed
- Python 3.8 or higher
- External USB 3.0 HDD

### Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Install libusb-win32 drivers for your USB HDD:
   - Download Zadig (https://zadig.akeo.ie/)
   - Connect your external HDD
   - Run Zadig, select your HDD, and install the libusb-win32 driver

3. Configure the miner in `config.json`:
   - Set your pool information
   - Configure quantum parameters

## Usage

### GUI Mode

```bash
python quantum_miner.py --gui
```

### Command-Line Mode

```bash
python quantum_miner.py --qubits 4 --decoherence 0.1 --auto-adjust
```

### Command-Line Options

- `--gui`: Start with GUI
- `--config`: Path to configuration file (default: config.json)
- `--qubits`: Number of qubits to use (default: 4)
- `--lba`: Starting LBA for qubit register (default: 1000)
- `--decoherence`: Decoherence rate (default: 0.1)
- `--auto-adjust`: Auto-adjust decoherence rate based on hardware metrics

## Performance Optimization

For optimal performance:

1. Find the optimal decoherence rate for your hardware (typically around 0.1-0.2)
2. Use 4-6 qubits for most systems (more qubits require more computational resources)
3. Enable auto-adjustment to dynamically optimize based on hardware conditions
4. Monitor CPU temperature and throttle if necessary

## Theory and Implementation

### Qubit Mapping

Each LBA read/write pair is treated as one qubit:
- |0⟩: READ success → amplitude αᵢ = ∥data_blockᵢ∥
- |1⟩: WRITE success → amplitude βᵢ = 1 if status==0 else 0

For an N-qubit register, we collect amplitudes into a state vector ψ of size 2ᴺ.

### Decoherence Channels

We implement three types of decoherence channels:

1. **Dephasing Channel**: Causes qubits to lose phase coherence (Z errors)
2. **Amplitude Damping Channel**: Models energy dissipation (|1⟩ → |0⟩ decay)
3. **Depolarizing Channel**: General noise (random X, Y, Z errors)

### Mining Algorithm

1. Prepare qubits in superposition
2. Apply HDD I/O operations as quantum gates
3. Apply controlled decoherence
4. Measure the quantum state
5. Use measurement outcome to allocate CPU workers
6. Repeat

## References

- ArXiv: "Environment-assisted quantum transport" (https://arxiv.org/abs/0806.4552)
- Nature: "Environment-assisted quantum walks in photosynthetic energy transfer" (https://www.nature.com/articles/nphys1652)
- ResearchGate: "Quantum walks with decoherence" (https://www.researchgate.net/publication/51938950_Quantum_walks_with_decoherence)

## License

This project is licensed under the MIT License - see the LICENSE file for details.