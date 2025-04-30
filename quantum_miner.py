"""
Quantum-Accelerated Bitcoin Miner

This is the main entry point for the quantum-accelerated Bitcoin miner.
It leverages quantum decoherence to optimize mining performance using
the Environment-Assisted Quantum Transport (ENAQT) principle.
"""

import sys
import argparse
import logging
import time

# Try to import GUI components, but don't fail if they're not available
try:
    from PySide6.QtWidgets import QApplication
    from gui import QuantumMinerGUI
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    logging.warning("GUI components not available. Install PySide6 and pyqtgraph to use GUI mode.")

from miner_core import QuantumMiner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quantum_miner.log')
    ]
)
logger = logging.getLogger('quantum_miner')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Quantum-Accelerated Bitcoin Miner')
    
    parser.add_argument('--gui', action='store_true', help='Start with GUI')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--qubits', type=int, default=4, help='Number of qubits to use')
    parser.add_argument('--lba', type=int, default=1000, help='Starting LBA for qubit register')
    parser.add_argument('--decoherence', type=float, default=0.1, help='Decoherence rate')
    parser.add_argument('--auto-adjust', action='store_true', help='Auto-adjust decoherence rate')
    
    return parser.parse_args()

def run_cli(args):
    """Run in command-line mode."""
    logger.info("Starting quantum miner in CLI mode")
    
    try:
        # Create miner
        miner = QuantumMiner(
            config_path=args.config,
            num_qubits=args.qubits,
            start_lba=args.lba
        )
    except Exception as e:
        logger.error(f"Error initializing QuantumMiner: {e}")
        print(f"Error initializing QuantumMiner: {e}")
        print("Please check your configuration and ensure the HDD is properly connected.")
        sys.exit(1)
    
    # Set decoherence rate
    miner.set_config('quantum.decoherence_rate', args.decoherence)
    miner.set_config('quantum.auto_adjust', args.auto_adjust)
    
    try:
        # Start miner
        miner.start()
        
        # Main loop
        while True:
            try:
                # Print statistics
                stats = miner.get_stats()
                logger.info(f"Hash rate: {stats['hash_rate']:.2f} H/s, "
                           f"Shares: {stats['shares_accepted']}/{stats['shares_submitted']}, "
                           f"Workers: {stats['worker_count']}, "
                           f"Decoherence: {stats['decoherence_rate']:.3f}")
                           
                time.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
                
    finally:
        # Stop miner
        miner.stop()
        logger.info("Quantum miner stopped")

def run_gui():
    """Run in GUI mode."""
    logger.info("Starting quantum miner in GUI mode")
    
    if not GUI_AVAILABLE:
        logger.error("GUI mode is not available. Please install PySide6 and pyqtgraph.")
        print("GUI mode is not available. Please install PySide6 and pyqtgraph.")
        print("You can run in CLI mode instead: python quantum_miner.py")
        sys.exit(1)
        
    app = QApplication(sys.argv)
    window = QuantumMinerGUI()
    window.show()
    sys.exit(app.exec())

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Print banner
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   Quantum-Accelerated Bitcoin Miner                           ║
    ║   Harnessing Decoherence for Enhanced Mining Performance      ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Run in appropriate mode
    if args.gui:
        if not GUI_AVAILABLE:
            print("GUI mode is not available. Please install PySide6 and pyqtgraph.")
            print("You can run in CLI mode instead: python quantum_miner.py")
            sys.exit(1)
        run_gui()
    else:
        run_cli(args)

if __name__ == "__main__":
    main()