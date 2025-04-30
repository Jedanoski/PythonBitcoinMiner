"""
HDD Interface Module for Quantum-Accelerated Bitcoin Miner

This module provides an interface to interact with an external USB 3.0 HDD
using libusb-win32 and PyUSB. It implements SCSI commands for reading and
writing to the HDD, which will be used as a qubit register.
"""

import usb.core
import usb.util
import struct
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('hdd_interface')

# SCSI Command Definitions
SCSI_READ10 = 0x28
SCSI_WRITE10 = 0x2A
SCSI_TEST_UNIT_READY = 0x00
SCSI_INQUIRY = 0x12

class HDDInterface:
    def __init__(self, vid=None, pid=None, auto_detect=True):
        """
        Initialize the HDD Interface.
        
        Args:
            vid (int, optional): Vendor ID of the USB device
            pid (int, optional): Product ID of the USB device
            auto_detect (bool): Whether to auto-detect the first available USB mass storage device
        """
        self.device = None
        self.interface = None
        self.endpoint_in = None
        self.endpoint_out = None
        
        if auto_detect and (vid is None or pid is None):
            self._auto_detect_devices()
        else:
            self._connect_device(vid, pid)
            
        if self.device:
            logger.info(f"Connected to USB device: {self.device}")
            self._setup_interface()
            
    def _auto_detect_devices(self):
        """Auto-detect the first available USB mass storage device."""
        # Find all USB devices with specified Vendor ID and Product ID
        try:
            # --- FIX: Convert generator to list immediately ---
            specific_devices = list(usb.core.find(find_all=True, idVendor=0x03EB, idProduct=0x6124))
            found_devices = specific_devices  # Changed devices to found_devices
        except Exception as e:
            logger.warning(f"Error finding specific devices: {e}")

        # If no specific devices found, search by interface class
        if not found_devices:
            logger.info("No specific VID/PID match, searching by Mass Storage Class...")
            try:
                # --- FIX: Convert generator to list immediately ---
                all_devs = list(usb.core.find(find_all=True))
                for dev in all_devs:
                    try:
                        # Check if already found
                        if dev in found_devices:
                            continue
                        for cfg in dev:
                            for intf in cfg:
                                if intf.bInterfaceClass == 0x08:  # Mass Storage Class
                                    found_devices.append(dev)
                                    # Break inner loops once found for this device
                                    raise StopIteration
                    except usb.core.USBError as e:
                        # Ignore devices we can't access
                        logger.debug(f"Could not access device {dev.idVendor:04x}:{dev.idProduct:04x}: {e}")
                        continue
                    except StopIteration:
                        continue  # Go to the next device
            except Exception as e:
                logger.error(f"Error during generic device scan: {e}")

        if found_devices:
            self.device_objects = found_devices
            # --- FIX: Populate self.devices with identifiers ---
            # Using bus/address as a simple identifier for now, might need refinement
            self.devices = [f"Bus {dev.bus} Addr {dev.address} ({dev.idVendor:04x}:{dev.idProduct:04x})" for dev in found_devices]

            # --- FIX: Assign the first device object correctly ---
            self.device = self.device_objects[0]
            logger.info(f"Found {len(self.devices)} USB Mass Storage device(s). Using first: {self.devices[0]}")
            logger.info(f"Device identifiers: {self.devices}")
        else:
            logger.error("No USB Mass Storage devices found.")
            self.device = None
            self.devices = []
            self.device_objects = []

    def initialize_hdd(self):
        for device in self.devices:
            # Execute SCSI commands to initialize the HDD
            try:
                device.ctrl_transfer(0x21, 0x09, 0x0000, 0x0000, [0x12, 0x00, 0x00, 0x00])
                logger.info(f"HDD initialized successfully for device: {device}")
            except Exception as e:
                logger.error(f"Error initializing HDD for device {device}: {e}")
            
    def _connect_device(self, vid, pid):
        """Connect to a specific USB device using VID and PID."""
        try:
            self.device = usb.core.find(idVendor=vid, idProduct=pid)
            if self.device is None:
                logger.error(f"Device not found with VID: {vid:04x} and PID: {pid:04x}")
            else:
                logger.info(f"Device found with VID: {vid:04x} and PID: {pid:04x}")
        except Exception as e:
            logger.error(f"Error connecting to device: {e}")

    def _setup_interface(self):
        """Set up the USB interface and endpoints."""
        try:
            # Detach kernel driver if active
            if self.device.is_kernel_driver_active(0):
                try:
                    self.device.detach_kernel_driver(0)
                    logger.info("Kernel driver detached")
                except Exception as e:
                    logger.warning(f"Could not detach kernel driver: {e}")
            
            # Set configuration
            self.device.set_configuration()
            
            # Find the Mass Storage interface
            cfg = self.device.get_active_configuration()
            for intf in cfg:
                if intf.bInterfaceClass == 0x08:  # Mass Storage Class
                    self.interface = intf
                    break
            
            if not self.interface:
                logger.error("Mass Storage interface not found")
                return
                
            # Find the bulk endpoints
            for ep in self.interface:
                if usb.util.endpoint_direction(ep.bEndpointAddress) == usb.util.ENDPOINT_IN:
                    self.endpoint_in = ep
                else:
                    self.endpoint_out = ep
                    
            if not self.endpoint_in or not self.endpoint_out:
                logger.error("Bulk endpoints not found")
                
            logger.info(f"Interface set up successfully: {self.interface}")
            logger.info(f"Endpoints: IN={self.endpoint_in.bEndpointAddress:02x}, OUT={self.endpoint_out.bEndpointAddress:02x}")
            
        except Exception as e:
            logger.error(f"Error setting up interface: {e}")
            
    def _send_scsi_command(self, cdb, data_out=None, data_in_length=0, timeout=5000):
        """
        Send a SCSI Command Descriptor Block (CDB) to the device.
        
        Args:
            cdb (bytes): The SCSI Command Descriptor Block
            data_out (bytes, optional): Data to send with the command
            data_in_length (int): Expected length of data to receive
            timeout (int): Timeout in milliseconds
            
        Returns:
            bytes: The data received from the device
        """
        if not self.device or not self.endpoint_in or not self.endpoint_out:
            logger.error("Device or endpoints not initialized")
            return None
            
        # Create Command Block Wrapper (CBW)
        cbw_signature = 0x43425355  # 'USBC'
        cbw_tag = int(time.time() * 1000) & 0xFFFFFFFF
        cbw_data_transfer_length = data_in_length if data_out is None else len(data_out)
        cbw_flags = 0x80 if data_out is None else 0x00  # 0x80 = Device to Host, 0x00 = Host to Device
        cbw_lun = 0
        cbw_cb_length = len(cdb)
        
        cbw = struct.pack('<IIBBBB', 
                          cbw_signature, 
                          cbw_tag, 
                          cbw_data_transfer_length, 
                          cbw_flags, 
                          cbw_lun, 
                          cbw_cb_length) + cdb
        
        try:
            # Send CBW
            bytes_sent = self.device.write(self.endpoint_out.bEndpointAddress, cbw, timeout=timeout)
            logger.debug(f"Sent {bytes_sent} bytes of CBW")
            
            # Send data if any
            if data_out:
                bytes_sent = self.device.write(self.endpoint_out.bEndpointAddress, data_out, timeout=timeout)
                logger.debug(f"Sent {bytes_sent} bytes of data")
                
            # Receive data if expected
            data_in = b''
            if data_in_length > 0:
                try:
                    data_in = self.device.read(self.endpoint_in.bEndpointAddress, data_in_length, timeout=timeout)
                    logger.debug(f"Received {len(data_in)} bytes of data")
                except usb.core.USBError as e:
                    if e.errno == 110:  # Timeout
                        logger.warning("Read timeout, continuing with CSW")
                    else:
                        raise
                        
            # Receive Command Status Wrapper (CSW)
            try:
                csw = self.device.read(self.endpoint_in.bEndpointAddress, 13, timeout=timeout)
                csw_signature, csw_tag, csw_data_residue, csw_status = struct.unpack('<IIIB', csw)
                
                if csw_signature != 0x53425355:  # 'USBS'
                    logger.error(f"Invalid CSW signature: {csw_signature:08x}")
                    return None
                    
                if csw_tag != cbw_tag:
                    logger.error(f"CSW tag mismatch: {csw_tag:08x} != {cbw_tag:08x}")
                    return None
                    
                if csw_status != 0:
                    logger.error(f"Command failed with status: {csw_status}")
                    return None
                    
                logger.debug(f"Command completed successfully, residue: {csw_data_residue}")
                return data_in
                
            except usb.core.USBError as e:
                logger.error(f"Error receiving CSW: {e}")
                return None
                
        except usb.core.USBError as e:
            logger.error(f"USB error during command: {e}")
            
            # Try to recover from STALL condition
            if e.errno == 32:  # EPIPE (Broken pipe) - endpoint is stalled
                try:
                    if data_out:
                        self.device.clear_halt(self.endpoint_out.bEndpointAddress)
                    else:
                        self.device.clear_halt(self.endpoint_in.bEndpointAddress)
                    logger.info("Cleared endpoint halt condition")
                except:
                    logger.error("Failed to clear halt condition")
                    
            return None
            
    def test_unit_ready(self):
        """
        Test if the device is ready to accept commands.
        
        Returns:
            bool: True if the device is ready, False otherwise
        """
        cdb = bytes([SCSI_TEST_UNIT_READY, 0, 0, 0, 0, 0])
        result = self._send_scsi_command(cdb)
        return result is not None
        
    def inquiry(self):
        """
        Send an INQUIRY command to get device information.
        
        Returns:
            dict: Device information including vendor, product, and version
        """
        cdb = bytes([SCSI_INQUIRY, 0, 0, 0, 36, 0])
        data = self._send_scsi_command(cdb, data_in_length=36)
        
        if data:
            vendor = data[8:16].decode('ascii', errors='ignore').strip()
            product = data[16:32].decode('ascii', errors='ignore').strip()
            version = data[32:36].decode('ascii', errors='ignore').strip()
            
            return {
                'vendor': vendor,
                'product': product,
                'version': version
            }
        return None
        
    def scsi_read_lba(self, lba, count=1):
        """
        Read data from the specified Logical Block Address (LBA).
        
        Args:
            lba (int): The Logical Block Address to read from
            count (int): Number of blocks to read
            
        Returns:
            bytes: The data read from the device, or None if the read failed
        """
        count = count  # Define the 'count' variable
        # Standard block size for most HDDs
        block_size = 512
        data_length = block_size * count
        
        # READ(10) command: opcode(1), flags(1), lba(4), group(1), count(2), control(1)
        cdb = struct.pack('>BBIBBB', 
                          SCSI_READ10,    # Opcode
                          0,              # Flags
                          lba,            # LBA (4 bytes, big-endian)
                          0,              # Group
                          count,          # Transfer length (number of blocks)
                          0)              # Control
                          
        logger.info(f"Reading LBA {lba}, count {count}")
        data = self._send_scsi_command(cdb, data_in_length=data_length)
        
        if data is None:
            logger.error(f"Failed to read LBA {lba}")
            return None
            
        logger.info(f"Successfully read {len(data)} bytes from LBA {lba}")
        return data
        
    def scsi_write_lba(self, lba, data, count=1):
        """
        Write data to the specified Logical Block Address (LBA).
        
        Args:
            lba (int): The Logical Block Address to write to
            data (bytes): The data to write
            count (int): Number of blocks to write
            
        Returns:
            int: Number of bytes written, or None if the write failed
        """
        count = count  # Define the 'count' variable
        # Standard block size for most HDDs
        block_size = 512
        data_length = block_size * count
        
        # WRITE(10) command: opcode(1), flags(1), lba(4), group(1), count(2), control(1)
        cdb = struct.pack('>BBIBBB', 
                          SCSI_WRITE10,   # Opcode
                          0,              # Flags
                          lba,            # LBA (4 bytes, big-endian)
                          0,              # Group
                          count,          # Transfer length (number of blocks)
                          0)              # Control
                          
        logger.info(f"Writing {len(data)} bytes to LBA {lba}, count {count}")
        result = self._send_scsi_command(cdb, data_out=data)
        
        if result is None:
            logger.error(f"Failed to write to LBA {lba}")
            return None
            
        logger.info(f"Successfully wrote to LBA {lba}")
        return len(data)
        
    def close(self):
        """Release the USB device."""
        if self.device:
            usb.util.dispose_resources(self.device)
            logger.info("USB device resources released")


# Example usage
if __name__ == "__main__":
    # Test the HDD interface
    hdd = HDDInterface(auto_detect=True)
    
    if hdd.device:
        print("HDD Interface initialized successfully")
        
        # Test if the device is ready
        if hdd.test_unit_ready():
            print("Device is ready")
            
            # Get device information
            info = hdd.inquiry()
            if info:
                print(f"Device: {info['vendor']} {info['product']} (Version: {info['version']})")
                
            # Read from LBA 0
            data = hdd.scsi_read_lba(0)
            if data:
                print(f"Read {len(data)} bytes from LBA 0")
                print(f"First 16 bytes: {data[:16].hex()}")
                
                # Write back the same data (for testing only)
                # WARNING: Writing to LBA 0 can damage the partition table!
                # result = hdd.scsi_write_lba(0, data)
                # if result:
                #     print(f"Wrote {result} bytes to LBA 0")
        else:
            print("Device is not ready")
            
        hdd.close()
    else:
        print("Failed to initialize HDD Interface")