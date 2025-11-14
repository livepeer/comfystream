#!/usr/bin/env python3
"""
Verify OpenCV CUDA installation for ComfyStream.

This script checks:
1. OpenCV (cv2) is installed
2. OpenCV has CUDA support compiled
3. CUDA devices are available
4. Basic CUDA operations work
"""

import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_opencv_installed():
    """Check if OpenCV is installed."""
    try:
        import cv2

        logger.info(f"✓ OpenCV installed: version {cv2.__version__}")
        return True, cv2
    except ImportError:
        logger.error("✗ OpenCV is not installed")
        logger.error("  Install with: python src/comfystream/scripts/install_opencv_cuda.py")
        return False, None


def check_opencv_cuda_support(cv2):
    """Check if OpenCV was compiled with CUDA support."""
    if not hasattr(cv2, "cuda"):
        logger.error("✗ OpenCV does not have CUDA support")
        logger.error("  OpenCV was not compiled with CUDA")
        logger.error("  Install with: python src/comfystream/scripts/install_opencv_cuda.py")
        return False

    logger.info("✓ OpenCV has CUDA support compiled")
    return True


def check_cuda_devices(cv2):
    """Check if CUDA devices are available."""
    try:
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        if device_count > 0:
            logger.info(f"✓ CUDA devices available: {device_count}")

            # Get device info for each device
            for i in range(device_count):
                try:
                    cv2.cuda.setDevice(i)
                    device_info = cv2.cuda.getDevice()
                    logger.info(f"  Device {i}: ID {device_info}")
                except Exception as e:
                    logger.warning(f"  Could not get info for device {i}: {e}")

            return True
        else:
            logger.error("✗ No CUDA-enabled devices found")
            logger.error("  Make sure you have an NVIDIA GPU and CUDA installed")
            return False
    except Exception as e:
        logger.error(f"✗ Error checking CUDA devices: {e}")
        return False


def test_cuda_operations(cv2):
    """Test basic CUDA operations."""
    try:
        import numpy as np

        logger.info("Testing basic CUDA operations...")

        # Create a simple test image
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Upload to GPU
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(test_img)
        logger.info("  ✓ Upload to GPU successful")

        # Download from GPU
        result = gpu_img.download()
        logger.info("  ✓ Download from GPU successful")

        # Verify data integrity
        if np.array_equal(test_img, result):
            logger.info("  ✓ Data integrity verified")
            return True
        else:
            logger.error("  ✗ Data integrity check failed")
            return False

    except Exception as e:
        logger.error(f"✗ CUDA operations test failed: {e}")
        return False


def check_package_info(cv2):
    """Display additional OpenCV package information."""
    logger.info("\nOpenCV Package Information:")
    logger.info(f"  Version: {cv2.__version__}")

    try:
        build_info = cv2.getBuildInformation()

        # Extract key information
        for line in build_info.split("\n"):
            line = line.strip()
            if any(keyword in line for keyword in ["CUDA", "cuDNN", "NVIDIA"]):
                logger.info(f"  {line}")
    except Exception as e:
        logger.warning(f"  Could not get build information: {e}")


def main():
    """Run all verification checks."""
    logger.info("=" * 70)
    logger.info("ComfyStream OpenCV CUDA Verification")
    logger.info("=" * 70)
    logger.info("")

    # Track results
    all_passed = True

    # Check OpenCV installation
    opencv_ok, cv2 = check_opencv_installed()
    if not opencv_ok:
        logger.error("\n" + "=" * 70)
        logger.error("VERIFICATION FAILED: OpenCV is not installed")
        logger.error("=" * 70)
        return 1

    logger.info("")

    # Check CUDA support
    cuda_support_ok = check_opencv_cuda_support(cv2)
    if not cuda_support_ok:
        all_passed = False

    logger.info("")

    # Check CUDA devices (only if CUDA support exists)
    if cuda_support_ok:
        cuda_devices_ok = check_cuda_devices(cv2)
        if not cuda_devices_ok:
            all_passed = False

        logger.info("")

        # Test CUDA operations (only if devices available)
        if cuda_devices_ok:
            cuda_ops_ok = test_cuda_operations(cv2)
            if not cuda_ops_ok:
                all_passed = False

            logger.info("")

    # Display package info
    check_package_info(cv2)

    # Final result
    logger.info("")
    logger.info("=" * 70)
    if all_passed:
        logger.info("✓ VERIFICATION PASSED: OpenCV CUDA is properly installed")
        logger.info("=" * 70)
        return 0
    else:
        logger.error("✗ VERIFICATION FAILED: Issues detected with OpenCV CUDA")
        logger.error("=" * 70)
        logger.error("")
        logger.error("To fix, run: python src/comfystream/scripts/install_opencv_cuda.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
