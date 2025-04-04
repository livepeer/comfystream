import os
import sys
import subprocess
import shutil
from pathlib import Path
import logging
import tempfile

logger = logging.getLogger(__name__)

def setup_opencv_cuda():
    """
    Set up OpenCV with CUDA support.
    This function downloads and installs the CUDA-enabled OpenCV build.
    Returns:
        bool: True if installation was successful, False otherwise
    """
    try:
        import cv2
        # Check if CUDA is already available
        if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.info("OpenCV CUDA already installed and working")
            return True
    except ImportError:
        pass
    
    try:
        workspace_dir = Path("/workspace/comfystream")
        site_packages_dir = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        
        # Create temporary directory for download and extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Download OpenCV CUDA build
            download_url = "https://github.com/JJassonn69/ComfyUI_SuperResolution/releases/download/v1/opencv-cuda.tar.gz"
            download_name = temp_dir_path / "opencv-cuda-latest.tar.gz"
            
            logger.info("Downloading OpenCV CUDA build...")
            subprocess.run(['wget', '-O', str(download_name), download_url], check=True)
            
            logger.info("Extracting OpenCV CUDA build...")
            subprocess.run(['tar', '-xzf', str(download_name), '-C', str(temp_dir_path)], check=True)
            
            # Move the extracted files to workspace
            extracted_dir = temp_dir_path / "opencv-cuda"
            if extracted_dir.exists():
                target_dir = workspace_dir / "opencv-cuda"
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.move(str(extracted_dir), str(target_dir))
            
            # The temporary directory and its contents will be automatically cleaned up
        
        # Install system dependencies
        # logger.info("Installing system dependencies...")
        # subprocess.run(['apt-get', 'update'], check=True)
        # dependencies = [
        #     'libgflags-dev',
        #     'libgoogle-glog-dev',
        #     'libjpeg-dev',
        #     'libavcodec-dev',
        #     'libavformat-dev',
        #     'libavutil-dev',
        #     'libswscale-dev'
        # ]
        # subprocess.run(['apt-get', 'install', '-y'] + dependencies, check=True)
        
        # Remove existing cv2 package
        cv2_path = site_packages_dir / "cv2"
        if cv2_path.exists():
            shutil.rmtree(cv2_path)
        
        # Copy new cv2 package
        opencv_cuda_path = workspace_dir / "opencv-cuda" / "cv2"
        shutil.copytree(opencv_cuda_path, cv2_path)
        
        # Handle library dependencies
        conda_env_lib = Path(sys.prefix) / "lib"
        
        # Remove existing libstdc++ and copy system one
        for lib in conda_env_lib.glob("libstdc++.so*"):
            lib.unlink()
        
        # Copy system libstdc++
        system_libs = Path("/usr/lib/x86_64-linux-gnu")
        for lib in system_libs.glob("libstdc++.so*"):
            shutil.copy2(lib, conda_env_lib)
        
        # Copy OpenCV libraries
        opencv_libs = workspace_dir / "opencv-cuda" / "opencv" / "build" / "lib"
        for lib in opencv_libs.glob("libopencv_*"):
            shutil.copy2(lib, system_libs)
        
        logger.info("OpenCV CUDA installation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to install OpenCV CUDA: {str(e)}")
        return False

def is_cuda_available():
    """
    Check if CUDA is available in the OpenCV installation.
    Returns:
        bool: True if CUDA is available, False otherwise
    """
    try:
        import cv2
        return hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except ImportError:
        return False 
    
    

def main():
    setup_opencv_cuda()

if __name__ == "__main__":
    main()