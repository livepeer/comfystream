from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import os
import sys

# Add the current directory to Python path to import opencv_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opencv_utils import setup_opencv_cuda

class CustomInstallCommand(install):
    def run(self):
        setup_opencv_cuda()
        install.run(self)
setup(
    name='opencv-cuda-superresolution',
    version='4.11.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    cmdclass={
        'install': CustomInstallCommand,
    },
)