from __future__ import annotations

import atexit
import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

_OPENCV_INSTALL_ATTEMPTED = False


def _should_skip_opencv_install() -> bool:
    return os.environ.get("COMFYSTREAM_SKIP_OPENCV_CUDA", "").lower() in {
        "1",
        "true",
        "yes",
    }


def ensure_opencv_cuda_installed() -> None:
    global _OPENCV_INSTALL_ATTEMPTED

    if _OPENCV_INSTALL_ATTEMPTED:
        return
    _OPENCV_INSTALL_ATTEMPTED = True

    if _should_skip_opencv_install():
        print("COMFYSTREAM_SKIP_OPENCV_CUDA is set; skipping OpenCV CUDA installation.")
        return

    try:
        import cv2  # type: ignore  # noqa: F401

        print("OpenCV (cv2) already installed, skipping OpenCV CUDA installation.")
        return
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parent
    script_path = repo_root / "src" / "comfystream" / "scripts" / "install_opencv_cuda.py"
    if not script_path.exists():
        print(f"OpenCV CUDA installer script not found at {script_path}, skipping.")
        print("To install OpenCV CUDA manually, run:")
        print(f"  python {script_path}")
        return

    cache_path = os.environ.get("CACHE_PATH", "/tmp/comfystream-opencv-cache")
    env = os.environ.copy()
    env.setdefault("CACHE_PATH", cache_path)

    print("\n" + "=" * 70)
    print("Installing OpenCV with CUDA support...")
    print("=" * 70)
    try:
        subprocess.check_call(
            [
                sys.executable,
                str(script_path),
                "--cache-path",
                cache_path,
            ],
            env=env,
        )
        print("=" * 70)
        print("OpenCV CUDA installation completed successfully!")
        print("To verify installation, run:")
        print(f"  python {script_path.parent / 'verify_opencv_cuda.py'}")
        print("=" * 70 + "\n")
    except subprocess.CalledProcessError as e:
        print("=" * 70)
        print(f"WARNING: OpenCV CUDA installation failed: {e}")
        print("You can install it manually by running:")
        print(f"  python {script_path}")
        print("=" * 70 + "\n")


class InstallWithOpenCV(install):
    def run(self) -> None:
        super().run()
        ensure_opencv_cuda_installed()


class DevelopWithOpenCV(develop):
    def run(self) -> None:
        super().run()
        ensure_opencv_cuda_installed()


# Register atexit handler to attempt installation at the end of setup
# This provides a fallback for PEP 517 builds where cmdclass might not be called
def _atexit_opencv_install():
    # Only run if we're in the main setup.py process (not in build isolation)
    if not _OPENCV_INSTALL_ATTEMPTED and "pip" in sys.argv[0].lower():
        ensure_opencv_cuda_installed()


atexit.register(_atexit_opencv_install)

setup(
    cmdclass={
        "install": InstallWithOpenCV,
        "develop": DevelopWithOpenCV,
    }
)
