import argparse
import os
import shutil
import site
import subprocess
import sys
import sysconfig
import tarfile
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ARTIFACT_NAME = "opencv-cuda-release.tar.gz"
DEFAULT_ARTIFACT_URL = (
    "https://github.com/JJassonn69/ComfyUI-Stream-Pack/releases/download/v2.1/"
    "opencv-cuda-release.tar.gz"
)
DEFAULT_LIBRARY_DIRS = ["/usr/lib/x86_64-linux-gnu"]
DEFAULT_PACKAGE_NAME = "opencv-cuda"
DEFAULT_PACKAGE_VERSION = "0.1.0"
DEFAULT_NUMPY_REQUIREMENT = "numpy<2.0.0"


@dataclass
class OpencvCudaConfig:
    artifact_url: str
    artifact_name: str
    package_name: str
    package_version: str
    library_dirs: list[str]


def load_config(project_root: Path) -> OpencvCudaConfig:
    config_path = project_root / "pyproject.toml"
    config_data: dict[str, object] = {}
    if config_path.exists():
        with config_path.open("rb") as infile:
            parsed = tomllib.load(infile)
            config_data = parsed.get("tool", {}).get("comfystream", {}).get("opencv_cuda", {})
    artifact_url = os.environ.get(
        "COMFYSTREAM_OPENCV_CUDA_URL",
        config_data.get("artifact_url", DEFAULT_ARTIFACT_URL),
    )
    artifact_name = os.environ.get(
        "COMFYSTREAM_OPENCV_CUDA_NAME",
        config_data.get("artifact_name", DEFAULT_ARTIFACT_NAME),
    )
    package_name = os.environ.get(
        "COMFYSTREAM_OPENCV_PACKAGE_NAME",
        config_data.get("package_name", DEFAULT_PACKAGE_NAME),
    )
    package_version = config_data.get(
        "package_version",
        DEFAULT_PACKAGE_VERSION,
    )
    library_dirs = config_data.get(
        "library_dirs",
        DEFAULT_LIBRARY_DIRS,
    )
    return OpencvCudaConfig(
        artifact_url=artifact_url,
        artifact_name=artifact_name,
        package_name=package_name,
        package_version=package_version,
        library_dirs=library_dirs,
    )


def download_artifact(source_url: str, destination: Path) -> None:
    """Download artifact from URL to destination path."""
    destination = destination.resolve()
    if destination.exists():
        print(f"Found cached artifact at {destination}")
        return

    # Ensure parent directory exists
    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading OpenCV CUDA artifact from {source_url}...")
    try:
        with urllib.request.urlopen(source_url) as response:
            # Verify we got a successful response
            if response.status != 200:
                raise urllib.error.HTTPError(
                    source_url,
                    response.status,
                    "Failed to download artifact",
                    response.headers,
                    None,
                )

            # Download to temporary file first, then rename (atomic operation)
            temp_destination = destination.with_suffix(destination.suffix + ".tmp")
            with temp_destination.open("wb") as out:
                shutil.copyfileobj(response, out)

            # Verify file was written and has content
            if not temp_destination.exists() or temp_destination.stat().st_size == 0:
                temp_destination.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Downloaded file is empty or does not exist: {temp_destination}"
                )

            # Atomic rename
            temp_destination.replace(destination)

            # Final verification
            if not destination.exists():
                raise RuntimeError(f"Failed to save artifact to {destination}")

            file_size = destination.stat().st_size
            print(f"Saved artifact to {destination} ({file_size:,} bytes)")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download artifact from {source_url}: {e}") from e


def extract_artifact(archive: Path, target_dir: Path) -> None:
    """Extract archive to target directory."""
    archive = archive.resolve()
    target_dir = target_dir.resolve()

    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")

    if archive.stat().st_size == 0:
        raise ValueError(f"Archive is empty: {archive}")

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting OpenCV CUDA artifact from {archive}...")
    try:
        with tarfile.open(archive) as tar:
            tar.extractall(target_dir)
        print(f"Extracted contents to {target_dir}")
    except tarfile.TarError as e:
        raise RuntimeError(f"Failed to extract archive {archive}: {e}") from e


def site_package_dirs() -> list[Path]:
    paths: set[Path] = set()
    for key in ("purelib", "platlib"):
        path = sysconfig.get_path(key)
        if path:
            paths.add(Path(path))
    extra_packages: list[str] = []
    try:
        extra_packages = site.getsitepackages()
    except AttributeError:
        pass
    for entry in extra_packages:
        if entry:
            paths.add(Path(entry))
    return [path for path in paths if path.exists()]


def remove_leftover_packages(paths: list[Path]) -> None:
    patterns = ("cv2", "opencv-cuda", "opencv_cuda")
    for base in paths:
        for pattern in patterns:
            candidate = base / pattern
            if candidate.exists():
                if candidate.is_dir():
                    shutil.rmtree(candidate)
                else:
                    candidate.unlink()
        for dist_info in base.glob("cv2*.dist-info"):
            shutil.rmtree(dist_info, ignore_errors=True)
        for dist_info in base.glob("opencv*-dist-info"):
            shutil.rmtree(dist_info, ignore_errors=True)


def pip_uninstall_existing(config: OpencvCudaConfig) -> None:
    packages = [
        "opencv-python",
        "opencv-python-headless",
        "opencv-contrib-python",
        config.package_name,
    ]
    for package in packages:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                package,
            ],
            check=False,
        )
    cleanup_paths = site_package_dirs()
    remove_leftover_packages(cleanup_paths)


def pip_install_package(config: OpencvCudaConfig, cv2_source: Path) -> None:
    """Install cv2 package directly from extracted folder using pip install."""
    # Create setup.py in parent directory (same level as cv2 folder)
    setup_py = cv2_source.parent / "setup.py"
    setup_py.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "from pathlib import Path",
                "from setuptools import setup",
                "",
                "",
                "def package_files(package_dir: str) -> list[str]:",
                "    base = Path(package_dir)",
                "    files: list[str] = []",
                "    for path in base.rglob('*'):",
                "        if path.is_file():",
                # ensure relative path to package root
                "            files.append(str(path.relative_to(base)))",
                "    return files",
                "",
                "setup(",
                f"    name={config.package_name!r},",
                f"    version={config.package_version!r},",
                '    packages=["cv2"],',
                '    package_dir={"cv2": "cv2"},',
                "    package_data={'cv2': package_files('cv2')},",
                "    include_package_data=True,",
                "    zip_safe=False,",
                ")",
            ]
        )
    )

    # Install from the directory containing cv2 folder
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--upgrade",
            "--force-reinstall",
            str(cv2_source.parent),
        ],
        check=True,
        cwd=str(cv2_source.parent),
    )


def ensure_numpy_dependency() -> None:
    """Ensure numpy is available for cv2 bindings."""
    try:
        import numpy  # type: ignore  # noqa: F401

        return
    except ModuleNotFoundError:
        numpy_spec = os.environ.get("COMFYSTREAM_NUMPY_SPEC", DEFAULT_NUMPY_REQUIREMENT)
        print(f"Installing numpy dependency ({numpy_spec}) for cv2...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                numpy_spec,
            ],
            check=True,
        )


def artifact_library_directories(extracted_dir: Path) -> list[Path]:
    """Return possible library directories within extracted artifact."""
    return [
        extracted_dir / "lib",
        extracted_dir / "opencv" / "build" / "lib",
    ]


def copy_libraries(extracted_dir: Path, target_dirs: list[str]) -> None:
    """Copy OpenCV CUDA libraries from artifact to system library directories."""
    lib_locations = artifact_library_directories(extracted_dir)
    lib_dir = next((loc for loc in lib_locations if loc.exists()), None)
    if not lib_dir:
        print("No lib directory found in expected locations, skipping library copy.")
        print(f"Checked: {[str(loc) for loc in lib_locations]}")
        return

    for lib_target in target_dirs:
        destination = Path(lib_target)
        destination.mkdir(parents=True, exist_ok=True)
        for so_file in lib_dir.glob("*.so*"):
            shutil.copy2(so_file, destination / so_file.name)
    print(f"Copied OpenCV CUDA libraries from {lib_dir} to {', '.join(target_dirs)}")


def verify_installation(
    config: OpencvCudaConfig, artifacts_dir: Path, skip_libraries: bool
) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "show",
            config.package_name,
        ],
        check=True,
    )
    if skip_libraries:
        print(
            "Skipping cv2 import verification because --skip-libraries was specified. "
            "Re-run without --skip-libraries once the libraries are available on this system."
        )
        return

    env = os.environ.copy()
    ld_paths: list[str] = []
    # Prefer freshly extracted libs first
    for candidate in artifact_library_directories(artifacts_dir):
        if candidate.exists():
            ld_paths.append(str(candidate))
    # Additional library dirs from config (typically system paths)
    ld_paths.extend(config.library_dirs)
    if ld_paths:
        existing = env.get("LD_LIBRARY_PATH")
        if existing:
            ld_paths.append(existing)
        env["LD_LIBRARY_PATH"] = ":".join(ld_paths)

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import cv2; print(cv2.__version__)",
        ],
        check=True,
        env=env,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install the custom OpenCV CUDA build from precompiled artifacts."
    )
    parser.add_argument(
        "--cache-path",
        default=os.environ.get("CACHE_PATH", "/tmp/comfystream-opencv-cache"),
        help="Cache directory for downloaded artifacts (default: /tmp/comfystream-opencv-cache).",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Path to already-extracted artifacts directory (skips download/extract).",
    )
    parser.add_argument(
        "--skip-libraries",
        action="store_true",
        help="Skip copying OpenCV libraries to system directories.",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[3]
    config = load_config(project_root)

    if args.artifacts_dir:
        # Use already-extracted directory
        artifacts_dir = Path(args.artifacts_dir).resolve()
        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
        print(f"Using already-extracted artifacts from: {artifacts_dir}")
    else:
        # Download and extract to pkgs directory
        cache_path = Path(args.cache_path).resolve()
        pkgs_dir = cache_path / "pkgs"
        # Ensure cache path and pkgs directory exist
        if not cache_path.exists():
            cache_path.mkdir(parents=True, exist_ok=True)
            print(f"Created cache directory at {cache_path}")
        if not pkgs_dir.exists():
            pkgs_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created pkgs directory at {pkgs_dir}")
        else:
            print(f"Using existing pkgs directory at {pkgs_dir}")

        # Resolve paths to ensure consistency
        artifact_path = (cache_path / config.artifact_name).resolve()
        artifacts_dir = pkgs_dir.resolve()

        # Backward compatibility: move cached artifact if stored inside pkgs_dir previously
        legacy_artifact_path = (pkgs_dir / config.artifact_name).resolve()
        if (
            legacy_artifact_path != artifact_path
            and legacy_artifact_path.exists()
            and not artifact_path.exists()
        ):
            print("Detected cached artifact in legacy location, moving to new cache path...")
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_artifact_path.replace(artifact_path)

        # Download artifact
        download_artifact(config.artifact_url, artifact_path)

        # Verify artifact exists before extraction
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact was not downloaded successfully: {artifact_path}")

        # Extract artifact
        extract_artifact(artifact_path, artifacts_dir)

    pip_uninstall_existing(config)
    cv2_source = artifacts_dir / "cv2"
    if not cv2_source.exists():
        raise FileNotFoundError(
            f"Expected cv2 package at {cv2_source}, extracted artifact may be corrupt."
        )
    pip_install_package(config, cv2_source)
    ensure_numpy_dependency()

    if not args.skip_libraries:
        copy_libraries(artifacts_dir, config.library_dirs)
    else:
        print("Skipping library copy (--skip-libraries specified)")

    verify_installation(config, artifacts_dir, args.skip_libraries)


if __name__ == "__main__":
    main()
