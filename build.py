#!/usr/bin/env python3
"""
Build script for comfystream
"""

import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a command and exit on failure."""
    # Find the full path to the executable on Windows
    if cmd and not Path(cmd[0]).is_absolute():
        executable_path = shutil.which(cmd[0])
        if executable_path:
            cmd[0] = executable_path
        else:
            print(f"❌ Error: Could not find executable '{cmd[0]}' in PATH")
            sys.exit(1)

    try:
        result = subprocess.run(
            cmd, cwd=cwd, check=True, capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {' '.join(cmd)}")
        if e.stderr:
            print(f"Error output: {e.stderr.strip()}")
        sys.exit(1)


def main() -> None:
    """Main build function."""
    print("🚀 Building comfystream...")

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print(
            "❌ Error: pyproject.toml not found. Please run this script from the project root."
        )
        sys.exit(1)

    # Build UI
    print("📦 Building UI...")
    ui_dir = Path("ui")

    if not ui_dir.exists():
        print("❌ Error: ui directory not found")
        sys.exit(1)

    # Always run npm install to ensure dependencies are up to date
    print("📥 Installing UI dependencies...")
    run_command(["npm", "install"], cwd=ui_dir)

    # Build the UI
    print("🔨 Building UI assets...")
    run_command(["npm", "run", "build"], cwd=ui_dir)

    # Check if build was successful
    # Next.js builds to .next or out directory depending on config
    next_dir = ui_dir / ".next"
    out_dir = ui_dir / "out"
    if not next_dir.exists() and not out_dir.exists():
        print("❌ Error: UI build failed - build directory not found")
        sys.exit(1)

    print("✅ UI build completed successfully")


if __name__ == "__main__":
    main()

