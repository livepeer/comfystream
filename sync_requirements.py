#!/usr/bin/env python3
"""
Sync script to keep requirements.txt and pyproject.toml dependencies in sync.

Usage:
  python sync_requirements.py --from-requirements  # Update pyproject.toml from requirements.txt
  python sync_requirements.py --from-pyproject     # Update requirements.txt from pyproject.toml
"""

import argparse
import re
import sys
from pathlib import Path


def read_requirements_txt() -> list[str]:
    """Read and parse requirements.txt"""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ Error: requirements.txt not found")
        sys.exit(1)

    dependencies = []
    with open(requirements_file) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                dependencies.append(line)
    return dependencies


def read_pyproject_dependencies() -> list[str]:
    """Read dependencies from pyproject.toml"""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    pyproject_file = Path("pyproject.toml")
    if not pyproject_file.exists():
        print("❌ Error: pyproject.toml not found")
        sys.exit(1)

    with open(pyproject_file, 'rb') as f:
        data = tomllib.load(f)

    dependencies = data.get('project', {}).get('dependencies', [])
    if not dependencies:
        print("❌ Error: No dependencies found in pyproject.toml")
        sys.exit(1)

    return dependencies


def update_pyproject_dependencies(dependencies: list[str]) -> None:
    """Update dependencies in pyproject.toml"""
    pyproject_file = Path("pyproject.toml")
    with open(pyproject_file) as f:
        content = f.read()

    # Format dependencies for TOML
    deps_lines = ['dependencies = [']
    for dep in dependencies:
        deps_lines.append(f'    "{dep}",')
    deps_lines.append(']')
    new_deps = '\n'.join(deps_lines)

    # Replace dependencies section
    new_content = re.sub(
        r'dependencies\s*=\s*\[.*?\]',
        new_deps,
        content,
        flags=re.DOTALL
    )

    with open(pyproject_file, 'w') as f:
        f.write(new_content)

    print("✅ Updated pyproject.toml dependencies")


def update_requirements_txt(dependencies: list[str]) -> None:
    """Update requirements.txt"""
    requirements_file = Path("requirements.txt")
    with open(requirements_file, 'w') as f:
        for dep in dependencies:
            f.write(f"{dep}\n")
    print("✅ Updated requirements.txt")


def main() -> None:
    """Main sync function"""
    parser = argparse.ArgumentParser(
        description="Sync dependencies between requirements.txt and pyproject.toml"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--from-requirements",
        action="store_true",
        help="Update pyproject.toml from requirements.txt"
    )
    group.add_argument(
        "--from-pyproject",
        action="store_true",
        help="Update requirements.txt from pyproject.toml"
    )

    args = parser.parse_args()

    if args.from_requirements:
        print("📝 Reading from requirements.txt...")
        dependencies = read_requirements_txt()
        print(f"Found {len(dependencies)} dependencies")
        update_pyproject_dependencies(dependencies)
    else:
        print("📝 Reading from pyproject.toml...")
        dependencies = read_pyproject_dependencies()
        print(f"Found {len(dependencies)} dependencies")
        update_requirements_txt(dependencies)


if __name__ == "__main__":
    main()

