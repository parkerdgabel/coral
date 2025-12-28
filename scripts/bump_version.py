#!/usr/bin/env python3
"""
Version bump script for Coral.

Updates version in:
- pyproject.toml
- src/coral/__init__.py

Usage:
    ./scripts/bump_version.py 1.2.3
    ./scripts/bump_version.py --check
"""

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
INIT_PATH = PROJECT_ROOT / "src" / "coral" / "__init__.py"


def get_current_versions() -> dict[str, str]:
    """Get current versions from all files."""
    versions = {}

    # Get version from pyproject.toml
    pyproject_content = PYPROJECT_PATH.read_text()
    match = re.search(r'^version = "([^"]+)"', pyproject_content, re.MULTILINE)
    if match:
        versions["pyproject.toml"] = match.group(1)

    # Get version from __init__.py
    init_content = INIT_PATH.read_text()
    match = re.search(r'^__version__ = "([^"]+)"', init_content, re.MULTILINE)
    if match:
        versions["__init__.py"] = match.group(1)

    return versions


def validate_version(version: str) -> bool:
    """Validate version string format."""
    # Matches: 1.0.0, 1.0.0-alpha.1, 1.0.0-beta.2, 1.0.0-rc.1
    pattern = r"^\d+\.\d+\.\d+(-(?:alpha|beta|rc)\.\d+)?$"
    return bool(re.match(pattern, version))


def update_pyproject(version: str) -> None:
    """Update version in pyproject.toml."""
    content = PYPROJECT_PATH.read_text()
    new_content = re.sub(
        r'^version = "[^"]+"',
        f'version = "{version}"',
        content,
        flags=re.MULTILINE,
    )
    PYPROJECT_PATH.write_text(new_content)
    print(f"  Updated pyproject.toml: {version}")


def update_init(version: str) -> None:
    """Update version in __init__.py."""
    content = INIT_PATH.read_text()
    new_content = re.sub(
        r'^__version__ = "[^"]+"',
        f'__version__ = "{version}"',
        content,
        flags=re.MULTILINE,
    )
    INIT_PATH.write_text(new_content)
    print(f"  Updated src/coral/__init__.py: {version}")


def check_versions() -> int:
    """Check if all versions are in sync."""
    versions = get_current_versions()

    if not versions:
        print("Error: Could not find any version strings")
        return 1

    unique_versions = set(versions.values())

    if len(unique_versions) == 1:
        version = list(unique_versions)[0]
        print(f"All versions in sync: {version}")
        return 0
    else:
        print("Version mismatch detected:")
        for file, version in versions.items():
            print(f"  {file}: {version}")
        return 1


def bump_version(new_version: str) -> int:
    """Bump version in all files."""
    if not validate_version(new_version):
        print(f"Error: Invalid version format: {new_version}")
        print("Expected: X.Y.Z or X.Y.Z-{alpha|beta|rc}.N")
        return 1

    current_versions = get_current_versions()
    current = list(current_versions.values())[0] if current_versions else "unknown"

    print(f"Bumping version: {current} -> {new_version}")
    update_pyproject(new_version)
    update_init(new_version)

    print("\nVersion bump complete!")
    print("\nNext steps:")
    print("  1. Update CHANGELOG.md")
    print("  2. Commit changes:")
    print(f'     git add pyproject.toml src/coral/__init__.py CHANGELOG.md')
    print(f'     git commit -m "chore: release version {new_version}"')
    print(f"  3. Create tag:")
    print(f'     git tag -a v{new_version} -m "Release version {new_version}"')
    print(f"  4. Push:")
    print(f"     git push origin main && git push origin v{new_version}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bump Coral version across all files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 1.2.3           Bump to version 1.2.3
  %(prog)s 2.0.0-beta.1    Bump to pre-release version
  %(prog)s --check         Check if versions are in sync
        """,
    )
    parser.add_argument(
        "version",
        nargs="?",
        help="New version string (e.g., 1.2.3 or 1.2.3-beta.1)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if all version strings are in sync",
    )

    args = parser.parse_args()

    if args.check:
        return check_versions()
    elif args.version:
        return bump_version(args.version)
    else:
        # No arguments: show current versions
        versions = get_current_versions()
        print("Current versions:")
        for file, version in versions.items():
            print(f"  {file}: {version}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
