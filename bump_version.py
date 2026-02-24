#!/usr/bin/env python3
"""
Version bump script for poly_arb_bot.
Usage:
  python bump_version.py patch   # 1.0.0 -> 1.0.1
  python bump_version.py minor   # 1.0.0 -> 1.1.0
  python bump_version.py major   # 1.0.0 -> 2.0.0
"""
import sys
import subprocess
from pathlib import Path

VERSION_FILE = Path(__file__).parent / "VERSION"

def get_version() -> str:
    return VERSION_FILE.read_text().strip()

def bump(part: str) -> str:
    current = get_version()
    major, minor, patch = [int(x) for x in current.split(".")]

    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        print(f"Unknown part: {part}. Use: major, minor, patch")
        sys.exit(1)

    new_version = f"{major}.{minor}.{patch}"
    VERSION_FILE.write_text(new_version + "\n")
    return new_version

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Current version: {get_version()}")
        print("Usage: python bump_version.py [major|minor|patch]")
        sys.exit(0)

    part = sys.argv[1].lower()
    old = get_version()
    new = bump(part)
    print(f"Version bumped: {old} -> {new}")

    # Git tag oluştur
    try:
        subprocess.run(["git", "add", "VERSION"], check=True)
        subprocess.run(["git", "commit", "-m", f"bump: v{new}"], check=True)
        subprocess.run(["git", "tag", f"v{new}"], check=True)
        print(f"Git tag v{new} created")
    except subprocess.CalledProcessError:
        print("Git commit/tag skipped (not in git repo or error)")
