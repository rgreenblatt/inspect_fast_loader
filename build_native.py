#!/usr/bin/env python3
"""Build the optional Rust native extension for faster .eval file reading.

Usage:
    python build_native.py

Requires a Rust toolchain (https://rustup.rs) and maturin (pip install maturin).
If the Rust extension can't be built, the package works in pure-Python mode.
"""

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
RUST_DIR = ROOT / "rust"
PKG_DIR = ROOT / "inspect_fast_loader"


def main() -> int:
    # Check Rust toolchain
    try:
        result = subprocess.run(["rustc", "--version"], capture_output=True, text=True, check=True)
        print(f"Rust: {result.stdout.strip()}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: Rust toolchain not found. Install from https://rustup.rs")
        return 1

    # Ensure maturin
    try:
        subprocess.run([sys.executable, "-m", "maturin", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Installing maturin...")
        subprocess.run([sys.executable, "-m", "pip", "install", "maturin>=1.0,<2.0"], check=True)

    # Build
    with tempfile.TemporaryDirectory() as tmpdir:
        print("Building native extension...")
        subprocess.run(
            [sys.executable, "-m", "maturin", "build", "--release", "-o", tmpdir,
             "--manifest-path", str(RUST_DIR / "Cargo.toml")],
            check=True, cwd=str(RUST_DIR),
        )

        # Find and extract the .so
        wheels = list(Path(tmpdir).glob("*.whl"))
        assert wheels, "No wheel produced"

        with zipfile.ZipFile(wheels[0]) as whl:
            for name in whl.namelist():
                if name.endswith(".so") or name.endswith(".pyd"):
                    so_filename = Path(name).name
                    dest = PKG_DIR / so_filename
                    with whl.open(name) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    print(f"Installed: {dest}")
                    return 0

    print("ERROR: No .so found in built wheel")
    return 1


if __name__ == "__main__":
    sys.exit(main())
