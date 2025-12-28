#!/usr/bin/env python3
"""
Build script for creating distributable packages of the Personal Assistant.
Uses PyInstaller to create standalone executables.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import shutil

def run_command(cmd, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {cmd}")
        print(f"Error: {e.stderr}")
        return None

def get_platform_info():
    """Get platform-specific information."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        return "windows", f"personal_assistant_windows_{machine}.exe"
    elif system == "linux":
        return "linux", f"personal_assistant_linux_{machine}"
    elif system == "darwin":  # macOS
        return "macos", f"personal_assistant_macos_{machine}"
    else:
        return system, f"personal_assistant_{system}_{machine}"

def create_spec_file():
    """Create PyInstaller spec file."""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

# Add the src directory to the path
src_dir = os.path.join(os.path.dirname(os.path.abspath(SPEC)), 'src')
sys.path.insert(0, src_dir)

block_cipher = None

a = Analysis(
    ['src/main.py'],
    pathex=[src_dir],
    binaries=[],
    datas=[
        ('data', 'data'),  # Include data directory
    ],
    hiddenimports=[
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'cryptography.hazmat.backends.openssl',
        'cryptography.hazmat.primitives.kdf.pbkdf2',
        'langchain_community.llms',
        'langchain_community.embeddings',
        'faiss',
        'memorisdk',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='personal_assistant',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False for production (no console window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''

    spec_file = Path("packaging/personal_assistant.spec")
    with open(spec_file, 'w') as f:
        f.write(spec_content)

    return spec_file

def build_with_pyinstaller():
    """Build the application using PyInstaller."""
    print("Building with PyInstaller...")

    # Create spec file
    spec_file = create_spec_file()
    print(f"Created spec file: {spec_file}")

    # Run PyInstaller
    cmd = f"pyinstaller --clean --noconfirm {spec_file}"
    result = run_command(cmd)

    if result is None:
        print("PyInstaller build failed!")
        return False

    print("PyInstaller build completed successfully!")
    return True

def create_distributable_archive():
    """Create a distributable archive of the built application."""
    platform_name, executable_name = get_platform_info()

    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("No dist directory found. Run PyInstaller build first.")
        return False

    # Find the built executable
    exe_path = None
    for file in dist_dir.iterdir():
        if file.is_file() and ("personal_assistant" in file.name.lower()):
            exe_path = file
            break

    if not exe_path:
        print("Could not find built executable in dist directory")
        return False

    # Create archive name
    archive_name = f"personal_assistant_v0.2.0_{platform_name}"
    archive_path = Path(f"dist/{archive_name}")

    print(f"Creating distributable archive: {archive_name}")

    try:
        if platform.system() == "Windows":
            # Create ZIP archive on Windows
            shutil.make_archive(str(archive_path), 'zip', str(dist_dir))
            archive_file = archive_path.with_suffix('.zip')
        else:
            # Create tar.gz archive on Unix-like systems
            shutil.make_archive(str(archive_path), 'gztar', str(dist_dir))
            archive_file = archive_path.with_suffix('.tar.gz')

        print(f"Created archive: {archive_file}")
        return True

    except Exception as e:
        print(f"Failed to create archive: {e}")
        return False

def main():
    """Main build function."""
    print("Personal Assistant Build Script")
    print("=" * 40)

    # Check if we're in the right directory
    if not Path("src/main.py").exists():
        print("Error: src/main.py not found. Run this script from the project root.")
        sys.exit(1)

    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller not found. Installing...")
        run_command("pip install pyinstaller")
        try:
            import PyInstaller
            print(f"PyInstaller version: {PyInstaller.__version__}")
        except ImportError:
            print("Failed to install PyInstaller. Please install it manually.")
            sys.exit(1)

    # Build with PyInstaller
    if not build_with_pyinstaller():
        sys.exit(1)

    # Create distributable archive
    if not create_distributable_archive():
        sys.exit(1)

    platform_name, _ = get_platform_info()
    print(f"\nBuild completed successfully for {platform_name}!")
    print("Check the 'dist' directory for the built executable and archive.")

if __name__ == "__main__":
    main()
