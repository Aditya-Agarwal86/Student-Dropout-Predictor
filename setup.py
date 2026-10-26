#!/usr/bin/env python3
"""
Setup and Environment Verification Script for Student Dropout Prediction System
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[-] Python 3.8 or higher is required")
        print(f"    Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"[+] Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['templates', 'static/css', 'static/js', 'uploads', 'tests']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[+] Directory verified: {directory}")

def install_requirements():
    """Install Python requirements"""
    try:
        print("[*] Installing Python requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[+] Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[-] Failed to install requirements: {e}")
        return False
    except FileNotFoundError:
        print("[-] requirements.txt not found")
        return False

def test_imports():
    """Test if all required modules can be imported"""
    required_modules = [
        'flask', 'pandas', 'numpy', 'sklearn', 
        'plotly', 'matplotlib', 'seaborn', 'openpyxl', 'dotenv', 'joblib'
    ]

    failed_imports = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"[+] Module available: {module}")
        except ImportError:
            print(f"[-] Missing module: {module}")
            failed_imports.append(module)

    return len(failed_imports) == 0

def check_files():
    """Check if all required files exist"""
    required_files = [
        'app.py',
        'requirements.txt',
        'Data.csv',
        'sample_student_data.csv',
        'templates/base.html',
        'templates/index.html',
        'templates/dashboard.html',
        'templates/student_detail.html'
    ]

    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"[+] File found: {file}")
        else:
            print(f"[-] Missing file: {file}")
            missing_files.append(file)

    return len(missing_files) == 0

def run_ml_tests():
    """Run the ML test script"""
    try:
        print("[*] Running ML functionality tests...")
        result = subprocess.run([sys.executable, "test_ml.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("[+] ML tests passed")
            return True
        else:
            print("[-] ML tests failed")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("[!] test_ml.py not found, skipping ML tests")
        return True

def main():
    print("=" * 60)
    print("Student Dropout Prediction System - Setup Verification")
    print("=" * 60)

    if not check_python_version():
        sys.exit(1)

    create_directories()

    if not check_files():
        print("[-] Required files are missing. Please ensure full repository is present.")
        sys.exit(1)

    if not test_imports():
        print("[*] Attempting to install missing requirements...")
        install_requirements()

    if not run_ml_tests():
        print("[-] Setup verification failed on ML tests.")
        sys.exit(1)

    print("\n[+] Environment is completely verified and ready!")
    print("    To start the application, run: python app.py or python run.py")

if __name__ == "__main__":
    main()
