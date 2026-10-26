#!/usr/bin/env python3
"""
Simple script to run the Student Dropout Prediction System
"""

import subprocess
import sys
import webbrowser
import os
from threading import Timer

def open_browser(port):
    """Open browser after a short delay"""
    webbrowser.open(f'http://localhost:{port}')

if __name__ == "__main__":
    port = os.environ.get('PORT', '5000')
    print("Starting Student Dropout Prediction System...")
    print(f"Opening browser at http://localhost:{port}")

    # Open browser after 2 seconds
    Timer(2.0, open_browser, args=(port,)).start()

    # Start Flask app
    subprocess.run([sys.executable, "app.py"])
