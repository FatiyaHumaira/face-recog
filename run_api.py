#!/usr/bin/env python
"""
Script untuk menjalankan Face Recognition FastAPI Server
Gunakan: python run_api.py
"""

import subprocess
import sys
import os

def main():
    print("=" * 60)
    print(" Face Recognition FastAPI Server".center(60))
    print("=" * 60)
    print()
    
    # Check if required packages are installed
    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    print("Starting API Server...")
    print("Server will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative Docs: http://localhost:8000/redoc")
    print()
    
    # Run the FastAPI server
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "api.app:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])


if __name__ == "__main__":
    main()
