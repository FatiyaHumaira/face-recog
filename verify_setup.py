#!/usr/bin/env python
"""
Setup Verification Script
Gunakan: python verify_setup.py
"""

import os
import sys
import subprocess
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def check(description, condition, details=""):
    status = f"{Colors.GREEN}✓{Colors.RESET}" if condition else f"{Colors.RED}✗{Colors.RESET}"
    print(f"{status} {description}")
    if details:
        print(f"  → {details}")
    return condition

def section(title):
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}{title:^60}{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")

def verify_python():
    section("1. Python Environment")
    
    version = sys.version_info
    py_version = f"{version.major}.{version.minor}.{version.micro}"
    
    checks = []
    checks.append(check(
        "Python version",
        version.major >= 3 and version.minor >= 8,
        f"Current: Python {py_version}"
    ))
    
    return all(checks)

def verify_dependencies():
    section("2. Dependencies")
    
    required_packages = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('insightface', 'InsightFace'),
        ('requests', 'Requests'),
        ('pydantic', 'Pydantic'),
    ]
    
    checks = []
    for package, name in required_packages:
        try:
            __import__(package)
            checks.append(check(f"{name} installed", True))
        except ImportError:
            checks.append(check(f"{name} installed", False, "Run: pip install -r requirements.txt"))
    
    return all(checks)

def verify_structure():
    section("3. Project Structure")
    
    required_files = [
        'requirements.txt',
        'config.py',
        'api/app.py',
        'api/models.py',
        'core/face_recog.py',
        'run_api.py',
    ]
    
    required_dirs = [
        'api',
        'core',
        'face_db',
    ]
    
    checks = []
    
    for file in required_files:
        exists = Path(file).exists()
        checks.append(check(f"File: {file}", exists))
    
    for dir_name in required_dirs:
        exists = Path(dir_name).is_dir()
        checks.append(check(f"Directory: {dir_name}", exists))
    
    return all(checks)

def verify_documentation():
    section("4. Documentation")
    
    docs = [
        'README_FASTAPI.md',
        'API_DOCUMENTATION.md',
        'REQUEST_RESPONSE_SPEC.md',
        'QUICK_REFERENCE.md',
        'DEPLOYMENT_GUIDE.md',
    ]
    
    checks = []
    for doc in docs:
        exists = Path(doc).exists()
        checks.append(check(f"Doc: {doc}", exists))
    
    return all(checks)

def verify_face_db():
    section("5. Face Database")
    
    face_db_dir = Path('face_db')
    checks = []
    
    checks.append(check(
        "face_db directory exists",
        face_db_dir.exists()
    ))
    
    if face_db_dir.exists():
        npy_files = list(face_db_dir.glob('*.npy'))
        count = len(npy_files)
        checks.append(check(
            f"Registered staff members",
            count > 0,
            f"Found: {count} staff"
        ))
        
        if npy_files:
            staff_list = [f.stem for f in npy_files]
            print(f"  Registered: {', '.join(staff_list)}")
    
    return all(checks)

def verify_imports():
    section("6. Code Imports")
    
    checks = []
    
    try:
        from api.app import app
        checks.append(check("FastAPI app imports", True))
    except Exception as e:
        checks.append(check("FastAPI app imports", False, str(e)))
    
    try:
        from api.models import RegisterRequest, RecognitionResponse
        checks.append(check("API models import", True))
    except Exception as e:
        checks.append(check("API models import", False, str(e)))
    
    try:
        from core.face_recog import FaceRecognition
        checks.append(check("Face recognition class import", True))
    except Exception as e:
        checks.append(check("Face recognition class import", False, str(e)))
    
    try:
        from config import Config
        checks.append(check("Config import", True))
    except Exception as e:
        checks.append(check("Config import", False, str(e)))
    
    return all(checks)

def verify_configurations():
    section("7. Configuration")
    
    checks = []
    
    try:
        from config import Config, current_config
        
        checks.append(check(
            "Config class loaded",
            Config is not None,
            f"Environment: {type(current_config).__name__}"
        ))
        
        checks.append(check(
            "Face DB path configured",
            hasattr(current_config, 'FACE_DB_PATH'),
            f"Path: {current_config.FACE_DB_PATH}"
        ))
        
        checks.append(check(
            "Recognition threshold set",
            hasattr(current_config, 'RECOGNITION_THRESHOLD'),
            f"Value: {current_config.RECOGNITION_THRESHOLD}"
        ))
        
        checks.append(check(
            "API port configured",
            hasattr(current_config, 'API_PORT'),
            f"Port: {current_config.API_PORT}"
        ))
        
    except Exception as e:
        checks.append(check("Configuration loading", False, str(e)))
    
    return all(checks)

def quick_test():
    section("8. Quick API Test")
    
    try:
        from api.app import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        health_ok = check("Health endpoint", response.status_code == 200)
        
        # Test registered staff endpoint
        response = client.get("/registered-staff")
        staff_ok = check("Get registered staff endpoint", response.status_code == 200)
        
        return health_ok and staff_ok
        
    except Exception as e:
        check("API testing", False, str(e))
        return False

def main():
    print(f"\n{Colors.BLUE}{'*'*60}{Colors.RESET}")
    print(f"{Colors.BLUE}{'Face Recognition API - Setup Verification':^60}{Colors.RESET}")
    print(f"{Colors.BLUE}{'*'*60}{Colors.RESET}")
    
    results = {
        "Python Environment": verify_python(),
        "Dependencies": verify_dependencies(),
        "Project Structure": verify_structure(),
        "Documentation": verify_documentation(),
        "Face Database": verify_face_db(),
        "Code Imports": verify_imports(),
        "Configuration": verify_configurations(),
        "Quick Tests": quick_test(),
    }
    
    section("Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for section_name, result in results.items():
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if result else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"{status} - {section_name}")
    
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"Result: {passed}/{total} checks passed")
    
    if passed == total:
        print(f"{Colors.GREEN}✓ Setup verification successful!{Colors.RESET}")
        print(f"\nRun the API with: {Colors.YELLOW}python run_api.py{Colors.RESET}")
        print(f"Access docs at: {Colors.YELLOW}http://localhost:8000/docs{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}✗ Some checks failed. Please fix the issues above.{Colors.RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
