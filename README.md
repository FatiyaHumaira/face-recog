# 🎯 Face Recognition API untuk 1000 Water Channel Doors

Sistem pengenalan wajah berbasis FastAPI + PostgreSQL untuk kontrol akses pintu air.

**⚡ Quick Commands:**
```bash
# Setup database
psql -U postgres -f setup_db.sql

# Register staff from CSV
python bulk_register.py water_channel_officers.csv

# Check missing registrations
python check_registration.py water_channel_officers.csv

# Start API with webhook
export WEBHOOK_ENABLED=true
export WEBHOOK_URL=http://192.168.1.10:9000/api/detection-webhook
python run_api.py

# Test webhook (mock server)
python test_webhook_server.py

# Reset all data
python reset_database.py

# Fix dimension errors
python cleanup_embeddings.py --cleanup
```

**📊 Current Status:**
- ✅ Database: PostgreSQL 14+ (unified storage)
- ✅ Model: InsightFace buffalo_l (512D embeddings)


## 📋 Daftar Isi
- [⚡ Quick Start](#quick-start)
- [📦 Architecture](#architecture)
- [🔧 Setup Lokal](#setup-lokal)
- [🚀 Deployment Produksi](#deployment-produksi)
- [📡 API Reference](#api-reference)
- [🔔 Webhook Integration](#webhook-integration)
- [🛠️ Management Scripts](#management-scripts)
- [⚙️ Konfigurasi](#konfigurasi)
- [🔍 Troubleshooting](#troubleshooting)
- [📂 Struktur Project](#struktur-project)
- [📊 Performance Metrics](#performance-metrics)
- [🔐 Security Notes](#security-notes)

---

## ⚡ Quick Start

### Prerequisite
- Python 3.8+
- PostgreSQL 14+
- Internet connection (untuk download models & photos)

### 1. Setup Database (5 menit)

```bash
# Login ke PostgreSQL
psql -U postgres

# Copy-paste commands berikut:
```

```sql
CREATE DATABASE face_recognition;
CREATE USER face_user WITH PASSWORD 'secure_password_123';
GRANT ALL PRIVILEGES ON DATABASE face_recognition TO face_user;

\c face_recognition postgres

CREATE TABLE IF NOT EXISTS face_embeddings (
    id SERIAL PRIMARY KEY,
    staff_id VARCHAR(50) UNIQUE NOT NULL,
    embedding BYTEA NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_staff_id ON face_embeddings(staff_id);
GRANT SELECT, INSERT, UPDATE ON face_embeddings TO face_user;
GRANT USAGE, SELECT ON SEQUENCE face_embeddings_id_seq TO face_user;

\q
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Test Database Connection

```bash
python test_db.py
```

Output yang diharapkan:
```
✅ Connection successful!
✅ Table created!
✅ Insert successful!
✅ Table has 1 records
```

### 4. Register Officers dari CSV

```bash
# Dari water_channel_officers.csv (2000+ staff)
python bulk_register.py water_channel_officers.csv
```

Output:
```
Registering officers: |████████████| 2093/2093
✅ Success: 1847
❌ Failed: 246
📊 Total: 2093
```

### 5. Jalankan API Server

```bash
# Development
python run_api.py

# Production (dengan uvicorn)
uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

Server berjalan di: `http://localhost:8000`

### 6. Test API

```bash
# Health check
curl http://localhost:8000/health

# Recognize dari image URL
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/photo.jpg",
    "water_channel_door_id": "DOOR_001"
  }'
```


### Data Flow: Recognition

```
Image URL (dari door)
    ↓
[Download image dari internet]
    ↓
[Load model InsightFace]
    ↓
[Detect faces di image]
    ↓
[Generate embedding untuk setiap face]
    ↓
[Query database: cari matching embeddings]
    ↓
[Return: [staff_id1, staff_id2, ...]]
    ↓
Door → Buka/Tutup akses
```

---

## 🔧 Setup Lokal

### Folder Structure

```
face-recog/
├── api/
│   ├── __init__.py
│   ├── app.py              # Main FastAPI application
│   ├── models.py           # Pydantic models (updated with webhook format)
│   └── webhook.py          # Webhook transformation & sending
├── core/
│   ├── __init__.py
│   ├── face_recog.py       # Face recognition logic (512D embeddings)
│   └── db_helper.py        # Database manager (PostgreSQL operations)
├── face_db/                # Local .npy backup (development)
│   ├── staff_285.npy
│   └── ...
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── test_db.py              # Database connection test
├── bulk_register.py        # Bulk registration from CSV
├── check_registration.py   # Check missing registrations
├── reset_database.py       # Reset/clear all data
├── cleanup_embeddings.py   # Fix dimension mismatches
├── test_webhook_server.py  # Mock webhook receiver
├── test_webhook_quick.py   # Webhook integration tester
├── run_api.py              # Simple API runner
├── run_api.bat             # Windows batch launcher
├── README.md               # This file
├── WEBHOOK_QUICKREF.md     # Webhook integration guide
├── DATABASE_REFERENCE.md   # Database operations reference
└── ARCHITECTURE_1000DOORS.md  # Detailed architecture explanation
```

### Install & Run

```bash
# 1. Clone/download project
cd face-recog

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup database (lihat Quick Start)

# 4. Run API
python run_api.py
```

Browser: `http://localhost:8000/docs` (Swagger UI)

---

## 🚀 Deployment Produksi

### Untuk 1000 Doors

#### Phase 1: Test (Sekarang)
```
Local Development
├─ 1 computer
├─ PostgreSQL lokal
└─ Test semua API
```

#### Phase 2: Production (Minggu Depan)
```
Server Kantor Pusat
├─ Buy/setup server dedicated
├─ Install PostgreSQL
├─ Deploy API dengan Gunicorn/Uvicorn
├─ Setup backup otomatis
└─ Configure firewall

1000 Doors
├─ Update connection string
├─ Point ke server pusat
└─ Test dari setiap door
```

### Server Requirements

**Minimum:**
- CPU: 4 core
- RAM: 8GB
- Storage: 500GB (untuk backup)
- Network: 100 Mbps

**Recommended:**
- CPU: 8 core
- RAM: 16GB
- Storage: 1TB SSD
- Network: 1 Gbps
- Backup: External drive / Cloud

### Deployment Steps

```bash
# 1. Di server produksi:
ssh admin@192.168.1.100

# 2. Install dependencies
sudo apt-get install postgresql postgresql-contrib python3-pip

# 3. Setup PostgreSQL (sama seperti lokal)
sudo -u postgres psql < setup_db.sql

# 4. Clone project
git clone <repo_url> /opt/face-recog
cd /opt/face-recog

# 5. Install Python packages
pip install -r requirements.txt

# 6. Update config (localhost → server IP)
nano config.py
# Ubah: 'host': 'localhost' → 'host': '192.168.1.100'

# 7. Setup systemd service
sudo cp face-recog.service /etc/systemd/system/
sudo systemctl enable face-recog
sudo systemctl start face-recog

# 8. Verify
curl http://192.168.1.100:8000/health
```

### Backup Strategy

```bash
# Automated daily backup (cron)
0 2 * * * /usr/local/bin/backup-face-db.sh

# Script: /usr/local/bin/backup-face-db.sh
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
pg_dump -U face_user face_recognition | \
  gzip > /backup/face_recognition_$TIMESTAMP.sql.gz
```

---

## 📡 API Reference

### 1. Health Check

**Endpoint:** `GET /health`

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "OK",
  "message": "Server is running, model loaded"
}
```

---

### 2. Register Staff (Manual)

**Endpoint:** `POST /faceregister`

Upload 4 photo + staff_id

```bash
curl -X POST http://localhost:8000/faceregister \
  -F "staff_id=EMP_001" \
  -F "photo1=@photo1.jpg" \
  -F "photo2=@photo2.jpg" \
  -F "photo3=@photo3.jpg" \
  -F "photo4=@photo4.jpg"
```

**Response:**
```json
{
  "success": true,
  "staff_id": "EMP_001",
  "message": "Registration successful"
}
```

**Error Cases:**
```json
// Foto invalid
{
  "success": false,
  "message": "Face not detected in photo1"
}

// Staff sudah ada
{
  "success": false,
  "message": "Staff already registered"
}
```

---

### 3. Recognize (Main Endpoint)

**Endpoint:** `POST /facerecognizer`

Detect wajah di image URL

```bash
curl -X POST http://localhost:8000/facerecognizer \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/camera_door_001.jpg",
    "water_channel_door_id": "DOOR_001"
  }'
```

**Response (Success):**
```json
{
  "recognized_ids": ["EMP_001", "EMP_045"],
  "water_channel_door_id": "DOOR_001",
  "message": "2 faces recognized"
}
```

**Response (Mixed):**
```json
{
  "recognized_ids": ["EMP_001", "unknown", "EMP_045"],
  "water_channel_door_id": "DOOR_001",
  "message": "2 recognized, 1 unknown"
}
```

**Response (No Faces):**
```json
{
  "recognized_ids": "0",
  "water_channel_door_id": "DOOR_001",
  "message": "No faces detected"
}
```

---

### 4. List Registered Staff

**Endpoint:** `GET /registered-staff`

```bash
curl http://localhost:8000/registered-staff
```

**Response:**
```json
{
  "count": 1847,
  "staff_ids": [
    "EMP_001",
    "EMP_002",
    ...
  ]
}
```

---

### 5. Reload Database

**Endpoint:** `POST /reload-database`

Reload embeddings dari disk (gunakan saat ada penghapusan staff)

```bash
curl -X POST http://localhost:8000/reload-database
```

**Response:**
```json
{
  "success": true,
  "message": "Database reloaded: 1847 embeddings"
}
```

---

### 6. Webhook Notification (NEW!)

**What is Webhook?**

Setelah `/facerecognizer` mengenali wajah, API secara otomatis mengirim hasil detection ke endpoint downstream Anda (Go system). Ini memungkinkan sistem Anda untuk:
- Menerima notifikasi real-time saat ada wajah terdeteksi
- Memproses akses/entry secara langsung
- Melakukan audit trail atau logging

**How it Works:**

```
Door System
    ↓
POST /facerecognizer (blocks 200-500ms)
    ↓
API Returns Response ← Webhook sent in background (async)
    ↓
Your Go System Receives Webhook (JSON over HTTP)
```

**Enable Webhook:**

```bash
export WEBHOOK_ENABLED=true
export WEBHOOK_URL=http://192.168.1.10:9000/api/detection-webhook
export WEBHOOK_TIMEOUT=10
python run_api.py
```

**Webhook Payload Format:**

Respons dari `/facerecognizer` ditransformasi otomatis ke format webhook:

```json
{
  "water_channel_door_id": 278,
  "detected_persons": [
    {
      "is_human": true,
      "is_known_person": true
    },
    {
      "is_human": true,
      "is_known_person": false
    }
  ]
}
```

**Go Struct for Integration:**

```go
type DetectedPerson struct {
    IsHuman       bool `json:"is_human"`
    IsKnownPerson bool `json:"is_known_person"`
}

type DetectionWebhookReq struct {
    WaterChannelDoorID int              `json:"water_channel_door_id"`
    DetectedPersons    []DetectedPerson `json:"detected_persons"`
}
```

**Testing Webhook:**

```bash
# Terminal 1: Start mock webhook server
python test_webhook_server.py

# Terminal 2: Enable webhook & start API
export WEBHOOK_ENABLED=true
export WEBHOOK_URL=http://localhost:9000/api/detection-webhook
python run_api.py

# Terminal 3: Test recognition
curl -X POST http://localhost:8000/facerecognizer \
  -H "Content-Type: application/json" \
  -d '{"image_url":"...", "water_channel_door_id":"278"}'
```

See [WEBHOOK_QUICKREF.md](WEBHOOK_QUICKREF.md) for complete guide.

---

## 🛠️ Management Scripts

### 1. Check Registration Status

Check staff yang belum di-register dari CSV:

```bash
python check_registration.py water_channel_officers.csv
```

**Output:**
```
================================================================================
SUMMARY
================================================================================
Total in CSV:        2093
Registered:          1847
Missing:             246
Registration Rate:   88.25%

Missing staff saved to: missing_registrations.csv
  You can register them using: python bulk_register.py missing_registrations.csv
```

### 2. Reset Database

Clear semua data untuk mulai dari awal:

```bash
python reset_database.py
```

**Warning:** Ini akan:
- Truncate table `face_embeddings` (PostgreSQL)
- Delete semua .npy files di `face_db/`
- Delete generated CSV files (failed_registrations.csv, etc.)

### 3. Cleanup Invalid Embeddings

Fix embeddings dengan dimensi salah (128D vs 512D):

```bash
# Check only (read-only)
python cleanup_embeddings.py

# Delete invalid embeddings
python cleanup_embeddings.py --cleanup
```

**Use Case:** Jika ada error "shapes (512,) and (128,) not aligned" saat recognition.

### 4. Test Webhook Integration

```bash
# Start mock webhook server
python test_webhook_server.py

# Run quick tests
python test_webhook_quick.py
```

---

## 🔔 Webhook Integration

### Overview

API automatically sends detection results to downstream endpoint (your Go system) setelah recognition selesai.

**Benefits:**
- ✅ Real-time notifications
- ✅ Async (non-blocking API response)
- ✅ Decoupled architecture
- ✅ Easy integration dengan existing system

### Configuration

**Environment Variables:**

```bash
# Enable webhook
WEBHOOK_ENABLED=true

# Downstream endpoint URL
WEBHOOK_URL=http://192.168.1.10:9000/api/detection-webhook

# Request timeout (seconds)
WEBHOOK_TIMEOUT=10
```

**Windows (PowerShell):**
```powershell
$env:WEBHOOK_ENABLED="true"
$env:WEBHOOK_URL="http://localhost:9000/api/detection-webhook"
$env:WEBHOOK_TIMEOUT="10"
```

### Payload Format

**Sent to your Go endpoint:**

```json
{
  "water_channel_door_id": 278,
  "detected_persons": [
    {
      "is_human": true,
      "is_known_person": true
    }
  ]
}
```

**Test Scenarios:**

| Scenario | API Response | Webhook Payload |
|----------|-------------|-----------------|
| Known person | `"285"` | `[{"is_human": true, "is_known_person": true}]` |
| Unknown person | `"unknown"` | `[{"is_human": true, "is_known_person": false}]` |
| Multiple faces | `["285", "unknown", "228"]` | Array with 3 persons (2 known, 1 unknown) |
| No faces | `"0"` | `[]` (empty array) |

### Implementation Guide

See complete guide: [WEBHOOK_QUICKREF.md](WEBHOOK_QUICKREF.md)

**How to Implement Go Webhook Endpoint:**

```go
type DetectedPerson struct {
    DetectedPersonName string `json:"detected_person_name"`
    IsKnownPerson      bool   `json:"is_known_person"`
}

type DetectionWebhookReq struct {
    WaterChannelDoorID int                `json:"water_channel_door_id"`
    DetectedPersons    []DetectedPerson   `json:"detected_persons"`
}

func HandleDetectionWebhook(w http.ResponseWriter, r *http.Request) {
    var webhook DetectionWebhookReq
    if err := json.NewDecoder(r.Body).Decode(&webhook); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    
    // Process detection event
    log.Printf("Door %d: %d persons detected", 
        webhook.WaterChannelDoorID, len(webhook.DetectedPersons))
    
    for _, person := range webhook.DetectedPersons {
        if person.IsKnownPerson {
            log.Printf("  - Recognized: %s", person.DetectedPersonID)
        } else {
            log.Printf("  - Unknown person")
        }
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}
```

**Testing Webhook:**

```bash
# Terminal 1: Start mock webhook server
pip install flask
python test_webhook_server.py

# Terminal 2: Enable and test API
export WEBHOOK_ENABLED=true
export WEBHOOK_URL=http://localhost:9000/api/detection-webhook
python run_api.py

# Terminal 3: Send test request
curl -X POST http://localhost:8000/facerecognizer \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/photo.jpg",
    "water_channel_door_id": "DOOR_001"
  }'

# Check webhook server terminal to see incoming webhook
```

**Important Notes:**
- ✅ Webhook is **non-blocking** (async, fire-and-forget)
- ✅ API returns **immediately** (200-500ms)
- ✅ Webhook is sent in **background** (~1-5ms overhead)
- ✅ Webhook **failures don't fail** the API response
- ✅ All webhook activity is **logged**

For more details, see [WEBHOOK_INTEGRATION.md](WEBHOOK_INTEGRATION.md)

---

## ⚙️ Konfigurasi

### config.py

```python
# Face Recognition Settings
FACE_DB_PATH = "face_db"                    # Tempat simpan .npy files
RECOGNITION_THRESHOLD = 0.45                # Similarity threshold (0-1)
BLUR_THRESHOLD = 100.0                      # Blur detection threshold
MAX_YAW = 30                                 # Max head rotation (degrees)

# API Settings
API_HOST = "0.0.0.0"                        # Bind to all interfaces
API_PORT = 8000                             # Port
API_RELOAD = True                           # Auto-reload on code change

# CORS Settings
CORS_ORIGINS = ["*"]                        # Allow all domains

# Model Settings
MODEL_NAME = "buffalo_l"                    # InsightFace model
PROVIDERS = ["CPUExecutionProvider"]        # CPU atau CUDAExecutionProvider (GPU)

# Database Settings (Production)
DB_HOST = "192.168.1.100"                   # PostgreSQL server
DB_PORT = 5432
DB_NAME = "face_recognition"
DB_USER = "face_user"
DB_PASSWORD = "secure_password_123"

# Webhook Settings (NEW!)
WEBHOOK_ENABLED = False                     # Enable webhook integration
WEBHOOK_URL = "http://localhost:9000/api/detection-webhook"  # Downstream endpoint
WEBHOOK_TIMEOUT = 10                        # Request timeout (seconds)
```

### Environment Variables

```bash
# Face Recognition
export RECOGNITION_THRESHOLD=0.45
export API_PORT=8000
export LOG_LEVEL=INFO

# Webhook (NEW!)
export WEBHOOK_ENABLED=true
export WEBHOOK_URL=http://192.168.1.10:9000/api/detection-webhook
export WEBHOOK_TIMEOUT=10
```

---

## 🛠️ Troubleshooting

### Error: "Connection to server at localhost failed"

**Penyebab:** PostgreSQL tidak running

**Solusi:**
```bash
# Check status
psql --version

# Start PostgreSQL (Windows)
net start postgresql-x64-14

# Start PostgreSQL (Linux)
sudo systemctl start postgresql
```

### Error: "FATAL: password authentication failed"

**Penyebab:** Password `postgres` salah

**Solusi:**
```bash
# Reset password
psql -U postgres -c "ALTER USER postgres WITH PASSWORD 'new_password'"
```

### Error: "No face detected in photo"

**Penyebab:** Foto tidak memiliki wajah yang jelas

**Solusi:**
- Pastikan wajah jelas, bukan blur/gelap
- Lighting minimal 300 lux
- Wajah minimal 100x100 pixel

### Error: "Transaction aborted"

**Penyebab:** Database table permissions salah

**Solusi:**
```sql
-- Re-run setup
psql -U postgres face_recognition
GRANT SELECT, INSERT, UPDATE ON face_embeddings TO face_user;
GRANT USAGE, SELECT ON SEQUENCE face_embeddings_id_seq TO face_user;
```

### Error: "OutOfMemory during model loading"

**Penyebab:** RAM tidak cukup

**Solusi:**
- Use GPU (CUDA) jika tersedia
- Upgrade RAM ke minimal 8GB
- Kurangi batch size

### Recognition Accuracy Rendah

**Solusi:**
1. Increase photo quality saat register (pencahayaan lebih baik)
2. Register 4+ photo dari berbagai angle
3. Adjust RECOGNITION_THRESHOLD di config.py:
   - Lower (0.35) = lebih sensitive, lebih false positive
   - Higher (0.55) = lebih strict, lebih false negative

---

## 📂 Struktur Code

### `api/app.py` (310 lines)

Main FastAPI application dengan 5 endpoints:

```python
# Endpoints:
GET  /health                    # Server status
POST /faceregister              # Register staff (4 photos)
POST /facerecognizer            # Recognize faces dari URL
GET  /registered-staff          # List semua staff
POST /reload-database           # Reload embeddings
```

Key features:
- Lifespan context manager untuk resource management
- CORS middleware untuk cross-origin requests
- Async/await untuk I/O operations
- Proper error handling & logging

### `api/models.py` (27 lines)

Pydantic models untuk validation:

```python
RegisterRequest          # staff_id + 4 photos
RegisterResponse         # success + message
RecognitionResponse      # recognized_ids + door_id + message
HealthResponse          # status + message
```

### `core/face_recog.py` (150+ lines)

Face recognition logic:

```python
FaceRecognition
├─ __init__()               # Load model
├─ load_database()          # Load .npy embeddings
├─ reload_database()        # Clear & reload (cache invalidation)
├─ register()               # Register dari 4 photos
├─ register_single_photo()  # Register dari 1 photo (bulk)
├─ recognize()              # Detect & match faces
└─ recognize_embedding()    # Similarity matching
```

### `config.py` (64 lines)

Centralized configuration:

```python
Config                 # Base configuration
├─ DevelopmentConfig   # DEBUG=True
└─ ProductionConfig    # DEBUG=False
```

### `test_db.py` (60 lines)

Database connection tester:

```bash
python test_db.py
```

### `bulk_register.py` (150+ lines)

Bulk registration script untuk CSV:

```bash
python bulk_register.py water_channel_officers.csv
```

Features:
- Download photo dari URL
- Generate embedding
- Save ke PostgreSQL
- Error handling & retry
- Progress bar
- Export failed list

### `run_api.py` (20 lines)

Simple API runner:

```bash
python run_api.py
```

---

## 📊 Performance Metrics

### Registration Time
- Per staff: 2-3 detik (depend internet speed)
- 2000 staff: ~1-2 jam

### Recognition Time
- Per image: 500-1000ms
- Throughput: ~1-2 faces/second

### Storage
- Per embedding: 512 bytes
- 2000 staff: ~1MB total
- Database file: ~100MB dengan index

### Network
- Image download: ~500KB typical
- Response size: <1KB
- Bandwidth per door/hari: ~50GB (1000 images/day)

---

## 🔐 Security Notes

### Database Security

✅ **Done:**
- Limited user `face_user` (tidak punya DELETE/DROP)
- Password hashed
- Network isolation (lokal saat ini)

✅ **Todo:**
- SSL/TLS encryption (untuk production)
- IP whitelist (hanya dari doors)
- Audit logging (siapa akses kapan)
- Rate limiting (prevent brute force)

### API Security

✅ **Todo:**
- API key authentication
- Request signing
- HTTPS only (production)
- DDoS protection

---

## 📖 Additional Resources

- [InsightFace Documentation](https://github.com/deepinsight/insightface)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
