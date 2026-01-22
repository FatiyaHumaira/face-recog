# Face Recognition API untuk 1000 Water Channel Doors

Sistem pengenalan wajah berbasis FastAPI + PostgreSQL untuk penggunaan sistem keamanan berbasis AI.

**⚡ Quick Commands:**
```bash
# Setup database
psql -U postgres -f setup_db.sql

# Mendafkan wajah petugas dari file CSV
python bulk_register.py water_channel_officers.csv

# Mengecek data petugas yang belum terdaftar
python check_registration.py water_channel_officers.csv

# Menjalankan API dengan webhook aktif
export WEBHOOK_ENABLED=true
export WEBHOOK_URL=http://192.168.1.10:9000/api/detection-webhook
python run_api.py

# Menjalankan webhook untuk pengujian (mock server)
python test_webhook_server.py

# Verifikasi keseluruhan setup sistem
python verify_setup.py
```

**📊 Status saat ini:**
- ✅ Database: PostgreSQL 14+ (unified storage)
- ✅ Model: InsightFace buffalo_l (512D embeddings)
- ✅ Webhook: Async integration ready

## 📋 Daftar Isi
- [⚡ Quick Start](#quick-start)
- [📦 Architecture](#architecture)
- [🔧 Setup Lokal](#setup-lokal)
- [🚀 Deployment Produksi](#deployment-produksi)
- [📡 API Reference](#api-reference)
- [🔔 Webhook Integration](#webhook-integration)
- [🛠️ Management Scripts](#management-scripts)
- [⚙️ Konfigurasi](#konfigurasi)
- [�️ Troubleshooting](#troubleshooting)
- [📂 Struktur Project](#struktur-project)

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
✅ Table has 1 records (TEST_STAFF)
```

### 4. Register petugas dari data CSV

```bash
# Dari water_channel_officers.csv 
python bulk_register.py water_channel_officers.csv
```

Contoh Output:
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
    "water_channel_door_id": "001"
  }'
```


### Data Flow: Recognition

```
Image URL (dari CC)
    ↓
[Download image]
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
Unknown terdeteksi → Alarm
```

---

## 🔧 Setup Lokal

### Folder Structure

```
face-recog/
├── api/
│   ├── __init__.py
│   ├── app.py                  # Aplikasi utama FastAPI
│   ├── models.py               # Model Pydantic (format request/response & webhook)
│   └── webhook.py              # Transformasi dan pengiriman webhook
│
├── core/
│   ├── __init__.py
│   ├── face_recog.py           # Logika pengenalan wajah (embedding 512 dimensi)
│   └── db_helper.py            # Helper database (operasi PostgreSQL)
│
├── runs/
│   └── detect/                 # Hasil deteksi (jika fitur diaktifkan)
│
├── config.py                   # Konfigurasi aplikasi
├── requirements.txt            # Daftar dependensi Python
├── .env.example                # Contoh variabel environment
│
├── bulk_register.py            # Bulk register dari file CSV
├── check_registration.py       # Pemeriksaan data pendaftaran yang belum terdaftar
├── test_db.py                  # Pengujian koneksi database
├── verify_setup.py             # Verifikasi keseluruhan setup sistem
│
├── test_api.py                 # Pengujian integrasi API
├── test_webhook_server.py      # Server webhook tiruan (mock)
├── test_webhook_integration.py # Pengujian integrasi webhook
├── test_webhook_quick.py       # Pengujian webhook cepat
│
├── run_api.py                  # Menjalankan API (mode pengembangan)
├── run_api.bat                 # Launcher API untuk Windows
│
├── water_channel_officers.csv  # Data petugas pintu air (2093 data)
└── README.md                   # Dokumentasi proyek

```

**Generated Files (during runtime):**
```
face_db/                    # .npy backup files (auto-created)
├── 285.npy
├── 228.npy
└── ...

missing_registrations.csv   # Generated by check_registration.py
failed_registrations.csv    # Generated by bulk_register.py
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


### Server Requirements

**Minimum:**
- CPU      : 4 core
- RAM      : 8 GB
- Storage  : 256 GB SSD
- Network  : 100 Mbps

**Recommended:**
- CPU      : 8 core
- RAM      : 16 GB
- Storage  : 512 GB – 1 TB SSD
- Network  : 300 Mbps – 1 Gbps
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

### Base URL
```
http://localhost:8000
```

API Documentation (Swagger): `http://localhost:8000/docs`

---

### 1. Health Check

**Endpoint:** `GET /health`

Cek status API server.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Face Recognition API is running"
}
```

---

### 2. Register Petugas (Manual - 4 Photos)

**Endpoint:** `POST /faceregister`

Register petugas dengan 4 foto dari sudut berbeda (front, left, right, top).

**Request:** multipart/form-data

```bash
curl -X POST http://localhost:8000/faceregister \
  -F "staff_id=285" \
  -F "photo_front=@front.jpg" \
  -F "photo_left=@left.jpg" \
  -F "photo_right=@right.jpg" \
  -F "photo_top=@top.jpg"
```

**Response (Success):**
```json
{
  "success": true,
  "staff_id": "285",
  "message": "Registration successful"
}
```

**Response (Error):**
```json
{
  "success": false,
  "staff_id": "285",
  "message": "Not enough valid face samples (minimum 3)"
}
```

**Notes:**
- Wajib mengunggah tepat 4 foto
- Wajah harus terdeteksi minimal di 3 foto
- Data disimpan ke database PostgreSQL dan cadangan (.npy)
- Akan mengembalikan error jika staf sudah terdaftar

---

### 3. Face Recognition (Main Endpoint)

**Endpoint:** `POST /facerecognizer`

Recognize faces dari image URL dan kirim webhook notification.
Endpoint ini digunakan untuk **mengenali wajah dari sebuah gambar** yang diambil melalui **URL kamera / CCTV**.

---

### Cara Kerja Singkat

1. Sistem **mengunduh gambar** dari URL yang diberikan  
2. Sistem **mendeteksi semua wajah** yang terdapat di dalam gambar  
3. Setiap wajah akan diperiksa apakah **dikenal (terdaftar)** atau **tidak dikenal (unknown)**  
4. Hasil pengenalan dikembalikan dalam bentuk **response JSON**  
5. Jika fitur webhook diaktifkan, sistem akan **mengirimkan data ke endpoint webhook secara otomatis**

---



**Request Body:**
```json
{
  "image_url": "http://10.44.44.2:9000/manganti-adapter/officer/photo.jpg",
  "water_channel_door_id": "278"
}
```

**Contoh:**
```bash
curl -X POST http://localhost:8000/facerecognizer \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/camera_door_278.jpg",
    "water_channel_door_id": "278"
  }'
```

**Response - Satu wajah dikenal:**
```json
{
  "recognized_ids": "285",
  "water_channel_door_id": "278",
  "message": "Recognized: 285"
}
```

**Response - Beberapa wajah (campuran):**
```json
{
  "recognized_ids": ["285", "unknown", "228"],
  "water_channel_door_id": "278",
  "message": "Found 3 face(s)"
}
```

**Response - Wajah tidak dikenal:**
```json
{
  "recognized_ids": "unknown",
  "water_channel_door_id": "278",
  "message": "Recognized: unknown"
}
```

**Response - Tidak ada wajah:**
```json
{
  "recognized_ids": "0",
  "water_channel_door_id": "278",
  "message": "No faces detected"
}
```

**Automatic Webhook:** </br>
Setelah proses pengenalan selesai, sistem akan secara otomatis mengirimkan webhook ke endpoint yang sudah dikonfigurasi (jika fitur webhook diaktifkan). 
Contoh data yang dikirimkan:

```json
{
  "water_channel_door_id": 278,
  "detected_persons": [
    {"is_human": true, "is_known_person": true},
    {"is_human": true, "is_known_person": false}
  ]
}
```

**Catatan:**
- Gambar akan diunduh langsung dari URL yang diberikan
- Sistem akan mendeteksi semua wajah dalam satu gambar
- Jika wajah lebih dari satu, hasil dikembalikan dalam bentuk array
- Webhook dikirim secara asynchronous (tidak menghambat response API)
- Estimasi waktu respon: ± 200–500 ms (tergantung ukuran gambar & jumlah wajah)

---

### 4. Daftar Petugas Terdaftar

**Endpoint:** `GET /registered-staff`

Mengambil daftar selutuh ID petugas yang sudah terdaftar di sistem. 

```bash
curl http://localhost:8000/registered-staff
```

**Response:**
```json
{
  "total": 2093,
  "staff_ids": ["285", "228", "1123", "2130", ...]
}
```

---

### 5. Reload Database

**Endpoint:** `POST /reload-database`

Memuat ulang embedding dari database (berguna setelah ada perubahan secara manual).

```bash
curl -X POST http://localhost:8000/reload-database
```

**Response:**
```json
{
  "success": true,
  "message": "Database reloaded successfully",
  "total_staff": 2093,
  "staff_ids": ["285", "228", ...]
}
```

**Use Cases:**
- Setelah menghapus data staf dari database secara manual
- Setelah melakukan impor data massal melalui SQL
- Setelah melakukan pemulihan database dari file cadangan
- Untuk mengosongkan cache di memori dan memuat ulang data terbaru

---

### 6. Webhook Notification

Setelah `/facerecognizer` mengenali wajah, API secara otomatis mengirim hasil detection ke endpoint downstream (Go system). 
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

---

## 🛠️ Management Scripts

### 1. Check Registration Status

Check petugas yang belum di-register dari CSV:

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

Petugas yang belum terdaftar disimpan ke: missing_registrations.csv
  Dapat didaftarkan dengan perintah: python bulk_register.py missing_registrations.csv
```

### 2. Reset Database

Manual reset menggunakan SQL:

```sql
-- Hubungkan ke database
psql -U face_user -d face_recognition

-- Kosongkan tabel face_embeddings
TRUNCATE TABLE face_embeddings RESTART IDENTITY CASCADE;

-- Verifikasi
SELECT COUNT(*) FROM face_embeddings;
```

**Pembersihan manual file .npy:**
```bash
# Windows
Remove-Item face_db\*.npy

# Linux/Mac
rm -rf face_db/*.npy
```

### 3. Membersihkan Embedding Tidak Valid

Digunakan untuk mengecek embedding dengan dimensi yang tidak sesuai (misalnya bukan 512D).

```python
# check_dimensions.py
from core.db_helper import DatabaseManager
import numpy as np

db = DatabaseManager()
embeddings = db.load_all_embeddings()

for staff_id, emb in embeddings.items():
    if emb.shape[0] != 512:
        print(f"Invalid: {staff_id} - {emb.shape[0]}D")
```

### 4. Pengujian integrasi webhook

```bash
# Jalankan mock webhook server
python test_webhook_server.py

# Jalankan quick tests
python test_webhook_quick.py
```

---

## Integrasi Webhook 

### Gambaran Umum

API akan secara otomatis mengirim hasil deteksi ke endpoint lanjutan (misalnya sistem Go) setelah proses pengenalan wajah selesai.

### Konfigurasi

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

**Data yang dikirim ke endpoint:**

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
---

## 📖 Additional Resources

- [InsightFace Documentation](https://github.com/deepinsight/insightface)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
