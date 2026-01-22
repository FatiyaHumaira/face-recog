#!/usr/bin/env python3
"""Test PostgreSQL Connection"""

import psycopg2
from psycopg2.extras import execute_values

# Database credentials (sesuaikan dengan setup Anda)
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'face_recognition',
    'user': 'face_user',
    'password': 'secure_password_123'
}

def test_connection():
    """Test koneksi ke PostgreSQL"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        print("✅ Connection successful!")
        
        # Create embeddings table (jika belum ada)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id SERIAL PRIMARY KEY,
                staff_id VARCHAR(50) UNIQUE NOT NULL,
                embedding BYTEA NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        print("✅ Table created!")
        
        # Test insert
        test_data = b'\x00' * 512  # Dummy embedding (512 bytes)
        cursor.execute(
            "INSERT INTO face_embeddings (staff_id, embedding) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            ('TEST_STAFF', test_data)
        )
        conn.commit()
        print("✅ Insert successful!")
        
        # Test select
        cursor.execute("SELECT COUNT(*) FROM face_embeddings")
        count = cursor.fetchone()[0]
        print(f"✅ Table has {count} records")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing PostgreSQL connection...")
    test_connection()
