#!/usr/bin/env python3
"""Bulk Register Officers dari CSV dengan Photo URL"""

import csv
import requests
import numpy as np
from pathlib import Path
import psycopg2
from psycopg2 import sql
from core.face_recog import FaceRecognition
import logging
from tqdm import tqdm
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Database config
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'face_recognition',
    'user': 'face_user',
    'password': 'secure_password_123'
}

class BulkRegistration:
    def __init__(self):
        self.face_recog = FaceRecognition()
        self.face_recog.load_database()
        
        # Test connection
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            self.cursor = self.conn.cursor()
            logger.info("Database connected")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
        
    def register_from_csv(self, csv_path, skip_existing=True):
        """Register officers dari CSV"""
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        logger.info(f"Total officers di CSV: {len(rows)}")
        
        success_count = 0
        failed_count = 0
        failed_list = []
        
        for row in tqdm(rows, desc="Registering officers"):
            external_id = None
            try:
                # Safe parsing dengan null check
                external_id = row.get('external_id', '').strip() if row.get('external_id') else None
                photo_url = row.get('photo', '').strip() if row.get('photo') else None
                name = row.get('name', '').strip() if row.get('name') else None
                
                if not external_id or not photo_url:
                    failed_count += 1
                    failed_list.append((external_id or "EMPTY", "Missing ID or photo URL"))
                    continue
                
                # Cek jika sudah ada
                if skip_existing:
                    try:
                        self.cursor.execute(
                            "SELECT id FROM face_embeddings WHERE staff_id = %s",
                            (external_id,)
                        )
                        if self.cursor.fetchone():
                            logger.debug(f"✓ {external_id} sudah ada, skip")
                            continue
                    except Exception as e:
                        logger.warning(f"Check existence error untuk {external_id}: {e}")
                        # Reset connection
                        self.conn = psycopg2.connect(**DB_CONFIG)
                        self.cursor = self.conn.cursor()
                        continue
                
                # Download foto
                try:
                    response = requests.get(photo_url, timeout=10)
                    if response.status_code != 200:
                        failed_count += 1
                        failed_list.append((external_id, f"HTTP {response.status_code}"))
                        continue
                    
                    image_bytes = response.content
                except Exception as e:
                    failed_count += 1
                    failed_list.append((external_id, f"Download error: {str(e)[:50]}"))
                    continue
                
                # Generate embedding
                embedding = None
                try:
                    embedding = self.face_recog.register_single_photo(
                        image_bytes, 
                        external_id
                    )
                    
                    if embedding is None:
                        failed_count += 1
                        failed_list.append((external_id, "No face detected"))
                        continue
                    
                except Exception as e:
                    failed_count += 1
                    failed_list.append((external_id, f"Embedding error: {str(e)[:50]}"))
                    continue
                
                # Save to database - TRANSACTION PER ROW
                try:
                    embedding_bytes = embedding.astype(np.float32).tobytes()
                    
                    # Clear any previous errors
                    self.conn.rollback()
                    
                    self.cursor.execute(
                        """
                        INSERT INTO face_embeddings (staff_id, embedding)
                        VALUES (%s, %s)
                        ON CONFLICT (staff_id) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (external_id, embedding_bytes)
                    )
                    self.conn.commit()
                    success_count += 1
                    logger.info(f"✓ {external_id} ({name}) registered")
                    
                except psycopg2.Error as e:
                    # Rollback transaksi yang error
                    try:
                        self.conn.rollback()
                    except:
                        pass
                    
                    failed_count += 1
                    failed_list.append((external_id, f"DB error: {str(e)[:50]}"))
                    logger.error(f"DB error untuk {external_id}: {e}")
                    
                    # Reconnect jika transaction aborted
                    if "aborted" in str(e).lower():
                        try:
                            self.conn.close()
                        except:
                            pass
                        self.conn = psycopg2.connect(**DB_CONFIG)
                        self.cursor = self.conn.cursor()
                    
                    # Reconnect
                    try:
                        self.conn = psycopg2.connect(**DB_CONFIG)
                        self.cursor = self.conn.cursor()
                    except:
                        logger.error("Failed to reconnect to database")
                        break
                
                # Rate limiting
                time.sleep(0.05)
                
            except Exception as e:
                if external_id:
                    failed_count += 1
                    failed_list.append((external_id, f"Unexpected: {str(e)[:50]}"))
                logger.error(f"Unexpected error for {external_id}: {e}")
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Success: {success_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Total: {success_count + failed_count}")
        
        if failed_list:
            logger.info(f"\n First 20 Failed registrations:")
            for staff_id, reason in failed_list[:20]:
                logger.info(f"  - {staff_id}: {reason}")
            
            if len(failed_list) > 20:
                logger.info(f"  ... and {len(failed_list) - 20} more")
        
        # Save failed list ke file
        if failed_list:
            with open('failed_registrations.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['staff_id', 'reason'])
                writer.writerows(failed_list)
            logger.info(f"\n📋 Failed list saved to: failed_registrations.csv")
        
        self.close()
        return {
            'success': success_count,
            'failed': failed_count,
            'total': success_count + failed_count
        }
    
    def close(self):
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
        except:
            pass


if __name__ == "__main__":
    import sys
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "water_channel_officers.csv"
    
    logger.info(f"Loading CSV: {csv_path}")
    
    bulk = BulkRegistration()
    result = bulk.register_from_csv(csv_path)
