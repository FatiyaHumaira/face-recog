"""
Database helper functions untuk face recognition embeddings
Menangani semua operasi penyimpanan embedding ke PostgreSQL
"""

import psycopg2
import numpy as np
import logging
from config import Config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manager untuk semua operasi database embedding"""
    
    def __init__(self):
        """Initialize database connection"""
        self.db_config = {
            'host': Config.DB_HOST,
            'port': Config.DB_PORT,
            'database': Config.DB_NAME,
            'user': Config.DB_USER,
            'password': Config.DB_PASSWORD
        }
    
    def get_connection(self):
        """
        Get database connection
        
        Returns:
            psycopg2 connection object
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except psycopg2.OperationalError as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def save_embedding(self, staff_id: str, embedding: np.ndarray) -> bool:
        """
        Save embedding ke database (BYTEA format)
        
        Args:
            staff_id: Unique staff identifier
            embedding: numpy array (512D float32)
        
        Returns:
            bool: True if success, False if failed
        """
        try:
            # Convert embedding ke bytes
            embedding_bytes = embedding.astype(np.float32).tobytes()
            
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                # Insert atau update jika sudah ada
                cursor.execute(
                    """
                    INSERT INTO face_embeddings (staff_id, embedding)
                    VALUES (%s, %s)
                    ON CONFLICT (staff_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (staff_id, embedding_bytes)
                )
                conn.commit()
                logger.info(f"✓ Embedding saved for staff_id: {staff_id}")
                return True
                
            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"Database error for {staff_id}: {e}")
                return False
                
            finally:
                cursor.close()
                conn.close()
                
        except Exception as e:
            logger.error(f"Error saving embedding: {e}")
            return False
    
    def get_embedding(self, staff_id: str) -> np.ndarray or None:
        """
        Get embedding dari database
        
        Args:
            staff_id: Unique staff identifier
        
        Returns:
            numpy array (512D) atau None jika tidak ditemukan
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute(
                    "SELECT embedding FROM face_embeddings WHERE staff_id = %s",
                    (staff_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    embedding_bytes = result[0]
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    return embedding
                return None
                
            finally:
                cursor.close()
                conn.close()
                
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def load_all_embeddings(self) -> dict:
        """
        Load semua embeddings dari database ke memory
        
        Returns:
            dict: {staff_id: embedding_array}
        """
        embeddings = {}
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute("SELECT staff_id, embedding FROM face_embeddings")
                rows = cursor.fetchall()
                
                for staff_id, embedding_bytes in rows:
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    embeddings[staff_id] = embedding
                
                logger.info(f"✓ Loaded {len(embeddings)} embeddings from database")
                return embeddings
                
            finally:
                cursor.close()
                conn.close()
                
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return {}
    
    def delete_embedding(self, staff_id: str) -> bool:
        """
        Delete embedding dari database
        
        Args:
            staff_id: Unique staff identifier
        
        Returns:
            bool: True if success
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute(
                    "DELETE FROM face_embeddings WHERE staff_id = %s",
                    (staff_id,)
                )
                conn.commit()
                logger.info(f"✓ Embedding deleted for staff_id: {staff_id}")
                return True
                
            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"Database error: {e}")
                return False
                
            finally:
                cursor.close()
                conn.close()
                
        except Exception as e:
            logger.error(f"Error deleting embedding: {e}")
            return False
    
    def check_table_exists(self) -> bool:
        """
        Check apakah table face_embeddings sudah ada
        
        Returns:
            bool: True jika table ada
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute(
                    """
                    SELECT EXISTS(
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_name = 'face_embeddings'
                    )
                    """
                )
                result = cursor.fetchone()[0]
                return result
                
            finally:
                cursor.close()
                conn.close()
                
        except Exception as e:
            logger.error(f"Error checking table: {e}")
            return False
