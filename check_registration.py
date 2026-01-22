"""
Script untuk mengecek staff yang belum diregister dari CSV
Compares CSV records dengan registered staff di database/face_db
"""

import sys
import pandas as pd
import os
from pathlib import Path
from core.db_helper import DatabaseManager
from core.face_recog import FaceRecognition

def check_missing_registrations(csv_path: str = "water_channel_officers.csv"):
    """
    Check staff yang ada di CSV tapi belum di-register
    
    Args:
        csv_path: Path to CSV file
    """
    print("=" * 80)
    print("CHECKING REGISTRATION STATUS")
    print("=" * 80)
    
    # Read CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"\n✓ CSV loaded: {csv_path}")
        print(f"  Total records in CSV: {len(df)}")
    except Exception as e:
        print(f"\n✗ Failed to load CSV: {e}")
        return
    
    # Get staff IDs from CSV (column: external_id)
    if 'external_id' not in df.columns:
        print("\n✗ Error: CSV must have 'external_id' column")
        print(f"  Available columns: {list(df.columns)}")
        return
    
    csv_staff_ids = set(df['external_id'].astype(str).tolist())
    print(f"  Unique staff IDs in CSV: {len(csv_staff_ids)}")
    
    # Get registered staff from database
    print("\n" + "-" * 80)
    print("Checking Database (PostgreSQL)...")
    print("-" * 80)
    
    try:
        db_manager = DatabaseManager()
        db_embeddings = db_manager.load_all_embeddings()
        db_staff_ids = set(db_embeddings.keys())
        print(f"✓ Registered in Database: {len(db_staff_ids)}")
    except Exception as e:
        print(f"✗ Database check failed: {e}")
        db_staff_ids = set()
    
    # Get registered staff from .npy files
    print("\n" + "-" * 80)
    print("Checking face_db/ folder (.npy files)...")
    print("-" * 80)
    
    face_db_path = Path("face_db")
    if face_db_path.exists():
        npy_files = list(face_db_path.glob("*.npy"))
        npy_staff_ids = set([f.stem for f in npy_files])
        print(f"✓ .npy files found: {len(npy_staff_ids)}")
    else:
        print("✗ face_db/ folder not found")
        npy_staff_ids = set()
    
    # Combine both sources
    registered_staff_ids = db_staff_ids | npy_staff_ids
    print(f"\n✓ Total Unique Registered: {len(registered_staff_ids)}")
    
    # Find missing
    missing_staff_ids = csv_staff_ids - registered_staff_ids
    registered_count = len(csv_staff_ids - missing_staff_ids)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total in CSV:        {len(csv_staff_ids)}")
    print(f"Registered:          {registered_count}")
    print(f"Missing:             {len(missing_staff_ids)}")
    print(f"Registration Rate:   {(registered_count/len(csv_staff_ids)*100):.2f}%")
    
    # Show missing staff
    if missing_staff_ids:
        print("\n" + "-" * 80)
        print("MISSING STAFF IDs:")
        print("-" * 80)
        
        missing_list = sorted(missing_staff_ids)
        
        # Show first 50 missing IDs
        display_count = min(50, len(missing_list))
        for i, staff_id in enumerate(missing_list[:display_count], 1):
            # Get staff info from CSV
            staff_info = df[df['external_id'].astype(str) == staff_id]
            if not staff_info.empty:
                name = staff_info.iloc[0].get('name', 'N/A')
                print(f"{i:3d}. {staff_id:15s} - {name}")
            else:
                print(f"{i:3d}. {staff_id}")
        
        if len(missing_list) > display_count:
            print(f"\n... and {len(missing_list) - display_count} more")
        
        # Save to file
        output_file = "missing_registrations.csv"
        missing_df = df[df['external_id'].astype(str).isin(missing_staff_ids)]
        missing_df.to_csv(output_file, index=False)
        print(f"\n✓ Missing staff saved to: {output_file}")
        print(f"  You can register them using: python bulk_register.py {output_file}")
    else:
        print("\n✓ All staff from CSV are registered!")
    
    # Check extra registrations (in system but not in CSV)
    extra_staff_ids = registered_staff_ids - csv_staff_ids
    if extra_staff_ids:
        print("\n" + "-" * 80)
        print(f"NOTE: {len(extra_staff_ids)} staff registered but NOT in CSV")
        print("-" * 80)
        extra_list = sorted(extra_staff_ids)[:20]
        for i, staff_id in enumerate(extra_list, 1):
            print(f"{i:3d}. {staff_id}")
        if len(extra_staff_ids) > 20:
            print(f"... and {len(extra_staff_ids) - 20} more")
    
    print("\n" + "=" * 80)


def show_registration_stats():
    """Show quick stats about registrations"""
    print("\n📊 QUICK STATS")
    print("-" * 80)
    
    # Database
    try:
        db_manager = DatabaseManager()
        db_embeddings = db_manager.load_all_embeddings()
        print(f"Database (PostgreSQL): {len(db_embeddings)} staff")
    except Exception as e:
        print(f"Database: Error - {e}")
    
    # .npy files
    face_db_path = Path("face_db")
    if face_db_path.exists():
        npy_count = len(list(face_db_path.glob("*.npy")))
        print(f".npy files:            {npy_count} files")
    else:
        print(".npy files:            0 files (folder not found)")
    
    print("-" * 80)


if __name__ == "__main__":
    # Show quick stats first
    show_registration_stats()
    
    # Run full check
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "water_channel_officers.csv"
    
    if not os.path.exists(csv_file):
        print(f"\n✗ Error: CSV file not found: {csv_file}")
        print(f"\nUsage: python check_registration.py [csv_file]")
        print(f"Example: python check_registration.py water_channel_officers.csv")
        sys.exit(1)
    
    check_missing_registrations(csv_file)
