# check_cache.py
import numpy as np
import os
from pathlib import Path

def check_npy_file(file_path):
    """Check the contents of a .npy file"""
    print(f"\n{'='*60}")
    print(f"Checking file: {file_path}")
    print(f"{'='*60}")
    
    try:
        # Load the data
        data = np.load(file_path, allow_pickle=True).item()
        
        print(f"Type of data: {type(data)}")
        print(f"Keys in data: {list(data.keys()) if isinstance(data, dict) else 'Not a dictionary'}")
        
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"\n--- Key: '{key}' ---")
                if isinstance(value, np.ndarray):
                    print(f"  Type: ndarray")
                    print(f"  Shape: {value.shape}")
                    print(f"  Dtype: {value.dtype}")
                    print(f"  Min: {value.min():.6f}")
                    print(f"  Max: {value.max():.6f}")
                    print(f"  Mean: {value.mean():.6f}")
                    if value.shape[0] < 10:  # Show small arrays
                        print(f"  Values: {value.flatten()[:10]}")
                else:
                    print(f"  Type: {type(value).__name__}")
                    print(f"  Value: {value}")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        print(f"\nFile size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

def main():
    cache_dir = Path("data_cache")
    
    if not cache_dir.exists():
        print(f"Cache directory {cache_dir} does not exist!")
        return
    
    # Find all .npy files
    npy_files = list(cache_dir.glob("*.npy"))
    
    if not npy_files:
        print("No .npy files found in cache directory!")
        return
    
    print(f"Found {len(npy_files)} .npy files in cache")
    
    # Check each file
    for npy_file in sorted(npy_files):
        check_npy_file(npy_file)
    
    # Also check metadata
    metadata_file = cache_dir / "file_metadata.json"
    if metadata_file.exists():
        print(f"\n{'='*60}")
        print(f"Metadata file: {metadata_file}")
        print(f"{'='*60}")
        import json
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"Number of cached files: {len(metadata)}")
            for file_path, info in metadata.items():
                filename = Path(file_path).name
                print(f"\n  {filename}:")
                print(f"    Size: {info['file_size']:,} bytes")
                print(f"    Hash: {info['file_hash']}")
                print(f"    Modified: {info['last_modified']}")
        except Exception as e:
            print(f"Error reading metadata: {e}")

if __name__ == "__main__":
    main()