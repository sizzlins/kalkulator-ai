
import csv
import os
import logging
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_csv_data(filepath: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Load data from a CSV file.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        Dictionary mapping column names to numpy arrays, or None if failed.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None
        
    try:
        # Try pandas first for speed/robustness if available
        import pandas as pd
        try:
            df = pd.read_csv(filepath)
            data_map = {}
            for col in df.columns:
                # Clean column name (strip whitespace)
                clean_col = col.strip()
                # coercion to numeric, errors='coerce' turns non-numeric to NaN
                # We do manual check
                vals = pd.to_numeric(df[col], errors='coerce').to_numpy()
                data_map[clean_col] = vals
            print(f"Loaded {len(df)} rows from '{filepath}' (using pandas)")
            return data_map
        except ImportError:
            pass # Fallback to standard csv
            
    except Exception:
        # Pandas not available or failed, silent fallback to standard CSV
        pass
        
    # Standard CSV fallback
    try:
        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                print("Error: CSV file is empty or missing header")
                return None
                
            cols = {name.strip(): [] for name in reader.fieldnames}
            
            row_count = 0
            for row in reader:
                row_count += 1
                for k, v in row.items():
                    k_clean = k.strip() if k else None
                    if k_clean and k_clean in cols:
                        try:
                            cols[k_clean].append(float(v))
                        except (ValueError, TypeError):
                            cols[k_clean].append(np.nan)
            
            # Convert to numpy
            result = {}
            for k, v in cols.items():
                result[k] = np.array(v)
                
            print(f"Loaded {row_count} rows from '{filepath}' (standard csv)")
            return result
            
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
