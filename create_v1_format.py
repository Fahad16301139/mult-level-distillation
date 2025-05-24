"""
Convert MMDetection3D V2 format pickle files to V1 format for compatibility.
This script helps with evaluation when you have a model trained with V1 format
but your current info files are in V2 format.
"""

import pickle
import os
import numpy as np

def convert_v2_to_v1_format(v2_file, output_file):
    """Convert V2 format info file to V1 format."""
    print(f"Converting {v2_file} to {output_file}")
    
    # Load V2 format data
    with open(v2_file, 'rb') as f:
        v2_data = pickle.load(f)
    
    # Check if it's already in V1 format (a list)
    if isinstance(v2_data, list):
        print("File is already in V1 format. No conversion needed.")
        return
    
    # Extract data_list from V2 format
    if 'data_list' not in v2_data:
        print("Error: Not a valid V2 format file. Missing 'data_list' key.")
        return
    
    # Convert to V1 format (a list of info dicts)
    v1_data = []
    for item in v2_data['data_list']:
        # Convert sample_idx to a token string format
        # This is a simplification - ideally we'd use the original tokens
        if 'sample_idx' in item:
            sample_idx = item.pop('sample_idx')
            item['token'] = f"sample_token_{sample_idx:08d}"
        
        v1_data.append(item)
    
    print(f"Converted {len(v1_data)} samples")
    
    # Save V1 format data
    with open(output_file, 'wb') as f:
        pickle.dump(v1_data, f)
    
    print(f"Saved to {output_file}")

def main():
    """Main function to convert files."""
    # Convert validation info file
    val_v2_file = 'data/nuscenes/nuscenes_infos_val.pkl'
    val_v1_file = 'data/nuscenes/nuscenes_infos_val_v1format.pkl'
    convert_v2_to_v1_format(val_v2_file, val_v1_file)
    
    # Convert training info file (optional)
    train_v2_file = 'data/nuscenes/nuscenes_infos_train.pkl'
    train_v1_file = 'data/nuscenes/nuscenes_infos_train_v1format.pkl'
    convert_v2_to_v1_format(train_v2_file, train_v1_file)

if __name__ == "__main__":
    main() 