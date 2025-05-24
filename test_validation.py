from mmengine.config import Config
from mmdet3d.registry import DATASETS

def main():
    try:
        # Load config
        cfg = Config.fromfile('projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py')
        
        # Try to build validation dataset
        val_dataset = DATASETS.build(cfg.val_dataloader.dataset)
        print('SUCCESS: Validation dataset loaded successfully!')
        print(f'Dataset length: {len(val_dataset)}')
        
        # Try to get one sample
        sample = val_dataset[0]
        print('SUCCESS: Could load a sample from validation dataset')
        
        # Now check evaluator
        print(f'Evaluator config: {cfg.val_evaluator}')
    except Exception as e:
        print(f'ERROR: {e}')

if __name__ == '__main__':
    main() 