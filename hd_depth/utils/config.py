import os
import sys

# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)

dataset_configs = {
    'indoor': {
        'encoder': 'vitl', 
        'dataset': 'hyperism', 
        'max_depth': 20,
        'repo_id': 'depth-anything/Depth-Anything-V2-Metric-Hypersim-Large',
        'checkpoint_file': 'depth_anything_v2_metric_hypersim_vitl.pth'
    },
    'outdoor': {
        'encoder': 'vitl', 
        'dataset': 'vkitti', 
        'max_depth': 80,
        'repo_id': 'depth-anything/Depth-Anything-V2-Metric-VKITTI-Large',
        'checkpoint_file': 'depth_anything_v2_metric_vkitti_vitl.pth'
    }
}

CHECKPOINT_DIR = os.path.join(parent_dir, "checkpoints")
DEFAULT_IN_DIR = os.path.join(root_dir, "demo", "input")
DEFAULT_OUT_DIR = os.path.join(root_dir, "demo", "output")