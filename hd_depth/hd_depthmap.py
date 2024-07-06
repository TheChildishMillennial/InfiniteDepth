import os
import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2
from hd_depth_tiles.hd_depth_tiles import HD_DepthTiles
from utils.system_utils import get_device
from utils.checkpoint_utils import download_pth_from_huggingface
from utils.config import dataset_configs, CHECKPOINT_DIR, DEFAULT_IN_DIR, DEFAULT_OUT_DIR

def load_depth_model(dataset: str = 'indoor', device: torch.device = 'cpu') -> DepthAnythingV2:
    """
    Loads the depth model for a specified dataset and device.

    Args:
        dataset (str): The dataset to use ('indoor' or 'outdoor'). Defaults to 'indoor'.
        device (torch.device): The device to load the model on. Defaults to 'cpu'.

    Returns:
        DepthAnythingV2: The loaded depth model.
    """
    config = dataset_configs[dataset]

    model = DepthAnythingV2(
        encoder=config['encoder'],
        max_depth=config['max_depth']
    ).to(device=device)
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, config['checkpoint_file'])
    if not os.path.exists(checkpoint_path):
        download_pth_from_huggingface(repo_id=config['repo_id'], filename=config['checkpoint_file'], local_path=checkpoint_path)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    return model

def main(args):
    """
    Main function to run the depth estimation.

    Args:
        args: Command line arguments.
    """
    input_path = args.input
    output_path = args.output
    dataset = args.dataset
    tile_size = args.tile_size

    if args.low_res_size == 'auto':
        low_res_size = None
    else:
        low_res_size = args.low_res_size

    if input_path == DEFAULT_IN_DIR:
        for file in os.listdir(input_path):
            if os.path.isfile(os.path.join(input_path, file)):
                input_path = os.path.join(input_path, file)

    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)

    torch.cuda.empty_cache()
    print(f"Loading depth model for dataset: {dataset} on device: {device}")
    depth_model = load_depth_model(dataset=dataset, device=device)

    print(f"Reading input image from path: {input_path}")
    depth_tiles = HD_DepthTiles(depth_model=depth_model, image_path=input_path, tile_size=tile_size, low_res_size=low_res_size)
    hd_depthmap = depth_tiles.hd_depthmap

    if args.visualize:
        plt.imshow(hd_depthmap, cmap="magma")
        plt.colorbar()
        plt.show()

    if args.data_output:
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"data_{os.path.basename(input_path)}")
        cv2.imwrite(output_file, hd_depthmap.astype(np.float32))
        print(f"Saved depth estimation to {output_file}")

    if args.normalize_output:
        output_file = os.path.join(output_path, f"img_{os.path.basename(input_path)}")
        normalized_hd_depthmap:np.ndarray = (hd_depthmap / np.max(hd_depthmap)) * 255
        cv2.imwrite(output_file, normalized_hd_depthmap.astype(np.uint8))
        print(f"Saved depth estimation to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=False, default=DEFAULT_IN_DIR, help="Path to input directory or media file")
    parser.add_argument("-o", "--output", required=False, default=DEFAULT_OUT_DIR, help="Path to output directory")
    parser.add_argument("-ds", "--dataset", required=False, default='indoor', help="Choose 'indoor' or 'outdoor'")
    parser.add_argument("-ts", "--tile_size", required=False, default=518, help="The size of the depth tiles. Max Size is 1024, default size is 518")
    parser.add_argument("-lr", "--low_res_size", required=False, default='auto', help="Preferred Low Resolution Size. Can be 'auto', 1024, 518, 256 or 128. Default is 'auto'")
    parser.add_argument("-d", "--device", required=False, default='auto', help="Preferred Device to Load Model")
    parser.add_argument("-v", "--visualize", required=False, action='store_true', help="Visualize Depth Estimation")
    parser.add_argument("-do", "--data_output", required=False, action='store_true', help="Save Metric Depth Data to float32 1 = 1m")
    parser.add_argument("-no", "--normalize_output", required=False, action='store_true', help="Convert Output Channel Data to uint8 0-255")
    args = parser.parse_args()
    main(args)
