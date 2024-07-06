# InfiniteDepth: HD Depth Map Estimation with Metric Accuracy

## Overview
InfiniteDepth is a powerful tool for generating high-definition depth maps from images of any size. Utilizing the DepthAnythingV2 model, it ensures accurate metric depth estimations. This tool is designed to work seamlessly with large images, breaking them down into manageable tiles, then combining, filtering and optimizing the results into a comprehensive high definition depth map.

## Example
[Test Image](/demo/demo_files/test_img.jpg)
[Data Image](/demo/demo_files/Figure_1.png)
[Data Image Output](/demo/demo_files/data_test_img.jpg)
[Normalized Image](/demo/demo_files/img_test_img.jpg)

## Features
- **No Image Size Limits**: Process images of any size by tiling and seamlessly combining results.
- **HD Depth Maps**: Generate high-definition depth maps for detailed analysis.
- **Metric Accuracy**: Ensures accurate depth measurements in metric units.

## Installation
1. **Clone the repository:**
   ```
   git clone https://github.com/TheChildishMillennial/InfiniteDepth.git
   cd InfiniteDepth
   ```

2. **Optional: Install PyTorch with CUDA for GPU acceleration**
    If you want to run the project on a GPU with CUDA, you need to install the appropriate PyTorch version for your CUDA setup. Follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) to install the correct version.

    Example for CUDA 11.8:
    ```
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ```

2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

3. **Download the model checkpoints:**
    The necessary model checkpoints will be automatically downloaded from HuggingFace if not already present.

## Usage
1. **Command Line Interface (CLI):**
    ```
    python hd_depth/hd_depthmap.py -i /path/to/input/image.jpg -o /path/to/output/directory -ds indoor -d auto -v -do -no
    ```

### Arguments
- **`-i`, `--input`**: Path to the input directory or media file. Default is `demo/input`
- **`-o`, `--output`**: Path to the output directory. Default is `demo/output`
- **`-ds`, `--dataset`**: Dataset type, choose `indoor` or `outdoor`. Default is `indoor`.
- **`-ts`, `--tile_size`**: Preferred size of depth tile. Max size is 1024. Default size is `518`.
- **`-lr`, `--low_res_size`**: Preferred Low Resolution Size. Can be `auto`, `1024`, `518`, `256` or `128`. Default is `auto`
- **`-d`, `--device`**: Preferred device to load the model, for example `cpu` or `cuda`. Use `auto` for automatic device selection. Default is `auto`.
- **`-v`, `--visualize`**: Visualize the depth estimation. Use this flag to enable.
- **`-do`, `--data_output`**: Save metric depth data to a float32 file. Use this flag to enable.
- **`-no`, `--normalized_output`**: Convert output channel data to uint8 (0-255). Use this flag to enable.

## Example
```
python main.py -i example.jpg -o results/ -ds indoor -d auto -v -do -no
```

## Important
    Ensure that your system has enough free memory to avoid memory allocation errors.

## Note
- dataset - `indoor` has max depth detection of 20 meters
- dataset - `outdoor` has max depth detection of 80 meters

## Credits
- **DepthAnythingV2**: Used for depth estimation. [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2)
- **TilingZoeDepth**: Adapted from Tiled Zoe Depth V3 [TilingZoeDepth](https://github.com/BillFSmith/TilingZoeDepth)

## License
This project is licensed under the MIT License. See the [LICENSE](.LICENSE) file for details."# InfiniteDepth" 
