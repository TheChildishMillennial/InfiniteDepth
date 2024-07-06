import numpy as np
import cv2
import torch
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2
from utils.image_utils import resize_and_pad_image, resize_to_original
from hd_depth_tiles.tile_data import TileData
from typing import List

class HD_DepthTiles:
    """
    HD_DepthTiles processes an input image to generate high-definition depth maps using tiles.

    Attributes:
        image_path (str): Path to the input image file.
        raw_image (np.ndarray): Raw image data if loaded directly.
        image_height (int): Height of the input image.
        image_width (int): Width of the input image.
        color_channels (int): Number of color channels in the input image (usually 3 for RGB).
        tile_size (int): Size of each square tile for depth estimation.
        low_resolution_depth (np.ndarray): Low-resolution depth map.
        low_resolution_scaled_depth (np.ndarray): Scaled low-resolution depth map.
        tiles (List[TileData]): List of TileData objects representing processed tiles.
        filter_image (np.ndarray): Filtered image for optimization.
        compiled_depth_tiles (List[np.ndarray]): List of compiled depth tiles.
        hd_depthmap (np.ndarray): Final high-definition depth map.
    """
    def __init__(self, depth_model: DepthAnythingV2, image_path: str=None, raw_image:np.ndarray=None, tile_size: int=518, low_res_size:int =None):
        """
        Initializes the HD_DepthTiles object with an input image or raw image data.

        Args:
            depth_model (DepthAnythingV2): Instance of the DepthAnythingV2 model for depth estimation.
            image_path (str, optional): Path to the input image file. Defaults to None.
            raw_image (np.ndarray, optional): Raw image data. Defaults to None.
            tile_size (int, optional): Size of each tile for depth estimation. Defaults to 518.
            low_res_size (int, optional): Preferred size for low-resolution depth estimation. Defaults to None.

        Raises:
            ValueError: If neither image_path nor raw_image is provided.
        """
        if image_path is None and raw_image is None:
            raise ValueError(f"No Image path or raw image given as input!")
        if image_path is not None:
            self.image_path = image_path        
            self.raw_image = cv2.imread(image_path)
        elif raw_image is not None:
            self.image_path = None
            self.raw_image = raw_image
        self.image_height, self.image_width, self.color_channels = self.raw_image.shape
        self.tile_size = tile_size
        self.low_resolution_depth = np.zeros((self.image_height, self.image_width, 1), dtype=np.float32)
        self.low_resolution_scaled_depth = np.zeros((self.image_height, self.image_width, 1), dtype=np.float32)
        self.generate_lo_res_depth(depth_model, low_res_size)
        self.tiles: List[TileData] = []
        self.create_tiles()
        self.filter_image = np.zeros((self.image_height, self.image_width, 1), dtype=np.float32)
        self.optimize_tiles()
        self.compiled_depth_tiles = []
        self.generate_depth_tiles(depth_model)
        self.hd_depthmap = np.zeros((self.image_height, self.image_width, 1), dtype=np.float32)
        self.combine_depth_tiles()
        self.refine_depth_map_knn(k=5, sigma=1)

    def generate_lo_res_depth(self, depth_model: DepthAnythingV2, low_res_size:int=None):
        """
        Generates a low-resolution depth map from the input image using different tile sizes.

        Args:
            depth_model (DepthAnythingV2): Instance of the DepthAnythingV2 model for depth estimation.
            low_res_size (int, optional): Preferred size for low-resolution depth estimation. 
                                        Defaults to None (tries multiple sizes from [1024, 518, 256, 128]).

        Raises:
            RuntimeError: If memory allocation fails or if an OpenCV error occurs during resizing.

        Notes:
            - Uses CUDA for GPU acceleration if available.
            - Falls back to CPU inference if CUDA memory allocation fails.

        """
        
        low_res_sizes = [1024, 518, 256, 128]
        depth_estimation = None

        for lr_idx, low_res_tile_size in enumerate(low_res_sizes):
            if low_res_size is not None:
                if low_res_tile_size != low_res_size:
                    continue
            try:
                torch.cuda.empty_cache()  # Clear CUDA cache before inference
                resized_image = resize_and_pad_image(self.raw_image, low_res_tile_size)
                print(f"Attempting depth estimation with Low Resolution Size: {low_res_tile_size}")  
                depth_estimation = depth_model.infer_image(resized_image, input_size=low_res_tile_size)
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    # If CUDA out of memory, switch to next lower resolution if available
                    if lr_idx + 1 < len(low_res_sizes):
                        print(f"CUDA out of memory. Switching to Low Resolution Size: {low_res_sizes[lr_idx + 1]}...")
                        continue
                    else:
                        break
                else:
                    # Raise any other RuntimeError encountered during inference
                    raise e

            # If depth estimation succeeds without memory errors, break the loop
            break    

        if depth_estimation is None:
            print(f"Memory Allocation Failed for CUDA, Attempting to Solve on CPU...")
            for lr_idx, low_res_tile_size in enumerate(low_res_sizes):
                try:
                    resized_image = resize_and_pad_image(self.raw_image, low_res_tile_size)
                    print(f"Attempting depth estimation with Low Resolution Size: {low_res_tile_size}")  
                    depth_model.to(torch.device('cpu'))
                    depth_estimation = depth_model.infer_image(resized_image, input_size=low_res_tile_size)
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        # If CUDA out of memory, switch to next lower resolution if available
                        if lr_idx + 1 < len(low_res_sizes):
                            print(f"CUDA out of memory. Switching to Low Resolution Size: {low_res_sizes[lr_idx + 1]}...")
                            continue
                        else:
                            break
                    else:
                        # Raise any other RuntimeError encountered during inference
                        raise e

                # If depth estimation succeeds without memory errors, break the loop
                break    

        if depth_estimation is None:
            raise RuntimeError("ERROR! Not enough memory available to estimate depth.")

        try:
            # Resize the depth map back to original dimensions
            low_res_depth = resize_to_original(depth_estimation, self.image_height, self.image_width)
            min_val = np.min(low_res_depth)
            max_val = np.max(low_res_depth)
            low_res_scaled_depth = (low_res_depth - min_val) / (max_val - min_val)

            self.low_resolution_depth = low_res_depth
            self.low_resolution_scaled_depth = low_res_scaled_depth
        except cv2.error as e:
            raise RuntimeError("OpenCV error during resizing. Failed to allocate memory.") from e


    def create_tiles(self):
        """
        Create tiles from the raw image based on the specified tile size and stride.

        Notes:
            - Tiles are created starting from the top-left corner of the image and moving horizontally, 
            then vertically.
            - Adjusts tile boundaries to fit within the image dimensions.

        """
        stride = self.tile_size // 2
        is_active = True
        x, y = 0, 0

        while is_active:
            x1 = int(x)
            x2 = int(x + self.tile_size)
            y1 = int(y)
            y2 = int(y + self.tile_size)

            # Adjust tile boundaries to fit within image dimensions
            if x2 > self.image_width:
                x1 = self.image_width - self.tile_size
                x2 = self.image_width

            if y2 > self.image_height:
                y1 = self.image_height - self.tile_size
                y2 = self.image_height

            # Create TileData object and add to tiles list
            tile_data = TileData(x1, x2, y1, y2, self.image_height, self.image_width, self.raw_image)
            self.tiles.append(tile_data)

            # Update x, y coordinates for next tile
            if x2 == self.image_width:
                x = 0
                y += stride
            else:
                x += stride

            # Check if all tiles have been created
            if x2 == self.image_width and y2 == self.image_height:
                is_active = False

    def optimize_tiles(self):
        """
        Optimize the filters of tiles to ensure that overlapping areas do not exceed a cumulative value of 1.

        Notes:
            - Creates a filter image to accumulate filters from all tiles.
            - Adjusts overlapping areas of filters to maintain a total cumulative value of 1.
            - Applies optimized filters back to each tile.

        """
        # Initialize a filter image to accumulate filters from all tiles
        filter_image = np.zeros((self.image_height, self.image_width, 1), dtype=np.float32)

        # Accumulate filters from all tiles into the filter_image
        for t in self.tiles:
            filter_image[t.y1:t.y2, t.x1:t.x2] += t.filter

        # Adjust overlapping areas in filter_image to ensure total cumulative value <= 1
        for i in range(filter_image.shape[0]):
            for j in range(filter_image.shape[1]):
                if filter_image[i, j] > 1:
                    # Gather overlapping tiles
                    overlapping_tiles = []
                    total_value = 0
                    for tile in self.tiles:
                        if tile.y1 <= i < tile.y2 and tile.x1 <= j < tile.x2:
                            overlapping_tiles.append(tile)
                            tile_i = i - tile.y1
                            tile_j = j - tile.x1
                            total_value += tile.filter[tile_i, tile_j]

                    # Normalize overlapping filters to ensure cumulative value <= 1
                    value_check = 0
                    for ol_tile in overlapping_tiles:
                        tile_i = i - ol_tile.y1
                        tile_j = j - ol_tile.x1
                        ol_tile.filter[tile_i, tile_j] /= total_value
                        value_check += ol_tile.filter[tile_i, tile_j]

                        # Adjust if total value is slightly over or under 1
                        if value_check > 1:
                            excess_amount = value_check - 1
                            ol_tile.filter[tile_i, tile_j] -= excess_amount
                        elif value_check < 1:
                            deficit_amount = 1 - value_check
                            ol_tile.filter[tile_i, tile_j] += deficit_amount

        # Apply optimized filters back to each tile
        for tile in self.tiles:
            self.filter_image[tile.y1:tile.y2, tile.x1:tile.x2] += tile.filter

    

    def generate_depth_tiles(self, depth_model: DepthAnythingV2):
        """
        Generate depth tiles for each tile in the image based on the inferred depth from a given depth model.

        Args:
            depth_model (DepthAnythingV2): The depth estimation model used to infer depth for each tile.

        Notes:
            - Calculates depth tiles by combining inferred depth values with weighted filters.
            - Normalizes depth tiles based on statistical properties of both low-resolution and inferred depths.
            - Applies Gaussian weighting to enhance central depth accuracy within each tile.

        """
        # Initialize arrays to compile depth tiles and their corresponding weights
        compiled_tiles = np.zeros((self.image_height, self.image_width))
        weight_sum = np.zeros((self.image_height, self.image_width))

        # Iterate through each tile to generate depth tiles
        for tile in self.tiles:
            x1, x2 = tile.x1, tile.x2
            y1, y2 = tile.y1, tile.y2

            # Perform depth estimation for the current tile
            torch.cuda.empty_cache()  # Clear CUDA cache before inference
            tile_depth = depth_model.infer_image(tile.rgb_raw_image_tile, input_size=self.tile_size)

            # Normalize the inferred depth within the tile
            scaled_depth_tile = (tile_depth - np.min(tile_depth)) / (np.max(tile_depth) - np.min(tile_depth))

            # Calculate statistical properties of both low-resolution and inferred depths within the tile
            mean_low_res_scaled_depth = np.mean(self.low_resolution_scaled_depth[y1:y2, x1:x2])
            std_low_res_scaled_depth = np.std(self.low_resolution_scaled_depth[y1:y2, x1:x2])
            mean_scaled_depth = np.mean(scaled_depth_tile)
            std_scaled_depth = np.std(scaled_depth_tile)

            # Scale the inferred depth using z-score normalization and combine with base value
            depth_z_scale = (scaled_depth_tile - mean_scaled_depth) / std_scaled_depth
            base_value = (mean_low_res_scaled_depth + std_low_res_scaled_depth) * tile.filter
            tile_combined = base_value.squeeze() * depth_z_scale

            # Apply Gaussian weighting to enhance central depth accuracy within the tile
            y, x = np.ogrid[:y2 - y1, :x2 - x1]
            y_center, x_center = (y2 - y1) // 2, (x2 - x1) // 2
            sigma = (y2 - y1) // 2
            weight = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * sigma ** 2))
            weight = (weight - weight.min()) / (weight.max() - weight.min())

            # Compile depth tiles and their corresponding weights
            compiled_tiles[y1:y2, x1:x2] += tile_combined * weight
            weight_sum[y1:y2, x1:x2] += weight

        # Normalize compiled depth tiles by valid weights and handle negative values
        valid_weights = weight_sum > 0
        compiled_tiles[valid_weights] /= weight_sum[valid_weights]
        compiled_tiles[compiled_tiles < 0] = 0

        # Append compiled depth tiles to the list
        self.compiled_depth_tiles.append(compiled_tiles)

    def combine_depth_tiles(self):
        """
        Combine low-resolution depth map and compiled high-resolution depth tiles using a blending technique based on image differences.

        Notes:
            - Calculates the difference between the blurred grayscale image and the original grayscale image.
            - Applies Gaussian filters to enhance and smooth the difference image.
            - Combines high-resolution depth tiles and low-resolution depth map based on the enhanced difference image.

        """
        # Compute grayscale image from raw image
        grey_im = np.mean(self.raw_image, axis=2)

        # Blur the grayscale image to create tiles blur
        tiles_blur = gaussian_filter(grey_im, sigma=20)

        # Compute difference between blurred image and grayscale image
        tiles_difference = tiles_blur - grey_im

        # Normalize the difference image
        tiles_difference = tiles_difference / np.max(tiles_difference)

        # Further enhance and smooth the difference image using Gaussian filters
        tiles_difference = gaussian_filter(tiles_difference, sigma=40)

        # Scale the difference image
        tiles_difference *= 5

        # Clip values to ensure they are within valid range
        tiles_difference = np.clip(tiles_difference, 0, 0.999)

        # Combine high-resolution depth tiles and low-resolution depth map based on the difference image
        combined_result = (tiles_difference * self.compiled_depth_tiles[0]) + ((1 - tiles_difference) * self.low_resolution_depth)

        # Assign the combined result to the HD depth map
        self.hd_depthmap = combined_result

    def refine_depth_map_knn(self, k=5, sigma=1):
        """
        Refine the HD depth map using a K-Nearest Neighbors (KNN) approach to interpolate depth values for undefined pixels.

        Args:
            k (int, optional): Number of nearest neighbors to consider. Default is 5.
            sigma (float, optional): Standard deviation for Gaussian weights. Default is 1.

        Notes:
            - Finds points in the HD depth map where depth values are zero (undefined).
            - Uses KNN algorithm with weighted averaging to estimate depth values for undefined pixels.
            - Updates the HD depth map with refined depth values.

        """
        print("Beginning KNN refinement...")

        # Get points with valid depth values in the HD depth map
        points = np.argwhere(self.hd_depthmap > 0)

        # Check if there are enough points to perform refinement
        if len(points) < k:
            print(f"Not enough points to refine depth map with k={k}.")
            return

        # Get depth values at valid points
        values = self.hd_depthmap[self.hd_depthmap > 0]

        # Build KD-tree for efficient nearest neighbor search
        tree = cKDTree(points)

        # Create a copy of the HD depth map for refining
        refined_depth_map = np.copy(self.hd_depthmap)

        # Iterate through each pixel in the image
        for i in range(self.image_height):
            for j in range(self.image_width):
                # Check if the depth value at the current pixel is zero (undefined)
                if self.hd_depthmap[i, j] == 0:
                    # Find k nearest neighbors and calculate Gaussian weights
                    distances, indices = tree.query([i, j], k=k)
                    weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
                    weights /= weights.sum()

                    # Estimate depth value using weighted averaging
                    refined_depth_map[i, j] = np.sum(weights * values[indices])

        # Update the HD depth map with refined depth values
        self.hd_depthmap = refined_depth_map

    def display_image(self):
        plt.imshow(cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    def display_depth_map(self):
        plt.imshow(self.hd_depthmap, cmap='gray')
        plt.axis('off')
        plt.show()


