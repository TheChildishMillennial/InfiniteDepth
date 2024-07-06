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
    def __init__(self, depth_model: DepthAnythingV2, image_path: str=None, raw_image:np.ndarray=None, tile_size: int=518, low_res_size:int =None):
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
        stride = self.tile_size // 2
        is_active = True
        x, y = 0, 0

        while is_active:
            x1 = int(x)
            x2 = int(x + self.tile_size)
            y1 = int(y)
            y2 = int(y + self.tile_size)

            if x2 > self.image_width:
                x1 = self.image_width - self.tile_size
                x2 = self.image_width

            if y2 > self.image_height:
                y1 = self.image_height - self.tile_size
                y2 = self.image_height

            tile_data = TileData(x1, x2, y1, y2, self.image_height, self.image_width, self.raw_image)
            self.tiles.append(tile_data)

            if x2 == self.image_width:
                x = 0
                y += stride
            else:
                x += stride

            if x2 == self.image_width and y2 == self.image_height:
                is_active = False

    def optimize_tiles(self):
        filter_image = np.zeros((self.image_height, self.image_width, 1), dtype=np.float32)
        for t in self.tiles:
            filter_image[t.y1:t.y2, t.x1:t.x2] += t.filter

        for i in range(filter_image.shape[0]):
            for j in range(filter_image.shape[1]):
                if filter_image[i, j] > 1:
                    ol_tiles:List[TileData] = []
                    total_value = 0
                    for tile in self.tiles:
                        if tile.y1 <= i < tile.y2 and tile.x1 <= j < tile.x2:
                            ol_tiles.append(tile)
                            tile_i = i - tile.y1
                            tile_j = j - tile.x1
                            total_value += tile.filter[tile_i, tile_j]
                    value_check = 0
                    for ol_idx, ol_tile in enumerate(ol_tiles):
                        tile_i = i - ol_tile.y1
                        tile_j = j - ol_tile.x1
                        ol_tile.filter[tile_i, tile_j] /= total_value
                        value_check += ol_tile.filter[tile_i, tile_j]
                        if ol_idx == (len(ol_tiles) - 1):
                            if value_check > 1:
                                ext_amt = value_check - 1
                                ol_tile.filter[tile_i, tile_j] -= ext_amt
                            if value_check < 1:
                                ext_amt = 1 - value_check
                                ol_tile.filter[tile_i, tile_j] += ext_amt

        for tile in self.tiles:
            self.filter_image[tile.y1:tile.y2, tile.x1:tile.x2] += tile.filter

    

    def generate_depth_tiles(self, depth_model: DepthAnythingV2):
        compiled_tiles = np.zeros((self.image_height, self.image_width))
        weight_sum = np.zeros((self.image_height, self.image_width))

        for tile in self.tiles:
            x1, x2 = tile.x1, tile.x2
            y1, y2 = tile.y1, tile.y2

            torch.cuda.empty_cache()  # Clear CUDA cache before inference
            tile_depth = depth_model.infer_image(tile.rgb_raw_image_tile, input_size=self.tile_size)

            scaled_depth_tile = (tile_depth - np.min(tile_depth)) / (np.max(tile_depth) - np.min(tile_depth))

            mean_low_res_scaled_depth = np.mean(self.low_resolution_scaled_depth[y1:y2, x1:x2])
            std_low_res_scaled_depth = np.std(self.low_resolution_scaled_depth[y1:y2, x1:x2])
            mean_scaled_depth = np.mean(scaled_depth_tile)
            std_scaled_depth = np.std(scaled_depth_tile)

            depth_z_scale = (scaled_depth_tile - mean_scaled_depth) / std_scaled_depth
            base_value = (mean_low_res_scaled_depth + std_low_res_scaled_depth) * tile.filter
            tile_combined = base_value.squeeze() * depth_z_scale

            y, x = np.ogrid[:y2 - y1, :x2 - x1]
            y_center, x_center = (y2 - y1) // 2, (x2 - x1) // 2
            sigma = (y2 - y1) // 2
            weight = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * sigma ** 2))
            weight = (weight - weight.min()) / (weight.max() - weight.min())

            compiled_tiles[y1:y2, x1:x2] += tile_combined * weight
            weight_sum[y1:y2, x1:x2] += weight

        valid_weights = weight_sum > 0
        compiled_tiles[valid_weights] /= weight_sum[valid_weights]
        compiled_tiles[compiled_tiles < 0] = 0

        self.compiled_depth_tiles.append(compiled_tiles)

    def combine_depth_tiles(self):
        grey_im = np.mean(self.raw_image, axis=2)
        tiles_blur = gaussian_filter(grey_im, sigma=20)
        tiles_difference = tiles_blur - grey_im
        tiles_difference = tiles_difference / np.max(tiles_difference)
        tiles_difference = gaussian_filter(tiles_difference, sigma=40)
        tiles_difference *= 5
        tiles_difference = np.clip(tiles_difference, 0, 0.999)

        combined_result = (tiles_difference * self.compiled_depth_tiles[0]) + ((1 - tiles_difference) * self.low_resolution_depth)
        self.hd_depthmap = combined_result

    def refine_depth_map_knn(self, k=5, sigma=1):
        print("Beginning KNN refinement...")
        points = np.argwhere(self.hd_depthmap > 0)
        if len(points) < k:
            print(f"Not enough points to refine depth map with k={k}.")
            return

        values = self.hd_depthmap[self.hd_depthmap > 0]
        tree = cKDTree(points)

        refined_depth_map = np.copy(self.hd_depthmap)
        for i in range(self.image_height):
            for j in range(self.image_width):
                if self.hd_depthmap[i, j] == 0:
                    distances, indices = tree.query([i, j], k=k)
                    weights:np.ndarray = np.exp(-distances ** 2 / (2 * sigma ** 2))
                    weights /= weights.sum()
                    refined_depth_map[i, j] = np.sum(weights * values[indices])

        self.hd_depthmap = refined_depth_map

    def display_image(self):
        plt.imshow(cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    def display_depth_map(self):
        plt.imshow(self.hd_depthmap, cmap='gray')
        plt.axis('off')
        plt.show()


