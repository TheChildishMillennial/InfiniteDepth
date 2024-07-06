import numpy as np
import math

class TileData:
    def __init__(self, x1: int, x2: int, y1: int, y2: int, original_image_height: int, original_image_width: int, raw_image: np.ndarray):
        self.x1: int = x1
        self.x2: int = x2
        self.y1: int = y1
        self.y2: int = y2        
        self.original_height: int = original_image_height
        self.original_width: int = original_image_width
        self.rgb_raw_image_tile: np.ndarray = raw_image[y1:y2, x1:x2]
        
        if (x2 - x1) != (y2 - y1):
            raise ValueError(f"Tile is not square, Shape: X - {x2 - x1}, Y - {y2 - y1}")
        
        self.tile_size: int = x2 - x1
        self.filter: np.ndarray = self.generate_filter()

    def generate_filter(self) -> np.ndarray:
        filter_array = np.zeros((self.tile_size, self.tile_size, 1), dtype=np.float32)

        for i in range(self.tile_size):
            for j in range(self.tile_size):
                x_value, y_value = 0, 0

                if self.y1 == 0:
                    if self.x1 == 0:
                        x_value, y_value = self.top_left_filter(i, j)
                    elif self.x1 > 0 and self.x2 < self.original_width:
                        x_value, y_value = self.top_filter(i, j)
                    elif self.x2 == self.original_width:
                        x_value, y_value = self.top_right_filter(i, j)
                elif self.y2 == self.original_height:
                    if self.x1 == 0:
                        x_value, y_value = self.bottom_left_filter(i, j)
                    elif self.x1 > 0 and self.x2 < self.original_width:
                        x_value, y_value = self.bottom_filter(i, j)
                    elif self.x2 == self.original_width:
                        x_value, y_value = self.bottom_right_filter(i, j)
                elif self.x1 == 0 and self.y1 > 0 and self.y2 < self.original_height:
                    x_value, y_value = self.left_filter(i, j)
                elif self.x2 == self.original_width and self.y1 > 0 and self.y2 < self.original_height:
                    x_value, y_value = self.right_filter(i, j)
                elif self.x1 > 0 and self.x2 < self.original_width and self.y1 > 0 and self.y2 < self.original_height:
                    x_value, y_value = self.center_filter(i, j)

                max_val = max(x_value, y_value)
                filter_array[i, j] = (x_value * y_value) / max_val if x_value * y_value > 1 else x_value * y_value

        return filter_array

    def top_right_filter(self, i: int, j: int) -> tuple:
        c = math.sqrt(((self.tile_size - j) ** 2) + (i ** 2))
        thresh_distance = self.tile_size * 0.5
        if c <= thresh_distance:
            return 1, 1
        x_value = j / (self.tile_size * 0.5)
        y_value = (self.tile_size - i) / (self.tile_size * 0.5)
        return max(0, min(1, x_value)), max(0, min(1, y_value))

    def top_left_filter(self, i: int, j: int) -> tuple:
        c = math.sqrt((j ** 2) + (i ** 2))
        thresh_distance = self.tile_size * 0.5
        if c <= thresh_distance:
            return 1, 1
        x_value = (self.tile_size - j) / (self.tile_size * 0.5)
        y_value = (self.tile_size - i) / (self.tile_size * 0.5)
        return max(0, min(1, x_value)), max(0, min(1, y_value))

    def top_filter(self, i: int, j: int) -> tuple:
        half_tile = self.tile_size // 2
        x_value = j / half_tile if j < half_tile else (half_tile - (j - half_tile)) / half_tile
        y_value = 1 if i <= half_tile else (half_tile - (i - half_tile)) / half_tile
        return x_value, y_value

    def bottom_filter(self, i: int, j: int) -> tuple:
        half_tile = self.tile_size // 2
        x_value = j / half_tile if j < half_tile else (half_tile - (j - half_tile)) / half_tile
        y_value = 1 if i >= half_tile else (half_tile - (half_tile - i)) / half_tile
        return x_value, y_value

    def left_filter(self, i: int, j: int) -> tuple:
        half_tile = self.tile_size // 2
        y_value = i / half_tile if i < half_tile else (half_tile - (i - half_tile)) / half_tile
        x_value = 1 if j <= half_tile else (half_tile - (j - half_tile)) / half_tile
        return x_value, y_value

    def right_filter(self, i: int, j: int) -> tuple:
        half_tile = self.tile_size // 2
        y_value = i / half_tile if i < half_tile else (half_tile - (i - half_tile)) / half_tile
        x_value = 1 if j >= half_tile else (half_tile - (half_tile - j)) / half_tile
        return x_value, y_value

    def bottom_left_filter(self, i: int, j: int) -> tuple:
        c = math.sqrt((j ** 2) + ((self.tile_size - i) ** 2))
        thresh_distance = self.tile_size * 0.5
        if c <= thresh_distance:
            return 1, 1
        x_value = (self.tile_size - j) / (self.tile_size * 0.5)
        y_value = i / (self.tile_size * 0.5)
        return max(0, min(1, x_value)), max(0, min(1, y_value))

    def bottom_right_filter(self, i: int, j: int) -> tuple:
        c = math.sqrt(((self.tile_size - j) ** 2) + ((self.tile_size - i) ** 2))
        thresh_distance = self.tile_size * 0.5
        if c <= thresh_distance:
            return 1, 1
        x_value = j / (self.tile_size * 0.5)
        y_value = i / (self.tile_size * 0.5)
        return max(0, min(1, x_value)), max(0, min(1, y_value))
    
    def center_filter(self, i: int, j: int) -> tuple:
        center_x, center_y = self.tile_size / 2, self.tile_size / 2
        radius = self.tile_size / 2
        inner_radius = radius * 0.25
        distance_to_center = math.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
        if distance_to_center <= inner_radius:
            return 1, 1
        if distance_to_center <= radius:
            normalized_distance = (distance_to_center - inner_radius) / (radius - inner_radius)
            return 1 - normalized_distance, 1 - normalized_distance
        return 0, 0