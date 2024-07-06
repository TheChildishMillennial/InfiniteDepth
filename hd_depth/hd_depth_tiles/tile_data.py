import numpy as np
import math

class TileData:
    def __init__(self, x1: int, x2: int, y1: int, y2: int, original_image_height: int, original_image_width: int, raw_image: np.ndarray):
        """
        Initialize a TileData object representing a tile extracted from a larger image.

        Args:
            x1 (int): Starting x-coordinate of the tile.
            x2 (int): Ending x-coordinate of the tile.
            y1 (int): Starting y-coordinate of the tile.
            y2 (int): Ending y-coordinate of the tile.
            original_image_height (int): Height of the original image.
            original_image_width (int): Width of the original image.
            raw_image (np.ndarray): Raw image data from which the tile is extracted.

        Raises:
            ValueError: If the extracted tile is not square.

        Attributes:
            x1 (int): Starting x-coordinate of the tile.
            x2 (int): Ending x-coordinate of the tile.
            y1 (int): Starting y-coordinate of the tile.
            y2 (int): Ending y-coordinate of the tile.
            original_height (int): Height of the original image.
            original_width (int): Width of the original image.
            rgb_raw_image_tile (np.ndarray): Extracted RGB tile from the raw image.
            tile_size (int): Size (width or height) of the tile (assumes it's square).
            filter (np.ndarray): Generated filter data for the tile.

        """
        self.x1: int = x1
        self.x2: int = x2
        self.y1: int = y1
        self.y2: int = y2        
        self.original_height: int = original_image_height
        self.original_width: int = original_image_width
        self.rgb_raw_image_tile: np.ndarray = raw_image[y1:y2, x1:x2]
        
        # Check if the tile is square
        if (x2 - x1) != (y2 - y1):
            raise ValueError(f"Tile is not square, Shape: X - {x2 - x1}, Y - {y2 - y1}")
        
        self.tile_size: int = x2 - x1
        self.filter: np.ndarray = self.generate_filter()

    def generate_filter(self) -> np.ndarray:
        """
        Generate a filter array specific to the tile based on its position in the original image.

        Returns:
            np.ndarray: Filter array of shape (tile_size, tile_size, 1) containing filter values.

        Notes:
            - This method constructs a filter based on the position of the tile within the original image.
            - It applies different filter computations depending on whether the tile is at the top-left,
            top-right, bottom-left, bottom-right, top, bottom, left, right, or center of the original image.
            - The generated filter values are normalized between 0 and 1.

        """
        filter_array = np.zeros((self.tile_size, self.tile_size, 1), dtype=np.float32)

        for i in range(self.tile_size):
            for j in range(self.tile_size):
                x_value, y_value = 0, 0

                # Determine which filter function to use based on tile position
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

                # Calculate maximum value and apply normalization
                max_val = max(x_value, y_value)
                filter_array[i, j] = (x_value * y_value) / max_val if x_value * y_value > 1 else x_value * y_value

        return filter_array

    def top_right_filter(self, i: int, j: int) -> tuple:
        """
        Generate filter values for the top-right corner of the tile.
        
        Args:
            i (int): Row index within the tile.
            j (int): Column index within the tile.
        
        Returns:
            tuple: Tuple containing x and y filter values.
        
        Notes:
            - If the distance from the tile's top-right corner to the center is within half the tile size,
            return (1, 1).
            - Otherwise, compute normalized x and y values based on distances from the top-right corner
            to the center normalized by half the tile size.
        """
        c = math.sqrt(((self.tile_size - j) ** 2) + (i ** 2))
        thresh_distance = self.tile_size * 0.5
        if c <= thresh_distance:
            return 1, 1
        x_value = j / (self.tile_size * 0.5)
        y_value = (self.tile_size - i) / (self.tile_size * 0.5)
        return max(0, min(1, x_value)), max(0, min(1, y_value))

    def top_left_filter(self, i: int, j: int) -> tuple:
        """
        Generate filter values for the top-left corner of the tile.
        
        Args:
            i (int): Row index within the tile.
            j (int): Column index within the tile.
        
        Returns:
            tuple: Tuple containing x and y filter values.
        
        Notes:
            - If the distance from the tile's top-left corner to the center is within half the tile size,
            return (1, 1).
            - Otherwise, compute normalized x and y values based on distances from the top-left corner
            to the center normalized by half the tile size.
        """
        c = math.sqrt((j ** 2) + (i ** 2))
        thresh_distance = self.tile_size * 0.5
        if c <= thresh_distance:
            return 1, 1
        x_value = (self.tile_size - j) / (self.tile_size * 0.5)
        y_value = (self.tile_size - i) / (self.tile_size * 0.5)
        return max(0, min(1, x_value)), max(0, min(1, y_value))

    def top_filter(self, i: int, j: int) -> tuple:
        """
        Generate filter values for the top edge of the tile.
        
        Args:
            i (int): Row index within the tile.
            j (int): Column index within the tile.
        
        Returns:
            tuple: Tuple containing x and y filter values.
        
        Notes:
            - Divide the column index `j` by half the tile size (`half_tile`).
            - If `j` is less than `half_tile`, use the calculated value directly.
            - If `j` is greater than or equal to `half_tile`, calculate the distance from the center
            and normalize it by `half_tile`.
            - For the row index `i`, set `y_value` to 1 if `i` is less than or equal to `half_tile`,
            otherwise calculate the normalized distance from `half_tile`.
        """
        half_tile = self.tile_size // 2
        x_value = j / half_tile if j < half_tile else (half_tile - (j - half_tile)) / half_tile
        y_value = 1 if i <= half_tile else (half_tile - (i - half_tile)) / half_tile
        return x_value, y_value

    def bottom_filter(self, i: int, j: int) -> tuple:
        """
        Generate filter values for the bottom edge of the tile.
        
        Args:
            i (int): Row index within the tile.
            j (int): Column index within the tile.
        
        Returns:
            tuple: Tuple containing x and y filter values.
        
        Notes:
            - Divide the column index `j` by half the tile size (`half_tile`).
            - If `j` is less than `half_tile`, use the calculated value directly.
            - If `j` is greater than or equal to `half_tile`, calculate the distance from the center
            and normalize it by `half_tile`.
            - For the row index `i`, set `y_value` to 1 if `i` is greater than or equal to `half_tile`,
            otherwise calculate the normalized distance from `half_tile`.
        """
        half_tile = self.tile_size // 2
        x_value = j / half_tile if j < half_tile else (half_tile - (j - half_tile)) / half_tile
        y_value = 1 if i >= half_tile else (half_tile - (half_tile - i)) / half_tile
        return x_value, y_value

    def left_filter(self, i: int, j: int) -> tuple:
        """
        Generate filter values for the left edge of the tile.
        
        Args:
            i (int): Row index within the tile.
            j (int): Column index within the tile.
        
        Returns:
            tuple: Tuple containing x and y filter values.
        
        Notes:
            - Divide the row index `i` by half the tile size (`half_tile`).
            - If `i` is less than `half_tile`, use the calculated value directly.
            - If `i` is greater than or equal to `half_tile`, calculate the distance from the center
            and normalize it by `half_tile`.
            - For the column index `j`, set `x_value` to 1 if `j` is less than or equal to `half_tile`,
            otherwise calculate the normalized distance from `half_tile`.
        """
        half_tile = self.tile_size // 2
        y_value = i / half_tile if i < half_tile else (half_tile - (i - half_tile)) / half_tile
        x_value = 1 if j <= half_tile else (half_tile - (j - half_tile)) / half_tile
        return x_value, y_value

    def right_filter(self, i: int, j: int) -> tuple:
        """
        Generate filter values for the right edge of the tile.
        
        Args:
            i (int): Row index within the tile.
            j (int): Column index within the tile.
        
        Returns:
            tuple: Tuple containing x and y filter values.
        
        Notes:
            - Divide the row index `i` by half the tile size (`half_tile`).
            - If `i` is less than `half_tile`, use the calculated value directly.
            - If `i` is greater than or equal to `half_tile`, calculate the distance from the center
            and normalize it by `half_tile`.
            - For the column index `j`, set `x_value` to 1 if `j` is greater than or equal to `half_tile`,
            otherwise calculate the normalized distance from `half_tile`.
        """
        half_tile = self.tile_size // 2
        y_value = i / half_tile if i < half_tile else (half_tile - (i - half_tile)) / half_tile
        x_value = 1 if j >= half_tile else (half_tile - (half_tile - j)) / half_tile
        return x_value, y_value

    def bottom_left_filter(self, i: int, j: int) -> tuple:
        """
        Generate filter values for the bottom-left corner of the tile.
        
        Args:
            i (int): Row index within the tile.
            j (int): Column index within the tile.
        
        Returns:
            tuple: Tuple containing x and y filter values.
        
        Notes:
            - Calculate the distance `c` from the bottom-left corner using the Pythagorean theorem.
            - If `c` is less than or equal to half the tile size (`thresh_distance`), return (1, 1) indicating full filter intensity.
            - Otherwise, normalize `x_value` and `y_value` based on their distance from the corners of the tile, scaled by half the tile size.
            - Ensure that the returned values are clipped between 0 and 1 to maintain valid filter values.
        """
        c = math.sqrt((j ** 2) + ((self.tile_size - i) ** 2))
        thresh_distance = self.tile_size * 0.5
        if c <= thresh_distance:
            return 1, 1
        x_value = (self.tile_size - j) / (self.tile_size * 0.5)
        y_value = i / (self.tile_size * 0.5)
        return max(0, min(1, x_value)), max(0, min(1, y_value))

    def bottom_right_filter(self, i: int, j: int) -> tuple:
        """
        Generate filter values for the bottom-right corner of the tile.
        
        Args:
            i (int): Row index within the tile.
            j (int): Column index within the tile.
        
        Returns:
            tuple: Tuple containing x and y filter values.
        
        Notes:
            - Calculate the distance `c` from the bottom-right corner using the Pythagorean theorem.
            - If `c` is less than or equal to half the tile size (`thresh_distance`), return (1, 1) indicating full filter intensity.
            - Otherwise, normalize `x_value` and `y_value` based on their distance from the corners of the tile, scaled by half the tile size.
            - Ensure that the returned values are clipped between 0 and 1 to maintain valid filter values.
        """
        c = math.sqrt(((self.tile_size - j) ** 2) + ((self.tile_size - i) ** 2))
        thresh_distance = self.tile_size * 0.5
        if c <= thresh_distance:
            return 1, 1
        x_value = j / (self.tile_size * 0.5)
        y_value = i / (self.tile_size * 0.5)
        return max(0, min(1, x_value)), max(0, min(1, y_value))
    
    def center_filter(self, i: int, j: int) -> tuple:
        """
        Generate filter values for the center region of the tile.
        
        Args:
            i (int): Row index within the tile.
            j (int): Column index within the tile.
        
        Returns:
            tuple: Tuple containing x and y filter values.
        
        Notes:
            - Calculate the center coordinates and radius of the tile.
            - Determine the inner radius as a fraction of the tile's radius (`inner_radius`).
            - Calculate the distance `distance_to_center` from the center of the tile.
            - If `distance_to_center` is within the inner radius, return (1, 1) indicating full filter intensity.
            - If `distance_to_center` is within the tile radius but outside the inner radius, normalize `x_value` and `y_value`
            based on their distance from the center, scaling by the tile radius.
            - Ensure that the returned values are clipped between 0 and 1 to maintain valid filter values.
        """
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