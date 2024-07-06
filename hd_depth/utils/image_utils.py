import cv2
import numpy as np
from typing import Tuple

def resize_and_pad_image(raw_image: np.ndarray, size: int, pad_to_square: bool = False, pad_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Resize an image while maintaining aspect ratio. Optionally, pad the image to make it square.

    Parameters:
    - raw_image (np.ndarray): Input image as a NumPy array.
    - size (int): Desired size of the larger dimension after resizing.
    - pad_to_square (bool): If True, pad the image to make it square.
    - pad_color (Tuple[int, int, int]): Color for padding (if pad_to_square is True). Default is black (0, 0, 0).

    Returns:
    - np.ndarray: Resized and optionally padded image in NumPy array format.
    
    Raises:
    - ValueError: If the input raw_image is None or has invalid dimensions.
    """
    if raw_image is None:
        raise ValueError(f"Failed to read raw image")

    h, w = raw_image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Invalid image dimensions: height={h}, width={w}")

    # Determine resizing ratio based on the larger dimension
    if h > w:
        new_h, new_w = size, int(size * (w / h))
    else:
        new_h, new_w = int(size * (h / w)), size

    # Resize the image
    resized_image = cv2.resize(raw_image, (new_w, new_h))

    if pad_to_square:
        # Calculate padding sizes
        delta_w = size - new_w
        delta_h = size - new_h

        # Set top, bottom, left, right padding sizes
        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left

        # Add padding
        resized_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)

    return resized_image


def resize_to_original(raw_image: np.ndarray, original_h: int, original_w: int) -> np.ndarray:
    """
    Resize an image back to its original height and width.

    Parameters:
    - raw_image (np.ndarray): Input image as a NumPy array.
    - original_h (int): Original height of the image.
    - original_w (int): Original width of the image.

    Returns:
    - np.ndarray: Resized image in NumPy array format.

    Raises:
    - ValueError: If the input raw_image is None.
    """
    if raw_image is None:
        raise ValueError(f"Failed to read raw image")

    resized_image = cv2.resize(raw_image, (original_w, original_h))
    return resized_image
