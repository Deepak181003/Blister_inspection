"""
preprocessing.py
================
Module responsible for image acquisition, validation, and standard
OpenCV classical preprocessing operations:
  1. Image loading & error handling
  2. Grayscale conversion
  3. Gaussian blur (noise reduction)
  4. Adaptive and Otsu thresholding (binarization)
  5. Canny edge detection (structural boundary extraction)
  6. Morphological operations (opening, closing, erosion, dilation)
  7. ROI (Region of Interest) utilities
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


def load_image(image_path: str) -> np.ndarray:
    """
    Safely loads an image from the specified file path.

    Why this matters for beginners:
      - cv2.imread() returns None if the path is invalid or the file is corrupted.
      - If you try to call cv2.cvtColor() on None, Python throws an unhelpful
        'NoneType has no shape' error.
      - This function validates file existence and format before returning.

    Args:
        image_path (str): The filesystem path to the blister pack image.

    Returns:
        np.ndarray: Loaded BGR image as a NumPy array.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the file exists but cannot be decoded as an image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: '{image_path}'")

    # Read image using standard OpenCV BGR format
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Failed to decode image at '{image_path}'. Supported formats: .jpg, .png, .bmp.")

    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts a 3-channel color image (BGR) into a 1-channel Grayscale image.

    Why this is used:
      - Color images contain 3 channels (Blue, Green, Red). Processing all 3
        triples computation time without adding structural or edge information.
      - Grayscale intensity represents surface brightness (0 = pitch black, 255 = bright white),
        which is ideal for detecting pill contours, edges, and cavity shadows.

    Args:
        image (np.ndarray): Input BGR image (Height x Width x 3).

    Returns:
        np.ndarray: Grayscale image (Height x Width).
    """
    if len(image.shape) == 2:
        # Already grayscale
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(gray_image: np.ndarray,
                        kernel_size: Tuple[int, int] = (5, 5),
                        sigma_x: float = 1.0) -> np.ndarray:
    """
    Applies Gaussian smoothing to suppress high-frequency sensor noise.

    Why this is used:
      - Digital camera sensors create subtle grain/noise.
      - Edge detectors (like Canny) calculate pixel gradients; without blurring,
        they mistake sensor noise for real cracks or tears.
      - A Gaussian kernel uses a bell-shaped weighted average where the center
        pixel has the highest influence, preserving edges better than a box blur.

    Args:
        gray_image (np.ndarray): Single-channel grayscale image.
        kernel_size (Tuple[int, int]): Dimensions of the Gaussian filter window.
                                       Must be positive odd integers (e.g., (3,3), (5,5)).
        sigma_x (float): Standard deviation along the horizontal axis.

    Returns:
        np.ndarray: Smoothed grayscale image.
    """
    return cv2.GaussianBlur(gray_image, kernel_size, sigmaX=sigma_x)


def apply_adaptive_threshold(blurred_image: np.ndarray,
                             max_value: int = 255,
                             block_size: int = 21,
                             c_constant: int = 4) -> np.ndarray:
    """
    Converts a grayscale image into a binary (black & white) image using
    local adaptive thresholding.

    Why this is used:
      - Pharmaceutical packaging lines suffer from non-uniform lighting
        (reflections on aluminium foil, shadows inside pockets).
      - Global thresholding (e.g., single threshold of 127) fails when one side
        of the blister pack is brighter than the other.
      - Adaptive thresholding computes an individual threshold for every pixel
        based on the Gaussian mean of its neighborhood (block_size x block_size)
        minus a tuning constant C.

    Args:
        blurred_image (np.ndarray): Smoothed grayscale image.
        max_value (int): Pixel value assigned to foreground (usually 255).
        block_size (int): Size of the pixel neighborhood (must be odd > 1).
        c_constant (int): Constant subtracted from the calculated mean.

    Returns:
        np.ndarray: Binary image (0 or 255).
    """
    return cv2.adaptiveThreshold(
        blurred_image,
        maxValue=max_value,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=c_constant
    )


def apply_otsu_threshold(blurred_image: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Automatically calculates an optimal global threshold using Otsu's method.

    Why this is used:
      - Otsu's algorithm maximizes inter-class variance between background and foreground.
      - Serves as a complementary binarization technique when inspecting high-contrast
        tablets inside darker cavity backgrounds.

    Args:
        blurred_image (np.ndarray): Smoothed grayscale image.

    Returns:
        Tuple[float, np.ndarray]: (Optimal threshold value, Binary mask).
    """
    thresh_val, binary_mask = cv2.threshold(
        blurred_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh_val, binary_mask


def apply_canny_edge(image: np.ndarray,
                     low_threshold: int = 50,
                     high_threshold: int = 150) -> np.ndarray:
    """
    Detects sharp boundaries and edges using the Canny Edge Detector.

    How Canny works:
      1. Computes directional gradients using Sobel kernels.
      2. Non-maximum suppression thins multi-pixel edges to 1-pixel wide ridges.
      3. Hysteresis thresholding uses two thresholds:
         - Gradient > high_threshold: definitely an edge.
         - Gradient < low_threshold: definitely not an edge (discarded).
         - Between low and high: retained only if connected to a 'sure' edge.

    Why this is used:
      - Ideal for locating tablet perimeter cracks, foil seal tear borders,
        and deformed pocket margins.

    Args:
        image (np.ndarray): Grayscale or blurred image.
        low_threshold (int): Lower hysteresis limit (typically 30-70).
        high_threshold (int): Upper hysteresis limit (typically 120-180).

    Returns:
        np.ndarray: Binary edge map where edge pixels are 255 and background is 0.
    """
    return cv2.Canny(image, low_threshold, high_threshold)


def apply_morphology(binary_image: np.ndarray,
                     operation: str = 'close',
                     kernel_size: Tuple[int, int] = (3, 3),
                     iterations: int = 1) -> np.ndarray:
    """
    Applies mathematical morphological operations to clean binary masks.

    Operations supported:
      - 'erode': Shrinks bright regions; eliminates tiny isolated white dots (salt noise).
      - 'dilate': Expands bright regions; bridges tiny gaps.
      - 'open': Erosion followed by Dilation; removes background noise without altering object shape.
      - 'close': Dilation followed by Erosion; fills pinholes inside pills and seals small cracks.

    Args:
        binary_image (np.ndarray): Single-channel binary image.
        operation (str): One of ['erode', 'dilate', 'open', 'close'].
        kernel_size (Tuple[int, int]): Size of structuring element.
        iterations (int): How many times to repeat the operation.

    Returns:
        np.ndarray: Cleaned binary image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    op = operation.lower()

    if op == 'erode':
        return cv2.erode(binary_image, kernel, iterations=iterations)
    elif op == 'dilate':
        return cv2.dilate(binary_image, kernel, iterations=iterations)
    elif op == 'open':
        return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif op == 'close':
        return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        raise ValueError(f"Unsupported morphological operation '{operation}'. Use 'erode', 'dilate', 'open', or 'close'.")


def preprocess_pipeline(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Executes the complete preprocessing sequence on a blister pack image.

    Returns a structured dictionary holding intermediate visual representations
    so the 5-panel inspection dashboard can render each stage:
      - 'original': Original BGR image
      - 'gray': Grayscale representation
      - 'blurred': Noise-suppressed image
      - 'thresh': Clean binary threshold mask
      - 'edges': Canny edge map

    Args:
        image (np.ndarray): Raw BGR blister pack image.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing all visual stages.
    """
    gray = to_grayscale(image)
    blurred = apply_gaussian_blur(gray, kernel_size=(5, 5), sigma_x=1.2)
    thresh = apply_adaptive_threshold(blurred, block_size=19, c_constant=3)
    cleaned_thresh = apply_morphology(thresh, operation='close', kernel_size=(3, 3), iterations=1)
    edges = apply_canny_edge(blurred, low_threshold=50, high_threshold=140)

    return {
        'original': image,
        'gray': gray,
        'blurred': blurred,
        'thresh': cleaned_thresh,
        'edges': edges
    }
