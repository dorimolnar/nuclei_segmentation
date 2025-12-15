"""
Functions for nuclei segmentation using thresholding and watershed methods.
"""

import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def to_grayscale(image):
    """
    Convert an RGB image to grayscale.
    
    Parameters:
        image (np.ndarray): RGB image (H, W, 3)
        
    Returns:
        np.ndarray: Grayscale image (H, W)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray

def smooth_image(gray, kernel_size=5):
    """
    Apply Gaussian blur to a grayscale image.
    
    Parameters:
        gray (np.ndarray): Grayscale image
        kernel_size (int): Size of the Gaussian kernel
        
    Returns:
        np.ndarray: Smoothed image
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    return blurred

def threshold_image(gray, thr_method='adaptive'):
    """
    Apply Otsu/Adaptive thresholding to grayscale image.
    
    Parameters:
        gray (np.ndarray): Grayscale image
        thr_method (str): Thresholding method ('otsu' or 'adaptive')
        
    Returns:
        np.ndarray: Binary mask (nuclei = 1, background = 0)
    """
    if thr_method == 'otsu':
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif thr_method == 'adaptive':
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 301, 10)
    else:
        raise ValueError(f"Unknown thresholding method: {thr_method}")

    # Convert to 0 = background, 1 = nuclei
    binary = (binary == 0).astype(np.uint8)
    return binary

def label_nuclei(binary):
    """
    Label connected components in a binary image.
    
    Parameters:
        binary (np.ndarray): Binary mask
        
    Returns:
        np.ndarray: Labeled mask (0 = background, 1,2,... = nuclei)
    """
    labeled, _ = ndi.label(binary)
    return labeled

def threshold_segmentation(image):
    """
    Perform nuclei segmentation using simple thresholding.
    
    Parameters:
        image (np.ndarray): RGB input image

    Returns:
        np.ndarray: Labeled mask (0 = background, 1,2,... = nuclei)
        np.ndarray: Colored labeled mask only for visualization
    """
    gray = to_grayscale(image)
    blurred = smooth_image(gray)
    binary = threshold_image(blurred)
    labeled = label_nuclei(binary)
    colored = label2rgb(labeled, bg_label=0)
    return labeled, colored

def watershed_segmentation(image, thr_method='adaptive'):
    """
    Perform nuclei segmentation using watershed to separate touching nuclei.
    
    Parameters:
        image (np.ndarray): RGB input image
        thr_method (str): Thresholding method ('otsu' or 'adaptive')

    Returns:
        np.ndarray: Labeled mask (0 = background, 1,2,... = nuclei)
        np.ndarray: Colored labeled mask only for visualization
    """
    gray = to_grayscale(image)
    blurred = smooth_image(gray)
    binary = threshold_image(blurred, thr_method)
    distance = ndi.distance_transform_edt(binary)
    distance_smooth = cv2.GaussianBlur(distance, (15,15), 0)

    coords = peak_local_max(distance_smooth, min_distance = 30, labels=binary, footprint=np.ones((5, 5)))
    markers = np.zeros_like(distance_smooth, dtype=int)
    for i, (y, x) in enumerate(coords, 1):
        markers[y, x] = i
    
    labeled = watershed(-distance_smooth, markers, mask=binary)
    colored = label2rgb(labeled, bg_label=0)
    return labeled, colored