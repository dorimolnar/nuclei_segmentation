import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale.
    
    Parameters:
        image (np.ndarray): RGB image (H, W, 3)
        
    Returns:
        np.ndarray: Grayscale image (H, W)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray

def smooth_image(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian blur to a grayscale image.
    
    Parameters:
        gray (np.ndarray): Grayscale image
        kernel_size (int): Size of the Gaussian kernel
        
    Returns:
        np.ndarray: Smoothed image
    """
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    return blurred

def threshold_image(gray: np.ndarray) -> np.ndarray:
    """
    Apply Otsu thresholding to grayscale image.
    
    Parameters:
        gray (np.ndarray): Grayscale image
        
    Returns:
        np.ndarray: Binary mask (nuclei = 1, background = 0)
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Convert to 0 and 1
    binary = binary // 255
    binary_corrected = 1 - binary
    return binary_corrected

def label_nuclei(binary: np.ndarray) -> np.ndarray:
    """
    Label connected components in a binary image.
    
    Parameters:
        binary (np.ndarray): Binary mask
        
    Returns:
        np.ndarray: Labeled mask (0 = background, 1,2,... = nuclei)
    """
    labeled, _ = ndi.label(binary)
    return labeled

def threshold_segmentation(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    blurred = smooth_image(gray)
    binary = threshold_image(blurred)
    labeled = label_nuclei(binary)
    colored = label2rgb(labeled, bg_label=0)
    return colored

def watershed_segmentation(image: np.ndarray) -> np.ndarray:
    """
    Perform nuclei segmentation using watershed to separate touching nuclei.
    
    Parameters:
        image (np.ndarray): RGB input image
        min_size (int): Minimum size of nuclei to keep (remove small noise)
        
    Returns:
        np.ndarray: Labeled mask (0 = background, 1,2,... = nuclei)
    """
    gray = to_grayscale(image)
    blurred = smooth_image(gray)
    binary = threshold_image(blurred)
    distance = ndi.distance_transform_edt(binary)

    coords = peak_local_max(distance, min_distance = 10, labels=binary, footprint=np.ones((3, 3)))
    markers = np.zeros_like(distance, dtype=int)
    for i, (y, x) in enumerate(coords, 1):
        markers[y, x] = i
    
    labeled = watershed(-distance, markers, mask=binary)
    colored = label2rgb(labeled, bg_label=0)
    return colored