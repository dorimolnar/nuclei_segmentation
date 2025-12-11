import cv2
import numpy as np

def load_image(path: str) -> np.ndarray:
    """
    Loads an RGB image from the given path using OpenCV.

    Parameters:
        path (str): Path to the image file.

    Returns:
        np.ndarray: Image array in RGB format (H, W, 3).
    """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    
    if image is None:
        raise FileNotFoundError(f"Image not found at {path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def save_image(path: str, image: np.ndarray) -> None:
    """
    Saves an RGB image to the given path using OpenCV.

    Parameters:
        path (str): Path where the image will be saved.
        image (np.ndarray): Image array in RGB format (H, W, 3).
    """
    # Convert RGB to BGR for OpenCV
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        cv2.imwrite(path, bgr_image)
    except Exception as e:
        raise IOError(f"An error occurred while saving the image: {e}")