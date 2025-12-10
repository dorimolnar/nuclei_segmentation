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