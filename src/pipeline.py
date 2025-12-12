import cv2
import numpy as np
import time

from segmentation import threshold_segmentation, watershed_segmentation
from classification import classify_nuclei_by_brownness
from parallel import process_image_in_parallel


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

def segment_and_classify(source: str, output: str) -> None:
    """
    Segments and classifies nuclei in the given image.

    Parameters:
        source (str): Path to the input image.
        output (str): Path to the output image.
    """
    image = load_image(source)

    # Crop the image
    crop_size = 2000
    image = image[:crop_size, :crop_size, :]

    # Perform segmentation and classification
    seg_labeled_image, _ = watershed_segmentation(image, 'adaptive')
    contour_color_pairs = classify_nuclei_by_brownness(image, seg_labeled_image)
    

    classified_image = image.copy()
    for outline, color in contour_color_pairs:
        cv2.drawContours(classified_image, [outline], -1, color, thickness=2)

    # Save the output
    save_image(output, classified_image)


def segment_and_classify_large(source: str, output: str, tile_size=2000, overlap=200, workers=None) -> None:
    image = load_image(source)
    classified_image = process_image_in_parallel(image, tile_size=tile_size, overlap=overlap, max_workers=workers)
    save_image(output, classified_image)


if __name__ == "__main__":
    start_time = time.time()

    segment_and_classify_large("../example/input.jpg", "../notebooks/classified_output_parallel.jpg")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Processing time: {elapsed:.2f} seconds")