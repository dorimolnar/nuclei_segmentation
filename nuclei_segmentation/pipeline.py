import cv2
import logging

from nuclei_segmentation.process import process_image, process_image_in_parallel


logger = logging.getLogger(__name__)


def load_image(path):
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

def save_image(path, image):
    """
    Saves an RGB image to the given path using OpenCV.

    Parameters:
        path (str): Path where the image will be saved.
        image (np.ndarray): Image array in RGB format (H, W, 3).
    """
    # Convert RGB to BGR for OpenCV
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        # cv2.imwrite(path, bgr_image)
        cv2.imwrite(path, bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 85]) # To make the file smaller
    except Exception as e:
        raise IOError(f"An error occurred while saving the image: {e}")
    
def crop_image(image, crop_size=2000):
    """
    Crops the image to the specified size from the top-left corner.

    Parameters:
        image (np.ndarray): Input image array (H, W, 3).
        crop_size (int): Size to crop the image to (crop_size x crop_size).

    Returns:
        np.ndarray: Cropped image array.
    """
    return image[:crop_size, :crop_size, :]
    


def segment_and_classify(source, output, method='watershed'):
    """
    Segments and classifies nuclei in the given image.

    Parameters:
        source (str): Path to the input image.
        output (str): Path to the output image.
    """

    logger.info(f"Loading image from {source}")
    image = load_image(source)

    logger.info("Cropping image...")
    image = crop_image(image, crop_size=2000)
    
    classified_image = process_image(image, method = method)

    logger.info(f"Saving output to {output}")
    save_image(output, classified_image)

    logger.info("Segmentation and classification finished")


def segment_and_classify_parallel(source, output, tile_size=2000, overlap=200, workers=None):
    """
    Segments and classifies nuclei in a large image using parallel processing.

    Parameters:
        source (str): Path to the input image.
        output (str): Path to the output image.
        tile_size (int): Size of each tile to process.
        overlap (int): Overlap between tiles.
        workers (int or None): Number of parallel workers to use.
    """

    logger.info(f"Loading image from {source}")
    image = load_image(source)

    logger.info("Processing image in parallel...")
    classified_image = process_image_in_parallel(image, tile_size=tile_size, overlap=overlap, max_workers=workers)

    logger.info(f"Saving output to {output}")
    save_image(output, classified_image)

    logger.info("Segmentation and classification finished")
