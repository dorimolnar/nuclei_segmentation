from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import numpy as np
from tqdm import tqdm
import logging
from cellpose import models

from nuclei_segmentation.segmentation import watershed_segmentation
from nuclei_segmentation.classification import classify_nuclei_by_brownness
from nuclei_segmentation.visualization import draw_outlines
from nuclei_segmentation.cellpose_seg import segment_cellpose, classify_cellpose


logger = logging.getLogger(__name__)

def process_image(image: np.ndarray, method: str = 'watershed') -> np.ndarray:
    """
    Segments and classifies nuclei in the given image.
    Parameters:
        image (np.ndarray): RGB input image
        method (str): Segmentation method, either 'watershed' or 'deep_learning'
    Returns:
        np.ndarray: Classified image with outlines drawn
    """
    if method == "watershed":
        logger.info("Performing watershed segmentation...")
        seg_labeled_image, _ = watershed_segmentation(image, 'adaptive')

        logger.info("Classifying nuclei by brownness...")
        outline_color_pairs = classify_nuclei_by_brownness(image, seg_labeled_image)
    
    elif method == "deep_learning":
        logger.info("Performing deep learning segmentation with cellpose...")
        model = models.CellposeModel(gpu=True)
        seg_labeled_image = segment_cellpose(image, model, diameter=30)
        outline_color_pairs = classify_cellpose(image, seg_labeled_image)
    else:
        raise ValueError("Unknown segmentation method")

    logger.info("Drawing outlines on image...")
    classified_image = draw_outlines(image, outline_color_pairs, thickness=2)
    return classified_image



def process_tile_for_outlines(tile: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    """
    Process a single tile: segment and classify nuclei, returning outline-color pairs.
    Parameters:
        tile (np.ndarray): RGB input tile
    Returns:
        list: List of (outline, color) pairs
    """
    # Temporarily suppress logging
    previous_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.CRITICAL)
    try:
        seg_labeled_tile, _  = watershed_segmentation(tile, 'adaptive')
        outline_color_pairs = classify_nuclei_by_brownness(tile, seg_labeled_tile)
    finally:
        logging.getLogger().setLevel(previous_level)
    return outline_color_pairs


def process_image_in_parallel(image: np.ndarray, 
                              tile_size: int = 1024,
                              overlap: int = 200,
                              max_workers: int | None = None) -> np.ndarray:
    """
    Process the image in tiles using parallel processing (with watershed segmentation), extracting only outline-color pairs.
    All outlines are drawn at the end on a full-size image.
    
    Parameters:
        image (np.ndarray): RGB input image
        tile_size (int): Size of each tile
        overlap (int): Overlap between tiles
        max_workers (int or None): Number of parallel workers to use
        
    Returns:
        np.ndarray: Classified image with outlines drawn
    """
    H, W = image.shape[:2]
    output = image.copy()

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 1) - 1)

    # Build list of tile coordinates
    tasks = []
    step = tile_size - overlap
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            tasks.append((y0, y1, x0, x1))

    # Collect all outline-color pairs
    all_outline_color_pairs = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_tile_for_outlines, image[y0:y1, x0:x1]): (y0, x0) for y0, y1, x0, x1 in tasks}

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Tiles"):
            origin = futures[fut]
            y0, x0 = origin
            try:
                tile_outlines = fut.result()
                if tile_outlines is None:
                    continue
                # Shift outlines to global coordinates
                for outline, color in tile_outlines:
                    outline_global = outline.copy()
                    outline_global[:, 0, 0] += x0 
                    outline_global[:, 0, 1] += y0
                    all_outline_color_pairs.append((outline_global, color))
            except Exception as e:
                raise RuntimeError(f"Error processing tile at {origin}: {e}")

    # Draw all outlines on the full output image
    classified_image = draw_outlines(output, all_outline_color_pairs, thickness=2)
    return classified_image