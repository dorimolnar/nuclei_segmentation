"""
Cellpose-based segmentation and classification utilities.
"""

from cellpose import utils
from cellpose.models import CellposeModel
import cv2
import numpy as np

from nuclei_segmentation.classification import smooth_outline
from nuclei_segmentation.classification import CLASS_COLORS


# For R-B based brownness metric
# CLASS_COLORS = {
#     3: [255, 0, 0],       # Red
#     2: [255, 165, 0],     # Orange
#     1: [255, 255, 0],     # Yellow
#     0: [0, 0, 255],       # Blue
# }

def segment_cellpose(image: np.ndarray, model: CellposeModel, diameter: int = 30) -> np.ndarray:
    """
    Segment nuclei in the image using Cellpose model.
    Parameters:
        image (np.ndarray): RGB input image
        model (CellposeModel): Preloaded Cellpose model
    Returns:
        np.ndarray: Labeled mask (0 = background, 1,2,... = nuclei)
    """
    masks, _, _ = model.eval(image, diameter=diameter)
    return masks

def classify_cellpose(image: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    """
    Classify nuclei by brownness and return outline–color pairs

    Parameters:
        image (np.ndarray): RGB original image
        mask (np.ndarray): Labeled mask (0=background, 1,2,...=nuclei)

    Returns:
        list: List of (outline, color) pairs
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    n_labels = mask.max()
    if n_labels == 0:
        return []
    mean_intensity = np.zeros(n_labels, dtype=float)
    for i in range(1, n_labels+1):
        mean_intensity[i-1] = np.mean(gray[mask == i])

    # Darker = lower
    thresholds = [120, 140, 160]
    classes_per_nucleus = np.digitize(mean_intensity, bins=thresholds)

    # R-B based brownness metric
    # R = image[:, :, 0]
    # B = image[:, :, 2]

    # n_labels = mask.max()
    # brownness = np.zeros(n_labels, dtype=float)

    # for i in range(1, n_labels + 1):
    #     nucleus = (mask == i)
    
    #     # Brownness metric: high R, low B
    #     brownness[i - 1] = np.mean(
    #         (R[nucleus] - B[nucleus]) / (R[nucleus] + B[nucleus] + 1e-6)
    #     )       

    # thresholds = [0.05, 0.15, 0.3]
    # classes_per_nucleus = np.digitize(brownness, bins=thresholds)



    outlines_list = utils.outlines_list_multi(mask)

    outline_color_pairs = []
    for i, outline in enumerate(outlines_list):
        if outline.shape[0] < 4:
            continue
        
        class_id = classes_per_nucleus[i]       
        color = CLASS_COLORS[class_id]

        # Smooth the outline - not needed
        # outline_smooth = smooth_outline(outline, sigma=10)


        outline_color_pairs.append((outline, color))
    
    return outline_color_pairs