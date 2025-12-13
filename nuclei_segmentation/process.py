from cellpose import models
import logging

from nuclei_segmentation.segmentation import watershed_segmentation
from nuclei_segmentation.classification import classify_nuclei_by_brownness
from nuclei_segmentation.visualization import draw_outlines
from nuclei_segmentation.cellpose_seg import segment_cellpose, classify_cellpose

logger = logging.getLogger(__name__)

def process_image(image, method = 'watershed'):
    if method == "watershed":
        logger.info("Performing watershed segmentation...")
        seg_labeled_image, _ = watershed_segmentation(image, 'adaptive')

        logger.info("Classifying nuclei by brownness...")
        outline_color_pairs = classify_nuclei_by_brownness(image, seg_labeled_image)
    
    elif method == "deep_learning":
        logger.info("Performing deep learning segmentation with cellpose...")
        model = models.CellposeModel(gpu=True)
        seg_labeled_image = segment_cellpose(image, model)
        outline_color_pairs = classify_cellpose(image, seg_labeled_image)
    else:
        raise ValueError("Unknown segmentation method")

    logger.info("Drawing outlines on image...")
    classified_image = draw_outlines(image, outline_color_pairs, thickness=2)
    return classified_image