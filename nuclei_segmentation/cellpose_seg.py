from cellpose import utils
import cv2
import numpy as np

from nuclei_segmentation.classification import smooth_outline

#io.logger_setup()


CLASS_COLORS = {
    0: [255, 0, 0],       # Red
    1: [255, 165, 0],     # Orange
    2: [255, 255, 0],     # Yellow
    3: [0, 0, 255],       # Blue
}

def segment_cellpose(image, model):
    masks, _, _ = model.eval(image, diameter=30)
    return masks

def classify_cellpose(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    n_labels = mask.max()
    mean_intensity = np.zeros(n_labels, dtype=float)
    for i in range(1, n_labels+1):
        mean_intensity[i-1] = np.mean(gray[mask == i])

    thresholds = [130, 150, 170]
    classes_per_nucleus = np.digitize(mean_intensity, bins=thresholds)

    outlines_list = utils.outlines_list_multi(mask)

    outline_color_pairs = []
    for i, outline in enumerate(outlines_list):
        if outline.shape[0] < 10:
            return None
        
        class_id = classes_per_nucleus[i]       
        color = CLASS_COLORS[class_id]

        # Smooth the outline
        outline_smooth = smooth_outline(outline, sigma=10)

        # Discard if the outline is not closed
        p0 = outline_smooth[0, 0]
        p_end = outline_smooth[-1, 0]
        if not np.allclose(p0, p_end, atol=2.0):
            return None

        outline_color_pairs.append((outline, color))
    
    return outline_color_pairs