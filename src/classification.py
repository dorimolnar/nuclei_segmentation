import numpy as np
from skimage.measure import regionprops
from skimage.draw import polygon_perimeter
from skimage import measure
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import cv2


CLASS_COLORS = {
    0: [255, 0, 0],       # Red
    1: [255, 165, 0],     # Orange
    2: [255, 255, 0],     # Yellow
    3: [0, 0, 255],       # Blue
}



def smooth_contour(contour, sigma=10, pts=200):
    # Ensure closed curve
    contour = contour[:, 0, :]    # (N, 2)
    contour = contour.astype(np.float32)

    if len(contour) < 4:
        # fallback to linear interpolation
        kind = "linear"
    else:
        kind = "cubic"

    # Close the contour if needed
    # if not np.allclose(contour[0], contour[-1]):
    #     contour = np.vstack([contour, contour[0]])

    # Interpolate to make uniformly sampled curve
    t = np.linspace(0, 1, len(contour))
    fx = interp1d(t, contour[:, 0], kind=kind)
    fy = interp1d(t, contour[:, 1], kind=kind)

    t2 = np.linspace(0, 1, pts)
    x2 = fx(t2)
    y2 = fy(t2)

    # Smooth with Gaussian filter
    x_s = gaussian_filter1d(x2, sigma=sigma, mode='wrap')
    y_s = gaussian_filter1d(y2, sigma=sigma, mode='wrap')

    smoothed = np.stack([x_s, y_s], axis=1).astype(np.int32)
    return smoothed.reshape(-1, 1, 2)

def draw_nucleus_outline(image, coords, bbox, color, method='contour', thickness=2):

    (min_r, min_c, max_r, max_c) = bbox
    h = max_r - min_r
    w = max_c - min_c
 
    # Create a binary mask for this nucleus
    mask = np.zeros((h,w), dtype=np.uint8)

    # Shift coordinates to local bounding box
    local_coords = coords.copy()
    local_coords[:, 0] -= min_r
    local_coords[:, 1] -= min_c

    mask[local_coords[:, 0], local_coords[:, 1]] = 1
    
    # Smooth the mask to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask_smooth = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_CLOSE, kernel)

    if method == 'contour':
        contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return
        outline = max(contours, key=lambda c: c.shape[0])

    elif method == 'conv_hull':
        ys, xs = np.where(mask_smooth > 0)
        pts = np.column_stack([xs, ys])
        outline = cv2.convexHull(pts)
        if outline is None:
            return
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Discard too short outlines
    if outline.shape[0] < 10:
        return

    # Smooth the contour
    outline_smooth = smooth_contour(outline, sigma=10)

    # Shift back to original image coordinates
    outline_smooth[:, 0, 0] += min_c  
    outline_smooth[:, 0, 1] += min_r 

    cv2.drawContours(image, [outline_smooth], -1, color, thickness)

    # Gives sharp outlines
    # approx = cv2.approxPolyDP(contour_int, 5.0, True)
    # cv2.drawContours(image, contours, -1, color=color, thickness=thickness)

def classify_nuclei_by_brownness(image: np.ndarray, labeled: np.ndarray) -> np.ndarray:
    """
    Classify nuclei by brownness and overlay 1–2 pixel outlines.

    Parameters:
        image (np.ndarray): RGB original image
        labeled (np.ndarray): Labeled image (0=background, 1,2,...=nuclei)

    Returns:
        np.ndarray: RGB image with nuclei outlines colored by class
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # 0: black, 255: white
    props = regionprops(labeled, intensity_image=gray)

    # Collect mean intensity of each nucleus
    mean_intensities = np.array([region.mean_intensity for region in props])

    # Define thresholds for classes
    thresholds_percentile = np.percentile(mean_intensities, [30, 55, 75])
    thresholds = [80,120,170]
    
    # Classes: 0,1,2,3
    classes = np.digitize(mean_intensities, bins=thresholds)

    # Prepare output image
    output = image.copy()

    for region, class_id in zip(props, classes):
        coords = region.coords
        bbox = region.bbox
        color = CLASS_COLORS[class_id]
        draw_nucleus_outline(output, coords, bbox, color=color, method ='contour', thickness=2)

    return output