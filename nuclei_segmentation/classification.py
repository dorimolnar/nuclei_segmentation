import numpy as np
from skimage.measure import regionprops
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



def smooth_outline(outline, sigma=10, pts=200):
    """
    Smooths the outline using cubic spline interpolation and Gaussian filtering.
    
    Parameters:
        outline (np.ndarray): Outline points (N, 1, 2)
        sigma (int): Standard deviation for Gaussian filter
        pts (int): Number of points in the smoothed outline

    Returns:
        np.ndarray: Smoothed outline points (pts, 1, 2)
    """
    if outline.ndim == 3:
        outline = outline[:, 0, :]    # (N, 2)
    outline = outline.astype(np.float32)

    kind = 'cubic'

    # Interpolate to make uniformly sampled curve
    t = np.linspace(0, 1, len(outline))
    fx = interp1d(t, outline[:, 0], kind=kind)
    fy = interp1d(t, outline[:, 1], kind=kind)

    t2 = np.linspace(0, 1, pts)
    x2 = fx(t2)
    y2 = fy(t2)

    # Smooth with Gaussian filter
    x_s = gaussian_filter1d(x2, sigma=sigma, mode='wrap')
    y_s = gaussian_filter1d(y2, sigma=sigma, mode='wrap')

    smoothed = np.stack([x_s, y_s], axis=1).astype(np.int32)
    return smoothed.reshape(-1, 1, 2)

def find_nucleus_outline(coords, bbox, method='contour'):
    """
    Finds the outline of one nucleus given its coordinates and bounding box.
    
    Parameters:
        coords (np.ndarray): Nucleus pixel coordinates (N, 2)
        bbox (tuple): Bounding box (min_row, min_col, max_row, max_col)
        method (str): Method to find outline ('contour' or 'conv_hull')
        
    Returns:
        np.ndarray: Smoothed outline points (M, 1, 2) or None if not found
    """

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
            return None
        outline = max(contours, key=lambda c: c.shape[0])

    elif method == 'conv_hull':
        ys, xs = np.where(mask_smooth > 0)
        pts = np.column_stack([xs, ys])
        outline = cv2.convexHull(pts)
        if outline is None:
            return None
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Discard too short outlines
    if outline.shape[0] < 10:
        return None

    # Smooth the outline
    outline_smooth = smooth_outline(outline, sigma=10)

    # Shift back to original image coordinates
    outline_smooth[:, 0, 0] += min_c  
    outline_smooth[:, 0, 1] += min_r 

    # Discard if the outline is not closed
    p0 = outline_smooth[0, 0]
    p_end = outline_smooth[-1, 0]
    if not np.allclose(p0, p_end, atol=2.0):
        return None

    return outline_smooth


    # Gives sharp outlines
    # approx = cv2.approxPolyDP(contour_int, 5.0, True)
    # cv2.drawContours(image, contours, -1, color=color, thickness=thickness)

def classify_nuclei_by_brownness(image, labeled):
    """
    Classify nuclei by brownness and return outline–color pairs

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
    thresholds = [120,135,150]

    # Classes: 0,1,2,3
    classes = np.digitize(mean_intensities, bins=thresholds)

    # Prepare output image
    outline_color_pairs = []

    for region, class_id in zip(props, classes):
        coords = region.coords
        bbox = region.bbox
        color = CLASS_COLORS[class_id]
        outline = find_nucleus_outline(coords, bbox, method ='contour')

        if outline is None:
            continue

        outline_color_pairs.append((outline, color))


    return outline_color_pairs

