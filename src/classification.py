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


def get_nucleus_outline(coords, mask_shape):
    # coords = region.coords
    # Create a binary mask for this nucleus
    mask = np.zeros(mask_shape, dtype=np.uint8)
    mask[coords[:,0], coords[:,1]] = 1
    
    # Find contours
    contours = measure.find_contours(mask, 0.5)  # 0.5 threshold
    if len(contours) == 0:
        return np.array([]), np.array([])
    
    # Take the longest contour (largest perimeter)
    contour = max(contours, key=lambda x: x.shape[0])
    rr, cc = contour[:,0].astype(int), contour[:,1].astype(int)
    return rr, cc

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
    if not np.allclose(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])

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

def draw_nucleus_outline(image, coords, color, thickness=2):
    # Create a binary mask for this nucleus
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[coords[:,0], coords[:,1]] = 1
    
    
    kernel = np.ones((5, 5), np.uint8)
    mask_smooth = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return
    contour = max(contours, key=lambda c: c.shape[0])
    cnt_smooth = smooth_contour(contour, sigma=10)

    cv2.drawContours(image, [cnt_smooth], -1, color, thickness)

    # Gives sharp outlines
    # approx = cv2.approxPolyDP(contour_int, 5.0, True)
    # cv2.drawContours(image, contours, -1, color=color, thickness=thickness)

def classify_nuclei_by_brownness(image: np.ndarray, labeled: np.ndarray) -> np.ndarray:
    """
    Classify nuclei by brownness and overlay 1–2 pixel outlines.

    Parameters:
        image (np.ndarray): RGB original image
        labeled_mask (np.ndarray): Labeled mask (0=background, 1,2,...=nuclei)

    Returns:
        np.ndarray: RGB image with outlines colored by class
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    props = regionprops(labeled, intensity_image=gray)

    # Collect mean intensity of each nucleus
    mean_intensities = np.array([region.mean_intensity for region in props])

    # Define thresholds for classes
    thresholds = np.percentile(mean_intensities, [15, 30, 65])
    # Classes: 0,1,2,3
    classes = np.digitize(mean_intensities, bins=thresholds)

    # Prepare output image (copy of original or blank)
    output = image.copy()

    # Draw outlines
    # for region, cls in zip(props, classes):
    #     # Coordinates of the nucleus
    #     coords = region.coords
    #     # Convert coords to polygon perimeter
    #     rr, cc = get_nucleus_outline(coords, mask_shape=labeled.shape)
    #     # Assign color
    #     color = CLASS_COLORS[cls]
    #     # Draw on output image
    #     output[rr, cc] = color

    for region, cls in zip(props, classes):
        coords = region.coords
        color = CLASS_COLORS[cls]
        draw_nucleus_outline(output, coords, color=color, thickness=2)

    return output