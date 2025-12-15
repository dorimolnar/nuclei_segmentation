"""
Functions to visualize images and draw colored outlines on nuclei.
"""


import matplotlib.pyplot as plt
import cv2

def visualize_image(image):
    """
    Visualizes an image using Matplotlib.

    Parameters:
        image (np.ndarray): Image array in RGB/grayscale format (H, W, 3) or (H, W).
    """
    plt.imshow(image, cmap = 'gray')
    plt.axis('off')
    plt.show()

def draw_outlines(image, outline_color_pairs, thickness=2):
    """
    Draws outlines on the image.

    Parameters:
        image (np.ndarray): RGB image
        outline_color_pairs (list): List of (outline: np.ndarray(N,1,2), color) tuples
        thickness (int): Thickness of the outline

    Returns:
        np.ndarray: Image with drawn outlines
    """
    outlined_image = image.copy()
    for outline, color in outline_color_pairs:
        cv2.drawContours(outlined_image, [outline], -1, color, thickness=thickness)
    return outlined_image