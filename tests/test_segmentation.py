import numpy as np
from nuclei_segmentation.segmentation import watershed_segmentation

def test_watershed_output_shape():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[40:80, 40:80] = 255  # fake nucleus

    labels, _ = watershed_segmentation(image)

    assert labels.shape == image.shape[:2]
    assert labels.max() >= 1  # at least one nucleus detected