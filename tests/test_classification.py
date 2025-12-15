import numpy as np
from nuclei_segmentation.classification import classify_nuclei_by_brownness

def test_brownness_classification_order():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # 2 fake nuclei
    labels = np.zeros((100, 100), dtype=int)
    labels[10:30, 10:30] = 1  
    labels[50:80, 50:80] = 2 

    image[10:30, 10:30] = [50, 50, 50]    # dark
    image[50:80, 50:80] = [200, 200, 200] # bright

    outlines = classify_nuclei_by_brownness(image, labels)

    assert len(outlines) == 2

def test_outline_generation():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    labels = np.zeros((64, 64), dtype=int)
    labels[20:40, 20:40] = 1

    outlines = classify_nuclei_by_brownness(image, labels)

    assert outlines is not None