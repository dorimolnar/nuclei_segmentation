import numpy as np
from nuclei_segmentation.classification import classify_nuclei_by_brownness

def test_returns_one_outline_per_label():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    labels = np.zeros((100, 100), dtype=int)
    labels[10:20, 10:20] = 1
    labels[30:40, 30:40] = 2

    outlines = classify_nuclei_by_brownness(image, labels)

    assert len(outlines) == 2 # one outline per nucleus

def test_brownness_affects_color_assignment():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    labels = np.zeros((64, 64), dtype=int)

    labels[10:20, 10:20] = 1
    labels[30:40, 30:40] = 2

    image[10:20, 10:20] = [40, 40, 40]    # dark
    image[30:40, 30:40] = [220, 220, 220] # bright

    outlines = classify_nuclei_by_brownness(image, labels)

    _, color1 = outlines[0]
    _, color2 = outlines[1]

    assert color1 != color2 # different brownness (darkness) should yield different colors