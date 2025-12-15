# Nuclei Segmentation

A Python package for **segmenting, classifying, and visualizing nuclei** in microscopy images. It provides both a fast analytical method using watershed segmentation and an optional deep learning-based method using Cellpose.  

---

## Features

- Segment nuclei in microscopy images using:
  - **Watershed (analytical, parallelizable with tiling, fast)**
  - **Deep learning (Cellpose, more accurate but slower, not parallelized)**
- Classify nuclei based on **brownness levels**.
- Visualize classified nuclei with colored outlines.
- CLI for batch processing of images.

> **Remark:** Parallelizing the watershed segmentation brings a 5–6× improvement in processing speed.
---

## Installation

```bash
git clone https://github.com/dorimolnar/nuclei_segmentation.git
cd nuclei_segmentation
pip install -e .
```

## Requirements

- **Python**: 3.10  (for `cellpose`)
- **Packages**:
  - `numpy`
  - `scikit-image`
  - `opencv-python`
  - `matplotlib`
  - `scipy`
  - `tqdm`
  - `cellpose`


## Usage

The package can be used via the command line or imported in Python scripts/notebooks.

### Command Line Interface (CLI)

Basic usage:

```bash
nuclei-seg input.jpg output.jpg --method [method]
```
#### Arguments:
- `input.jpg`: Path to the input image.
- `output.jpg`: Path to save the output image.
- `method`: Optional. Segmentation method to use (watershed or deep_learning). Default is watershed.

### Python Usage

```bash
from nuclei_segmentation.pipeline import segment_and_classify, segment_and_classify_parallel

# Segment and classify a single image
segment_and_classify("example/input.jpg", "example/output.jpg", method="watershed")

# For large images, use parallel tiling (only with watershed method)
segment_and_classify_parallel("example/input.jpg", "example/output_parallel.jpg")
```

## Example Notebook

The notebooks/example_notebook.ipynb shows how to:
- Load an example image
- Run segmentation and classification
- Visualize intermediate results (masks, outlines, classification colors)

This is useful for exploring how parameters affect segmentation or for creating custom visualizations.

> **Remark:** The package was developed and tested with the dependency versions listed in `setup.py`.

