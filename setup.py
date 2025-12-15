from setuptools import setup, find_packages

setup(
    name="nuclei_segmentation",  
    version="0.1",
    description="Python package for detecting, classifying, and outlining cell nuclei in microscopy images.",
    author="Dora Molnar",
    packages=find_packages(),
    python_requires=">=3.10,<3.11",
    install_requires=[
        "numpy",
        "scikit-image",
        "opencv-python",
        "matplotlib",
        "scipy",
        "tqdm",
        "cellpose",
    ],
    entry_points={
        "console_scripts": [
            "nuclei-seg=nuclei_segmentation.cli:main",
        ],
    },
)