from setuptools import setup, find_packages

setup(
    name="nuclei_segmentation",  
    version="0.1",
    description="Python package for detecting, classifying, and outlining cell nuclei in microscopy images.",
    author="Dora Molnar",
    packages=find_packages(),
    python_requires=">=3.10,<3.11",
    install_requires=[
        "numpy==2.2.6",
        "scikit-image==0.25.2",
        "opencv-python==4.12.0.88",
        "matplotlib==3.10.8",
        "scipy==1.15.3",
        "tqdm==4.67.1",
        "cellpose==4.0.8",
    ],
    entry_points={
        "console_scripts": [
            "nuclei-seg=nuclei_segmentation.cli:main",
        ],
    },
)