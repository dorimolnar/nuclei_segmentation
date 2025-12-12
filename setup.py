from setuptools import setup, find_packages

setup(
    name="nuclei_segmentation",  
    version="0.1",
    description="Python package for detecting, classifying, and outlining cell nuclei in microscopy images.",
    author="Dora Molnar",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "scikit-image",
        "opencv-python",
        "matplotlib"
    ],
)