from setuptools import find_packages, setup

setup(
    name="agcm-net",
    version="1.0.0",
    description="Adaptive Gating and Connected-Component Merging Network for "
                "small-object detection in high-resolution aerial maritime imagery",
    packages=find_packages(exclude=["scripts", "examples", "tests"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "opencv-python>=4.5",
        "torch>=1.13",
        "ultralytics>=8.3.0",
        "pycocotools>=2.0",
    ],
)
