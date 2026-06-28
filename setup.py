from setuptools import setup, find_packages

setup(
    name="vnl_loss",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0"
    ],
    author="Dhyey Pandya",
    description="GPU-accelerated Virtual Normal Loss (VNL) for 3D depth estimation",
    python_requires=">=3.7",
)