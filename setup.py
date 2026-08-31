"""Setup script for CIFAR-10 Image Classification project."""

from pathlib import Path
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="cifar10-image-classification",
    version="1.0.0",
    author="Hamza Ahmed Sharif",
    author_email="hamzaahmedsharif@example.com",
    description="End-to-End Deep Learning Pipeline on CIFAR-10 with GPU Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HamzaAhmedSharif/Project-4-CIFAR-10-Image-Classification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.7.1",
        "torchvision>=0.22.1",
        "gradio>=4.44.1",
        "pillow>=10.0.0",
        "numpy>=2.0.0",
        "pandas>=2.2.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cifar10-demo=app.gradio_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "app": ["examples/*.png"],
    },
)