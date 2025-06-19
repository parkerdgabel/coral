"""Setup script for coral package"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="coral-ml",
    version="1.0.0",
    author="Coral Contributors",
    description="Neural network weight storage and deduplication system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/parkerdgabel/coral",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "h5py>=3.0.0",
        "protobuf>=3.19.0",
        "xxhash>=3.0.0",
        "tqdm>=4.60.0",
        "networkx>=2.6.0",
    ],
    entry_points={
        "console_scripts": [
            "coral-ml=coral.cli.main:main",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
            "flake8>=4.0.0",
        ],
        "torch": ["torch>=1.10.0"],
        "tensorflow": ["tensorflow>=2.8.0"],
    },
)