"""
ASAM: Adaptive Sparse Attention Module
=======================================

An efficient attention mechanism with hardware-optimized implementations.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="asam-attention",
    version="1.1.0",
    author="ASAM Team",
    author_email="your.email@example.com",
    description="Adaptive Sparse Attention Module with Flash Attention optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/asam-attention",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "benchmark": [
            "matplotlib>=3.5.0",
        ],
    },
)
