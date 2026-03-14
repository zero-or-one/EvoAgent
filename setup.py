"""
setup.py
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="evoagent",
    version="1.0.0",
    description="A self-evolving neural agent that reads and modifies its own weights",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="EvoAgent Contributors",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
        "viz": [
            "matplotlib>=3.6",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="neural network evolutionary strategy self-modifying weights meta-learning",
)
