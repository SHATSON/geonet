"""GeoNet package setup."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="geonet",
    version="1.0.0",
    author="[Author Name]",
    author_email="[author.email@university.edu]",
    description=(
        "GeoNet: Hyperbolic and Riemannian Geometry as Inductive Biases "
        "for Deep Neural Network Architecture Design"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[author]/geonet",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "geonet-train=train:main",
            "geonet-eval=evaluate:main",
        ],
    },
)
