from setuptools import setup, find_packages
import os

if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = ""

setup(
    name="vector-lighting-edges",
    version="1.0.0",
    author="Панченко Александр Алексеевич",
    author_email="sascha.panchenko2018@yandex.ru", 
    description="Edge detection using vector lighting method for RGB images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nervni-Sanya/Edge-detection-benchmark",
    project_urls={
        "Bug Tracker": "https://github.com/Nervni-Sanya/Edge-detection-benchmark/issues",
        "Source Code": "https://github.com/Nervni-Sanya/Edge-detection-benchmark",
        "Documentation": "https://github.com/Nervni-Sanya/Edge-detection-benchmark#readme",
        "Author Profile": "https://orcid.org/0009-0009-9104-1214", 
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
        "examples": ["pillow", "matplotlib"],
    },
    keywords="edge detection, image processing, vector lighting, computer vision",
)
