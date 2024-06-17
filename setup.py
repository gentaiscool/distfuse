import os
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="distfuse",
    version="0.1.0",
    author="Genta Indra Winata",
    author_email="gentaindrawinata@gmail.com",
    description="Automatic Factorization package for PyTorch modules",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gentaiscool/distfuse",
    project_urls={
        "Bug Tracker": "https://github.com/SamuelCahyawijaya/greenformer/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "scikit-learn>=1.5.0",
        "sentence_transformers",
        "numpy"
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.3",
)