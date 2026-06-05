from setuptools import find_packages, setup

setup(
    name="sdteoray-practica2",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ray[tune]",
        "torch",
        "torchvision",
        "matplotlib",
        "pandas",
    ],
)
