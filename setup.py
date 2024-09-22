from setuptools import setup, find_packages

setup(
    name="MathModel",  # The name of your module/package
    version="0.1",     # The version of your module
    packages=find_packages(),  # This automatically finds all the subpackages
    install_requires=[
        'trimesh',  # Add trimesh as a required package
        'numpy',    # Add numpy as a required package
    ],  # You can list any dependencies here
)
