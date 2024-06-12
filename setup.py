import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="in2IN",
    version="1.0",
    description="",
    author="Pablo Ruiz Ponce",
    packages=find_packages(include=['in2in', 'in2in.*']),
    install_requires=[
        'numpy',
        'tqdm',
        'lightning',
        'scipy',
        'matplotlib',
        'pillow',
        'yacs',
        'mmcv',
        'opencv-python',
        'tabulate',
        'termcolor',
        'smplx',
        'torch',
        'torchvision',
        'torchaudio',
        'pykeops',
    ],
    dependency_links=[
        'git+https://github.com/openai/CLIP.git',
    ]
)