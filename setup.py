import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="in2IN",
    version="1.0",
    description="",
    author="Pablo Ruiz Ponce",
    packages=find_packages(include=['src', 'src.*']),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
)