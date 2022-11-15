"""Builds the marktech package from the root folder of the project.

To do so run the command below in the root folder:
pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="marktech",
    version="0.1.0",
    packages=find_packages(exclude=('tests',)),
    author="Pierre Gouedard",
    author_email="pierre.mgouedard@gmail.com",
    description="Package implementing financial time series regression, with basic analysis functionality",
)
