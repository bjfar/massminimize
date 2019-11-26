import os
import re
import sys
import sysconfig
import site
import platform
from distutils.version import LooseVersion
from setuptools import setup, Extension

setup(
    name='massminimize',
    version='0.0.1',
    author='Ben Farmer',
    # Add yourself if you contribute to this package
    author_email='ben.farmer@gmail.com',
    description='A tool for performing many parallel minimisations of a single target function with varying input data, built on TensorFlow',
    long_description='',
    packages=['massminimize'],
    install_requires=[
        'numpy',
        'tensorflow>=2',
    ],
)
