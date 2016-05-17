#!/usr/bin/env python
from distutils.core import setup
from setuptools import find_packages

setup(
    name='keras-vgg-buddy',
    version='0.1.1',
    description='Your pal when you want some VGG16 with your Keras.',
    author='Adam Wentz',
    author_email='adam@adamwentz.com',
    url='https://github.com/awentzonline/keras-vgg-buddy/',
    packages=find_packages(),
    install_requires=[
        'h5py>=2.5.0',
        'Keras>=1.0.1',
        'numpy>=1.10.4',
        'Pillow>=3.1.1',
        'PyYAML>=3.11',
        'scipy>=0.17.0',
        'scikit-learn>=0.17.0',
        'six>=1.10.0',
    ],
)
