import io
import os
from os import path
import re
from setuptools import setup, find_packages
# To use consisten encodings
from codecs import open

# Function from: https://github.com/pytorch/vision/blob/master/setup.py


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()

# Function from: https://github.com/pytorch/vision/blob/master/setup.py


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    long_description = readme_file.read()

VERSION = find_version('pthflops', '__init__.py')

requirements = [
    'torch'
]

setup(
    name='pthflops',
    version=VERSION,

    description="Estimate FLOPs of neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",

    # Author details
    author="Adrian Bulat",
    author_email="adrian@adrianbulat.com",
    url="https://github.com/1adrianb/pytorch-estimate-flops",

    # Package info
    packages=find_packages(exclude=('test',)),

    install_requires=requirements,
    license='BSD',
    zip_safe=True,

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',

        # Supported python versions
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
