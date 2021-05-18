#!/usr/bin/env python

import os
from typing import Dict

from setuptools import find_packages, setup

NAME = "evalutils"
REQUIRES_PYTHON = ">=3.7"

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "pandas!=0.24.0",
    "imageio",
    "SimpleITK",
    "cookiecutter",
    "click",
    "scipy",
    "scikit-learn",
    "numpy",
]

test_requirements = ["pytest", "pytest-cov", "pytest-xdist", "pytest-randomly"]

here = os.path.abspath(os.path.dirname(__file__))

# Load the package's __version__.py module as a dictionary.
about: Dict[str, str] = {}
with open(os.path.join(here, NAME, "__version__.py")) as f:
    exec(f.read(), about)

setup(
    author="James Meakin",
    author_email="code@jmsmkn.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description=(
        "evalutils helps users create extensions for grand-challenge.org."
    ),
    extras_require={"test": test_requirements},
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="evalutils",
    name=NAME,
    packages=find_packages(include=["evalutils"]),
    python_requires=REQUIRES_PYTHON,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/comic/evalutils",
    version=about["__version__"],
    zip_safe=False,
    entry_points={"console_scripts": ["evalutils = evalutils.__main__:main"]},
)
