# evalutils

[![Build Status](https://github.com/comic/evalutils/workflows/CI/badge.svg)](https://github.com/comic/evalutils/actions?query=workflow%3ACI+branch%3Amaster)
[![PyPI version](https://badge.fury.io/py/evalutils.svg)](https://badge.fury.io/py/evalutils)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/evalutils)](https://pypi.org/project/evalutils/)
[![Documentation Status](https://img.shields.io/badge/docs-passing-4a4c4c1.svg)](https://comic.github.io/evalutils/)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

evalutils helps challenge administrators easily create evaluation
containers for grand-challenge.org.

  - Free software: MIT license
  - Documentation: <https://comic.github.io/evalutils/>.

## Features

  - Generation your challenge evaluation project boilerplate using
    [Cookiecutter](https://github.com/audreyr/cookiecutter)
  - Scripts to build, test and export your generated docker container
    for grand-challenge.org
  - Loading of CSV, ITK and Pillow compatible prediction files
  - Validation of submitted predictions
  - Interface to SciKit-Learn metrics and Pandas aggregations
  - Bounding box annotations with Jaccard Index calculations

## Getting Started

[evalutils](https://github.com/comic/evalutils) requires Python 3.7 or
above, and can be installed from `pip`.
Please see the [Getting Started](https://comic.github.io/evalutils/usage.html)
documentation for more details.
