# evalutils

[![image](https://badge.fury.io/py/evalutils.svg)](https://badge.fury.io/py/evalutils)

[![Build
Status](https://github.com/comic/evalutils/workflows/CI/badge.svg)](https://github.com/comic/evalutils/actions?query=workflow%3ACI+branch%3Amaster)

[![Code Coverage
Status](https://codecov.io/gh/comic/evalutils/branch/master/graph/badge.svg)](https://codecov.io/gh/comic/evalutils)

[![Documentation
Status](https://readthedocs.org/projects/evalutils/badge/?version=latest)](https://evalutils.readthedocs.io/en/latest/?badge=latest)

[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

evalutils helps challenge administrators easily create evaluation
containers for grand-challenge.org.

  - Free software: MIT license
  - Documentation: <https://evalutils.readthedocs.io>.

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
Please see the [Getting
Started](https://evalutils.readthedocs.io/en/latest/usage.html)
documentation for more details.
