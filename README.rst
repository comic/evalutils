=========
evalutils
=========

.. image:: https://badge.fury.io/py/evalutils.svg
   :target: https://badge.fury.io/py/evalutils

.. image:: https://github.com/comic/evalutils/workflows/CI/badge.svg
   :target: https://github.com/comic/evalutils/actions?query=workflow%3ACI+branch%3Amaster
   :alt: Build Status

.. image:: https://codecov.io/gh/comic/evalutils/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/comic/evalutils
   :alt: Code Coverage Status

.. image:: https://readthedocs.org/projects/evalutils/badge/?version=latest
   :target: https://evalutils.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black


evalutils helps challenge administrators easily create evaluation containers for grand-challenge.org.

* Free software: MIT license
* Documentation: https://evalutils.readthedocs.io.

Features
--------

* Generation your challenge evaluation project boilerplate using Cookiecutter_
* Scripts to build, test and export your generated docker container for grand-challenge.org
* Loading of CSV, ITK and Pillow compatible prediction files
* Validation of submitted predictions
* Interface to SciKit-Learn metrics and Pandas aggregations
* Bounding box annotations with Jaccard Index calculations


Getting Started
---------------

evalutils_ requires Python 3.6, and can be installed from `pip`. Please
see the `Getting Started`_ documentation for more details.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _evalutils: https://github.com/comic/evalutils
.. _`Getting Started`: https://evalutils.readthedocs.io/en/latest/usage.html
