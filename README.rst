=========
evalutils
=========

.. image:: https://badge.fury.io/py/evalutils.svg
    :target: https://badge.fury.io/py/evalutils

.. image:: https://travis-ci.org/comic/evalutils.svg?branch=master
    :target: https://travis-ci.org/comic/evalutils

.. image:: https://ci.appveyor.com/api/projects/status/jc27lltkwkxp06y1/branch/master?svg=true
    :target: https://ci.appveyor.com/project/jmsmkn/evalutils/branch/master

.. image:: https://api.codeclimate.com/v1/badges/5c3b7f45f6a476d0f21e/maintainability
   :target: https://codeclimate.com/github/comic/evalutils/maintainability
   :alt: Maintainability

.. image:: https://api.codeclimate.com/v1/badges/5c3b7f45f6a476d0f21e/test_coverage
   :target: https://codeclimate.com/github/comic/evalutils/test_coverage
   :alt: Test Coverage

.. image:: https://readthedocs.org/projects/evalutils/badge/?version=latest
        :target: https://evalutils.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/comic/evalutils/shield.svg
     :target: https://pyup.io/repos/github/comic/evalutils/
     :alt: Updates

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
