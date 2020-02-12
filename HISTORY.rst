=======
History
=======

0.2.2 (2020-02-12)
------------------

* Updated default container labels for algorithms

0.2.1 (2020-02-12)
------------------

* Added support for mha input for algorithms

0.2.0 (2020-02-11)
------------------

* Added experimental support for templating algorithms
  * Note that the `init` cli command has changed
* Added support for Python 3.8

0.1.17 (2019-05-23)
-------------------

* Fixes build and test scripts on windows
* Fixes BallTree execution with scipy 0.21.1

0.1.16 (2019-02-13)
-------------------

* Fixes bug in `export.sh` due to extra }

0.1.15 (2019-02-01)
-------------------

* Pins Pandas version due to a bug with dataframe conversion to dictionaries in Pandas 0.24.0
* Updates the export.sh template to gzip the docker container after creation
* Added Python 3.7 support

0.1.14 (2018-10-16)
-------------------

* Adds `evalutils.stats` for calculating common metrics in medical imaging
* Adds **experimental**  Windows support - we do not have CI on all of the windows tests so please report any errors
* Adds `evalutils.roc` for calculating bootstrapped roc curves

0.1.13 (2018-09-12)
-------------------

* Adds Segmentation and Detection challenges to the project generator
* File loaders now return lists of dictionaries rather than dictionaries
* Renamed `intersection_over_union` to `jaccard_index`
* Improved image memory management. Added `io.ImageLoader` and separate `load_image` and `hash_image` functions.
* `score_case` is no longer a static method


0.1.12 (2018-05-16)
-------------------

* Fixed a bug where the number of `false_negatives` could be less than 0.

0.1.11 (2018-05-15)
-------------------

* **Breaking change:** Renamed ``Evaluation`` to ``ClassificationEvaluation``
* Adds support for ``DetectionEvaluation``

0.1.10 (2018-04-19)
-------------------

* Simplifies the example template

0.1.9 (2018-04-19)
------------------

* **Breaking change:** Renamed ``bb2`` to ``other`` in ``BoundingBox()``
* ``ground_truth_path`` is no longer a required argument

0.1.8 (2018-04-18)
------------------

* Fixes template folder in distribution

0.1.7 (2018-04-18)
------------------

* Adds cookiecutter templating for generating new projects
* Adds equality check for ``BoundingBox``

0.1.6 (2018-03-30)
------------------

* Improves pandas csv handling

0.1.5 (2018-03-30)
------------------

* Corrects loading of some CSV files
* Adds logging and more tests
* Adds referencing to ``_ground_truth`` and ``_prediction`` in joined pandas tables


0.1.3 (2018-03-29)
------------------

* Adds basic implementation with
    * Full Evaluation workflow
    * CSV, SimpleITK, and ImageIO loaders
    * BoundingBox annotations with intersection, union and intersection over union metrics
    * Unique File Indices, Unique Image, Expected Column Names and Number of cases validators


0.1.0 (2018-03-22)
------------------

* First release on PyPI.
