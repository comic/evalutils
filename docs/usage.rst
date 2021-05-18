=====
Usage
=====

Getting Started
---------------

This guide shows you how to use ``evalutils`` to generate either an
evaluation container or an algorithm container for `Grand Challenge`_.
Select the appropriate project below to get started.

Evaluation container
--------------------

This guide will show you how to use ``evalutils`` to generate an evaluation
container for `Grand Challenge`_. In this example we will call our project
``myproject``, substitute your project name wherever you see this.

Prerequisites
^^^^^^^^^^^^^

Before you start you will need to have:

* A local `docker`_ installation
* Your challenge test set ground truth data, this can be a CSV file or a set of images
* An idea about what metrics you want to score the submissions on

Generate The Project Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evalutils contains a project generator based on `CookieCutter`_ that you can
use to generate the boilerplate for your evaluation.
Once you have installed evalutils you can see the options with:

.. code-block:: console

    $ evalutils init evaluation --help

Say that you want to create an evaluation for ``myproject``, you can initialize
it with:

.. code-block:: console

    $ evalutils init evaluation myproject

You will then be prompted to choose a challenge type:

.. code-block:: console

    $ What kind of challenge is this? [Classification|Segmentation|Detection]:

so type in your challenge type and press <enter>.
The different challenge types that you can select are:

- **Classification**:
    The submission and ground truth are csv files with the same number of rows.
    For instance, this evaluation could be used for scoring classification of whole images into 1 or multiple classes.
    The result of the evaluation is not reported on the case level to prevent leaking of the ground truth data.
- **Segmentation**:
    A special case of a classification task, the difference is that the submission and ground truth are image files (eg, ITK images or a collection of PNGs).
    For instance, this evaluation could be used for scoring structure segmentation in 3D images.
    There are the same number images in the ground truth dataset as there are in each submission.
    By default, the results per case are also reported.
- **Detection**:
    The submission and ground truth are csv files, but with differing number of rows.
    For instance, this evaluation could be used for scoring detection of tumours in images.
    For this sort of challenge, you may have many candidate points and many ground truth points per case.
    By default, the results per case are also reported.

If you do not have a local python 3.7+ environment you can also
generate your project with docker by running a container and sharing your current user id:

.. code-block:: console

    $ docker run -it --rm -u `id -u` -v $(pwd):/usr/src/myapp -w /usr/src/myapp python:3 bash -c "pip install evalutils && evalutils init evaluation myproject"

Either of these commands will generate a folder called ``myproject``
with everything you need to get started.

It is a good idea to commit your project to git right now. You can do this with:

.. code-block:: console

    $ cd myproject
    $ git init
    $ git lfs install   (see the warning below)
    $ git add --all
    $ git commit -m "Initial Commit"

.. warning:: The test set ground truth will be stored in this repo,
    so remember to use a private repo if you're going to push this to github or gitlab,
    and use `git lfs`_ if your ground truth data are large.

    The .gitattributes file at the root of the repository specifies all the files which should be
    tracked by git-lfs. By default all files in the ground truth and test directories
    are configured to be tracked by git-lfs, but they will only be registered
    once the `git lfs`_ extension is installed on your system and the :console:`git lfs install`
    command has been issued inside the generated repository.


The structure of the project will be:

.. code-block:: console

    .
    └── myproject
        ├── build.sh            # Builds your evaluation container
        ├── Dockerfile          # Defines how to build your evaluation container
        ├── evaluation.py       # Contains your evaluation code - this is where you will extend the Evaluation class
        ├── export.sh           # Exports your container to a .tar file for use on grand-challenge.org
        ├── .gitattributes      # Define which files git should put under git-lfs
        ├── .gitignore          # Define which files git should ignore
        ├── ground-truth        # A folder that contains your ground truth annotations
        │   └── reference.csv   # In this example the ground truth is a csv file
        ├── README.md           # For describing your evaluation to others
        ├── requirements.txt    # The python dependencies of your evaluation container - add any new dependencies here
        ├── test                # A folder that contains an example submission for testing
        │   └── submission.csv  # In this example the participants will submit a csv file
        └── test.sh             # A script that runs your evaluation container on the test submission

For Segmentation tasks, some example mhd/zraw files will be in the ground-truth and test directories instead.

The most important file is ``evaluation.py``.
This is the file where you will extend the ``Evaluation`` class and implement the evaluation for your challenge.
In this file, a new class has been created for you, and it is instantiated and run with:

.. code-block:: python

    if __name__ == "__main__":
        Myproject().evaluate()


This is all that is needed for ``evalutils`` to perform the evaluation and generate the output for each new submission.
The superclass of ``Evaluation`` is what you need to adapt to your specific challenge.

Classification Tasks
~~~~~~~~~~~~~~~~~~~~

The boilerplate for classification challenges looks like this:

.. code-block:: python

    class Myproject(ClassificationEvaluation):
        def __init__(self):
            super().__init__(
                file_loader=CSVLoader(),
                validators=(
                    ExpectedColumnNamesValidator(expected=("case", "class",)),
                    NumberOfCasesValidator(num_cases=8),
                ),
                join_key="case",
            )

        def score_aggregates(self):
            return {
                "accuracy_score": accuracy_score(
                    self._cases["class_ground_truth"],
                    self._cases["class_prediction"],
                 ),
            }

In this case the evaluation is loading csv files, so uses an instance ``CSVLoader`` which will do the loading of the data.
In this example, both the ground truth and the prediction CSV files will contain the columns `case` (an index) and `class` (the predicted class of this case).
We want to validate that the correct columns appear in both the ground truth and submitted predictions, so we use the ``ExpectedColumnNamesValidator`` with the names of the columns we expect to find.
We also use the ``NumberOfCasesValidator`` to check that the correct number of cases has been submitted by the challenge participant.
See :mod:`evalutils.validators` for a list of other validators that you can use.

The ground truth and predictions will be loaded into two DataFrames.
The last argument is a ``join_key``, the is the name of the column that will appear in both DataFrames that serves as an index to join the dataframes on in order to create ``self._cases``.
The ``join_key`` is manditory when you use a ``CSVLoader``.
This should be set to some sort of common index, such as a `case` identifier.
When loading in files they are first going to be sorted so you might not need a ``join_key``, but you could also write a function that matches the cases based on filename.

.. warning:: It is best practice to include an integer in the (file) name that uniquely defines each case.
    For instance, name your testing set files case_001, case_002, ... etc.

The last part is performing the actual evaluation.
In this example we are only getting one number per submission, the accuracy score.
This number is calculated using ``sklearn.metrics.accuracy_score``.
The ``self._cases`` data frame will contain all of the columns that you expect, and for those that have not been joined they will be available as ``"<column_name>_ground_truth"`` and ``"<column_name>_prediction"``.

If you need to score cases individually before aggregating them, you should remove the implementation of ``score_aggregates`` and implement ``score_case`` instead.

Segmentation Tasks
~~~~~~~~~~~~~~~~~~

For segmentation tasks, the generated code will look like this:

.. code-block:: python

    class Myproject(ClassificationEvaluation):
        def __init__(self):
            super().__init__(
                file_loader=SimpleITKLoader(),
                validators=(
                    NumberOfCasesValidator(num_cases=2),
                    UniquePathIndicesValidator(),
                    UniqueImagesValidator(),
                ),
            )

        def score_case(self, *, idx, case):
            gt_path = case["path_ground_truth"]
            pred_path = case["path_prediction"]

            # Load the images for this case
            gt = self._file_loader.load_image(gt_path)
            pred = self._file_loader.load_image(pred_path)

            # Check that they're the right images
            assert self._file_loader.hash_image(gt) == case["hash_ground_truth"]
            assert self._file_loader.hash_image(pred) == case["hash_prediction"]

            # Cast to the same type
            caster = SimpleITK.CastImageFilter()
            caster.SetOutputPixelType(SimpleITK.sitkUInt8)
            gt = caster.Execute(gt)
            pred = caster.Execute(pred)

            # Score the case
            overlap_measures = SimpleITK.LabelOverlapMeasuresImageFilter()
            overlap_measures.Execute(gt, pred)

            return {
                'FalseNegativeError': overlap_measures.GetFalseNegativeError(),
                'FalsePositiveError': overlap_measures.GetFalsePositiveError(),
                'MeanOverlap': overlap_measures.GetMeanOverlap(),
                'UnionOverlap': overlap_measures.GetUnionOverlap(),
                'VolumeSimilarity': overlap_measures.GetVolumeSimilarity(),
                'JaccardCoefficient': overlap_measures.GetJaccardCoefficient(),
                'DiceCoefficient': overlap_measures.GetDiceCoefficient(),
                'pred_fname': pred_path.name,
                'gt_fname': gt_path.name,
            }

Here, we are loading ITK files in the ground-truth and test folders using ``SimpleITKLoader``.
See :mod:`evalutils.io` for the other image loaders you could use.
By default, the files will be matched together based on the first integer found in the filename, so name your ground truth files, for example, case_001.mha, case_002.mha, etc.
Have the participants for your challenge do the same.

The loader will try to load all of the files in the ground-truth and submission folders.
To check that the correct number of images were submitted by the participant and loaded we use ``NumberOfCasesValidator``, and check that the images are unique by using ``UniquePathIndicesValidator`` and ``UniqueImagesValidator``

The ``score_case`` function will calculate the score for each case, in this case we're calculating some overlap measures using ``SimpleITK``.
The images are not stored in the case dataframe to save memory, so first they are loaded using the file loader, and are then checked that they are the valid images by calculating the hash.
The filenames are also stored for the case for matching later on grand-challenge.

The aggregate results are automatically calculated using ``score_aggregates``, which calls ``DataFrame.describe()``.
By default, this will calculate the mean, quartile ranges and counts of each individual metric.

Detection Tasks
~~~~~~~~~~~~~~~

The generated boilerplate for detection tasks is:

.. code-block:: python

    class Myproject(DetectionEvaluation):
        def __init__(self):
            super().__init__(
                file_loader=CSVLoader(),
                validators=(
                    ExpectedColumnNamesValidator(
                        expected=("image_id", "x", "y", "score")
                    ),
                ),
                join_key="image_id",
                detection_radius=1.0,
                detection_threshold=0.5,
            )

        def get_points(self, *, case, key):
            """
            Converts the set of ground truth or predictions for this case, into
            points that represent true positives or predictions
            """
            try:
                points = case.loc[key]
            except KeyError:
                # There are no ground truth/prediction points for this case
                return []

            return [
                (p["x"], p["y"])
                for _, p in points.iterrows()
                if p["score"] > self._detection_threshold
            ]

In this case, we are loading a CSV file with ``CSVLoader``, but do not validate the number of rows as they can be different between the ground truth and submissions.
We validate the column headers in both files.
In this case, we identify the cases with ``image_id``, and both files contain ``x`` and ``y`` locations, with a confidence score of ``score``.
In the ground truth dataset the score should be set to 1.

By default, The predictions will be thresholded at ``detection_threshold``.
The detection evaluation will count the closest prediction that lies within distance ``detection_radius`` from the ground truth point as a true positive.
See :mod:`evalutils.scorers` for more information on the algorithm.

The only function that needs to be implemented is ``get_points``, which converts a case row to a list of points which are later matched.
In this case, we're acting on 2D images, but you could extend ``(p["x"], p["y"])`` to say ``(p["x"], p["y"], p["z"])`` if you have 3D data.

By default, the f1 score, precision and accuracy are calculated for each case, see the ``DetectionEvaluation`` class for more information.

Add The Ground Truth and Test Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The next step is to add your ground truth and test data (an example submission) to the repo.
If using CSV data simply update the ``ground-truth/reference.csv`` file, and then update the expected column names and join key in evaluate.py.
Otherwise, see :mod:`evalutils.io` for other loaders such as the ones for ITK files or images.
You can also add your own loader by extending the ``FileLoader`` class.

Adapt The Evaluation
^^^^^^^^^^^^^^^^^^^^

Change the function in the boilerplate to fit your needs, refer to the superclass methods for more information on return types.
See :class:`evalutils.Evaluation` for more possibilities.

Build, Test and Export
^^^^^^^^^^^^^^^^^^^^^^

When you're ready to test your evaluation you can simply invoke

.. code-block:: console

    $ ./test.sh

This will build your docker container, add the test data as a temporary volume, run the evaluation, and then ``cat /output/metrics.json``.
If the output looks ok, then you're ready to go.

You can export the evaluation container with

.. code-block:: console

    $ ./export.sh

which will create myproject.tar in the folder.
You can then upload this directly to `Grand Challenge`_ on your evaluation methods page.


Algorithm container
-------------------

This guide will show you how to use ``evalutils`` to generate an algorithm
container for `Grand Challenge`_. In this example we will call our project
``myproject``, substitute your project name wherever you see this.

Prerequisites
^^^^^^^^^^^^^

Before you start you will need to have:

* A local `docker`_ installation
* Some test images for your algorithm, preferably with expected output

Generate The Project Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evalutils contains a project generator based on `CookieCutter`_ that you can
use to generate the boilerplate for your algorithm.
Once you have installed evalutils you can see the options with:

.. code-block:: console

    $ evalutils init algorithm --help

Say that you want to create an evaluation for ``myproject``, you can initialize
it with:

.. code-block:: console

    $ evalutils init algorithm myproject

You will then be prompted to choose an algorithm type:

.. code-block:: console

    $ What kind of algorithm is this? [Classification|Segmentation|Detection]:

so type in your algorithm type and press <enter>.
The different algorithm types that you can select are:

- **Classification**:
    This type of algorithm takes in an image (eg, ITK images or a collection of PNGs) and outputs a `/output/results.json` file.
    For instance, this algorithm could be used for classification of whole images into 1 or multiple classes.
    By default, the algorithm outputs a single `/output/results.json` which lists the results per case.
- **Segmentation**:
    A special case of a classification task, that takes in an image and outputs an image file to `/output/images/` (eg, ITK images or a collection of PNGs).
    For instance, this algorithm could be used for structure segmentation in 3D images.
    By default, the algorithm outputs an image file at `/output/images/` per case and a single `/output/results.json` file with additional information for all cases.
- **Detection**:
    This type of algorithm detects one or more candidates in an image and returns the positions relative to the image.
    For instance, this evaluation could be used for detection of tumours in images.
    By default, the algorithm outputs a single `/output/results.json` which lists the results per case.

If you do not have a local python 3.7+ environment you can also
generate your project with docker by running a container and sharing your current user id:

.. code-block:: console

    $ docker run -it --rm -u `id -u` -v $(pwd):/usr/src/myapp -w /usr/src/myapp python:3 bash -c "pip install evalutils && evalutils init algorithm myproject"

Either of these commands will generate a folder called ``myproject``
with everything you need to get started.

It is a good idea to commit your project to git right now. You can do this with:

.. code-block:: console

    $ cd myproject
    $ git init
    $ git lfs install   (see the warning below)
    $ git add --all
    $ git commit -m "Initial Commit"

.. warning:: The test input images and the expected output will be stored in this repo,
    so remember to use a private repo if you're going to push this to github or gitlab,
    and use `git lfs`_ if your ground truth data are large.

    The .gitattributes file at the root of the repository specifies all the files which should be
    tracked by git-lfs. By default all files in the test directories
    are configured to be tracked by git-lfs, but they will only be registered
    once the `git lfs`_ extension is installed on your system and the :console:`git lfs install`
    command has been issued inside the generated repository.


The structure of the project will be:

.. code-block:: console

    .
    └── myproject
        ├── build.sh                 # Builds your algorithm container
        ├── Dockerfile               # Defines how to build your algorithm container
        ├── export.sh                # Exports your algorithm container to a .tar file for use on grand-challenge.org
        ├── .gitattributes           # Define which files git should put under git-lfs
        ├── .github/workflows/ci.yml # Contains a CI configuration file for github workflows for your project
        ├── .gitignore               # Define which files git should ignore
        ├── process.py               # Contains your algorithm code - this is where you will extend the BaseAlgorithm class
        ├── README.md                # For describing your algorithm to others
        ├── requirements.txt         # The python dependencies of your algorithm container - add any new dependencies here
        ├── test                     # A folder that contains an example test image for testing
        │   ├── 1.0.000.000000*.mhd  # An example test image metaio header file
        │   ├── 1.0.000.000000*.zraw # An example test image metaio data file
        │   └── expected_output.json # Output file expected to be produced by the algorithm container
        └── test.sh                  # A script that runs your algorithm container using the example test image and validates the output

The most important file is ``process.py``.
This is the file where you will extend the ``Algorithm`` class and implement your algorithm.
In this file, a new class has been created for you, and it is instantiated and run with:

.. code-block:: python

    if __name__ == "__main__":
        Myproject().process()


This is all that is needed for ``evalutils`` to run the algorithm and process input images.
The subclass of ``Algorithm`` is what you need to modify for your specific algorithms.

By default all algorithms will try to load all files in the /input directory using ``SimpleITKLoader`` for loading the data.
After successfully loading a single image a ``SimpleITK.Image`` object is ready to be manipulated by the algorithms in
the ``predict`` method for algorithm tasks like classification, segmentation, or detection.


Classification Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~

The boilerplate for classification algorithms looks like this:

.. code-block:: python

    class Myproject(ClassificationAlgorithm):
        def __init__(self):
            super().__init__(
                validators=dict(
                    input_image=(
                        UniqueImagesValidator(),
                        UniquePathIndicesValidator(),
                    )
                ),

        def predict(self, *, input_image: SimpleITK.Image) -> Dict:
            # Checks if there are any nodules voxels (> 1) in the input image
            return {
                "values_exceeding_one": bool(np.any(SimpleITK.GetArrayFromImage(input_image) > 1))
            }


The input images can be validated by specifying validators. Here, we want to validate that the input images are unique,
so we use the ``UniqueImagesValidator`` and ``UniquePathIndicesValidator``.
See :mod:`evalutils.validators` for a list of other validators that you can use.

All that needs to be modified is the ``predict`` method of your Algorithm class. By default, it takes an input ``SimpleITK.Image``
and expects a dictionary that contains some values based on the image.
In this example it takes the input image, converts it to a ``numpy.ndarray``, checks if there is any value larger than 1 in the image, and writes the boolean result to a dictionary.

The output dictionary can have an arbitrary number of key/value pairs. Extending the outputs can be done by adding new dictionary keys and associated values.

The resulting dictionary will be written to ``/output/results.json`` output file by the super class after running the ``predict`` method on all inputs.

Segmentation Algorithm
~~~~~~~~~~~~~~~~~~~~~~

For segmentation tasks, the generated code will look like this:

.. code-block:: python

    class Myproject(SegmentationAlgorithm):
        def __init__(self):
            super().__init__(
                validators=dict(
                    input_image=(
                        UniqueImagesValidator(),
                        UniquePathIndicesValidator(),
                    )
                ),

        def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
            # Segment all values greater than 2 in the input image
            return SimpleITK.BinaryThreshold(
                image1=input_image, lowerThreshold=2, insideValue=1, outsideValue=0
            )


Similar as before, all that needs to be modified is the ``predict`` method of your Algorithm class. By default, it takes an input ``SimpleITK.Image``
and expects another ``SimpleITK.Image`` as an output.

In this example it takes the input image, thresholds this for all values greater than or equal to 2 and returns the resulting image.

Besides the default ``/output/results.json`` output file the SegmentationAlgorithm outputs the resulting images at: ``/output/images/``.


Detection Algorithm
~~~~~~~~~~~~~~~~~~~

The generated boilerplate for detection tasks is:

.. code-block:: python

    class Myproject(DetectionAlgorithm):
        def __init__(self):
            super().__init__(
                validators=dict(
                    input_image=(
                        UniqueImagesValidator(),
                        UniquePathIndicesValidator(),
                    )
                ),

        def predict(self, *, input_image: SimpleITK.Image) -> DataFrame:
            # Extract a numpy array with image data from the SimpleITK Image
            image_data = SimpleITK.GetArrayFromImage(input_image)

            # Detection: Compute connected components of the maximum values
            # in the input image and compute their center of mass
            sample_mask = image_data >= np.max(image_data)
            labels, num_labels = label(sample_mask)
            candidates = center_of_mass(
                input=sample_mask, labels=labels, index=np.arange(num_labels) + 1
            )

            # Scoring: Score each candidate cluster with the value at its center
            candidate_scores = [
                image_data[tuple(coord)]
                for coord in np.array(candidates).astype(np.uint16)
            ]

            # Serialize candidates and scores as a list of dictionary entries
            data = self._serialize_candidates(
                candidates=candidates,
                candidate_scores=candidate_scores,
                ref_image=input_image,
            )

            # Convert serialized candidates to a pandas.DataFrame
            return DataFrame(data)


The only function that needs to be implemented is ``predict``, which should extract a list of candidate points from the input image and
return the candidates with some associated scores or labels to a ``pandas.DataFrame``.

The ``_serialize_candidates`` helper function takes in a list of candidates in image coordinate space and converts these to world coordinates given a reference ``SimpleItk.Image``.
Additionally, the function adds scores per candidate.

The resulting DataFrame is added to the ``/output/results.json``.

Add The Test Data And The Expected Output File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The next step is to add your test data (an example image and a json file with the expected output) to the repo.
The test images go into the ``test`` folder in your repo.
To update the expected output simply update the ``test/expected_output.json`` file.

Adapt The Algorithm
^^^^^^^^^^^^^^^^^^^

Change the function in the boilerplate to fit your needs, refer to the superclass methods for more information on return types.
See :class:`evalutils.BaseAlgorithm` for more possibilities.

Build, Test and Export
^^^^^^^^^^^^^^^^^^^^^^

When you're ready to test your algorithm you can simply invoke

.. code-block:: console

    $ ./test.sh

This will build your docker container, add the test data as a temporary volume, run the algorithm, ``cat /output/results.json``,
and test if the ``/output/results.json`` matches ``expected_output.json`` in the test folder of ``myproject``.
If the output looks ok and prints ``Tests successfully passed...``, then you're ready to go.

You can export the algorithm container with

.. code-block:: console

    $ ./export.sh

which will create ``myproject.tar.gz`` in the folder.
You can then upload this directly to `Grand Challenge`_ on your algorithms page.



.. _`Grand Challenge`: https://grand-challenge.org
.. _docker: https://www.docker.com/
.. _`git lfs`: https://git-lfs.github.com/
.. _`CookieCutter`: https://github.com/audreyr/cookiecutter
