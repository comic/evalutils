=====
Usage
=====

Getting Started
---------------

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

Once you have installed evalutils you can generate your project structure
by using the command line:

.. code-block:: console

    $ evalutils init myproject

If you do not have a local python 3.6+ environment you can also
do this with docker, and then change the permissions of the generated folder
to your own ``USER`` and ``GROUP`` afterwards:

.. code-block:: console

    $ docker run -it --rm -v $(pwd):/usr/src/myapp -w /usr/src/myapp python:3 bash -c "pip install evalutils && evalutils init myproject"
    $ sudo chown -R <USER>:<GROUP> myproject/

Either of these commands will generate a folder called ``myproject``
with everything you need to get started.

It is a good idea to commit your project to git right now. You can do this with:

.. code-block:: console

    $ cd myproject
    $ git init
    $ git add --all
    $ git commit -m "Initial Commit"

.. warning:: The test set ground truth will be stored in this repo,
    so remember to use a private repo if you're going to push this to github or gitlab,
    and use `git lfs`_ if your ground truth data are large.


The structure of the project will be:

.. code-block:: console

    .
    └── myproject
        ├── build.sh            # Builds your evaluation container
        ├── Dockerfile          # Defines how to build your evaluation container
        ├── evaluation.py       # Contains your evaluation code - this is where you will extend the Evaluation class
        ├── export.sh           # Exports your container to a .tar file for use on grand-challenge.org
        ├── ground-truth        # A folder that contains your ground truth annotations
        │   └── reference.csv   # In this example the ground truth is a csv file
        ├── README.md           # For describing your evaluation to others
        ├── requirements.txt    # The python dependencies of your evaluation container - add any new dependencies here
        ├── test                # A folder that contains an example submission for testing
        │   └── submission.csv  # In this example the participants will submit a csv file
        └── test.sh             # A script that runs your evaluation container on the test submission

The most important file is ``evaluation.py``.
This is the file where you will extend the ``Evaluation`` class and implement the evaluation for your challenge.
In this file, a new class has been created for you, and it is instantiated and run with:

.. code-block:: python

    if __name__ == "__main__":
        evaluation = Myproject()
        evaluation.evaluate()


This is all that is needed for ``evalutils`` to perform the evaluation and generate the output for each new submission.
The superclass of ``Evaluation`` is what you need to adapt to your specific challenge.

.. code-block:: python

    class Myproject(Evaluation):
        def __init__(self):
            super().__init__(
                file_loader=CSVLoader(),
                validators=(
                    ExpectedColumnNamesValidator(expected=("case", "class",)),
                ),
                join_key="case",
            )

        def score_aggregates(self):
            return {
                "accuracy_score": accuracy_score(
                    self._cases["class_ground_truth"], self._cases["class_prediction"]
                 ),
            }

In this case the evaluation is loading csv files, so uses an instance ``CSVLoader`` which will do the loading of the data.
In this example, both the ground truth and the prediction CSV files will contain the columns `case` (an index) and `class` (the predicted class of this case).
We want to validate that the correct columns appear in both the ground truth and submitted predictions, so we use the ``ExpectedColumnNamesValidator`` with the names of the columns we expect to find.
See :mod:`evalutils.validators` for a list of other validators that you can use.

The ground truth and predictions will be loaded into two DataFrames.
The last argument is a ``join_key``, the is the name of the column that will appear in both DataFrames that serves as an index to join the dataframes on in order to create ``self._cases``.
The ``join_key`` is manditory when you use a ``CSVLoader``.
This should be set to some sort of common index, such as a `case` identifier.
When loading in files they are first going to be sorted so you might not need a ``join_key``, but you could also write a function that matches the cases based on filename.

The last part is performing the actual evaluation.
In this example we are only getting one number per submission, the accuracy score.
This number is calculated using ``sklearn.metrics.accuracy_score``.
The ``self._cases`` data frame will contain all of the columns that you expect, and for those that have not been joined they will be available as ``"<column_name>_ground_truth"`` and ``"<column_name>_prediction"``.

Add The Ground Truth and Test Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The next step is to add the ground truth and test data (an example submission) to the repo.
If using CSV data simply update the ``ground-truth/reference.csv`` file, and then update the expected column names and join key in evaluate.py.
Otherwise, see :mod:`evalutils.io` for other loaders such as the ones for ITK files or images.
You can also add your own loader by extending the ``FileLoader`` class.

Adapt The Evaluation
^^^^^^^^^^^^^^^^^^^^

Change the function in ``score_aggregates`` to fit your needs.
The only requirement here is that it returns a dictionary.
If you need to score cases individually before aggregating them, you should remove the implementation of ``score_aggregates`` and implement ``score_case`` instead.
See :class:`evalutils.Evaluation` for more possibilities.

Build, Test and Deploy
^^^^^^^^^^^^^^^^^^^^^^

When you're ready to test your evaluation you can simply invoke

.. code-block:: console
    $ ./test.sh

This will build your docker container, add the test data as a temporary volume, run the evaluation, and then ``cat /output/metrics.json``.
If the output looks ok, then you're ready to go.

You can deploy the evaluation container with

.. code-block:: console
    $ ./deploy.sh

which will create myproject.tar in the folder.
You can then upload this directly to `Grand Challenge`_ on your evaluation methods page.

.. _`Grand Challenge`: https://grand-challenge.org
.. _docker: https://www.docker.com/
.. _`git lfs`: https://git-lfs.github.com/
