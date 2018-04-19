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

        @staticmethod
        def score_case(*, idx: int, case: Series) -> Dict:
            return {
                "accuracy": 1.0 if case["class_ground_truth"] == case[
                    "class_prediction"] else 0.0,
                "case_id": str(idx),
            }

        def save(self):
            # In this example we do not want to report the case wise accuracy
            # results to metrics.json as this can leak the ground truth data.
            # So, we remove it from the _case_results DataFrame
            del self._case_results["accuracy"]
            super().save()




.. _`Grand Challenge`: https://grand-challenge.org
.. _docker: https://www.docker.com/
.. _`git lfs`: https://git-lfs.github.com/
