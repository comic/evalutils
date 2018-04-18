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

Generate the project structure
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
        ├── build.sh
        ├── Dockerfile
        ├── evaluation.py
        ├── export.sh
        ├── ground-truth
        │   └── reference.csv
        ├── README.md
        ├── requirements.txt
        ├── test
        │   └── submission.csv
        └── test.sh

.. _`Grand Challenge`: https://grand-challenge.org
.. _docker: https://www.docker.com/
.. _`git lfs`: https://git-lfs.github.com/
