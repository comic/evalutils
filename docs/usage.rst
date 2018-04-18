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

This will generate a folder called ``myproject`` with everything you need to
get started. If you do not have a local python 3.6+ environment you can also
do this with docker

.. code-block:: console

    $ docker run -it --rm -v $(pwd):/usr/src/myapp -w /usr/src/myapp python:3 bash -c "pip install evalutils && evalutils init myproject"

.. _`Grand Challenge`: https://grand-challenge.org
.. _docker: https://www.docker.com/
