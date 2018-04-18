Welcome to evalutils's documentation!
======================================

The automated evaluation system on `Grand Challenge`_ will run an instance of
your evaluation docker container image on each new submission.
The users submission will be extracted on ``/input/``, and your container
is expected to calculate all of the metrics for this submission, and write them
to ``/output/metrics.json``.
If the metrics cannot be calculated, the container should write to ``stderr``
and the last line of this will be passed to the user so that they can debug
their submission.

`evalutils`_ helps you do this by providing a package that helps you create
a project structure, load and validate the submissions, score the submission,
write the json file, and package this in a docker container image so that
you can then upload it to `Grand Challenge`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   readme
   installation
   usage
   modules
   contributing
   authors
   history

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _evalutils: https://github.com/comic/evalutils
.. _`Grand Challenge`: https://grand-challenge.org
