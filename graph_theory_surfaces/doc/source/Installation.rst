=========================
Installation Instructions
=========================

The first step will be to obtain the code either via pip or directly on gitlab.

*********************
Pip (Not Working Yet)
*********************

.. code-block:: bash

    pip install surfgraph

******
Gitlab
******

.. code-block:: bash

    git clone git@gitlab.com:jgreeley-group/graph-theory-surfaces.git

*****************
Post-Installation
*****************

Next you should ensure the python module is importable and that you can find the executable scripts(example: compare-chem-env).  If you used pip to install this should already be done, but if installing from source you will need to add the "surfgraph" directory to PYTHONPATH and the "bin" directory to PATH.

This is also a good time to ensure the tests pass correctly.  The tests should be able to be run using the following command.  

**TODO: This is currently not implemented.**

.. code-block:: bash

    python -m surfgraph
