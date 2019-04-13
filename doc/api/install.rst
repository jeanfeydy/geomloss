Getting started
=================

Install with pip (recommended)
---------------------------------

To install GeomLoss and all its (optional) dependencies, please:

  1. Install the CUDA toolkit, including the **nvcc** compiler.
  2. Install `PyTorch <https://pytorch.org/>`_.
  3. Install the `KeOps library <http://www.kernel-operations.io/keops/python/installation.html>`_.
  4. Install GeomLoss with::
    
      pip install geomloss

On `Google Colab <https://colab.research.google.com/>`_,
simply typing::

  !pip install geomloss[full]

should allow you to get a working setup in less than twenty seconds.

Install with git
-------------------

Alternatively, you may:

  1. Install the CUDA toolkit, including the **nvcc** compiler.
  2. Install `PyTorch <https://pytorch.org/>`_.
  3. Install the `KeOps library <http://www.kernel-operations.io/keops/python/installation.html>`_.
  4. Clone the GeomLoss repository::
    
        git clone https://github.com/jeanfeydy/geomloss.git

  5. Add ``/path/to/geomloss`` to your ``$PYTHONPATH``.



Build the documentation
--------------------------

This website was generated on a `free Google Colab session <https://colab.research.google.com/>`_.
To reproduce our results and benchmarks, feel free to create
a new Colab notebook and to type the following instructions in the first few cells::

    # Mount Google drive to save the final documentation
    from google.colab import drive
    drive.mount('/content/gdrive')

    # Make sure that GDrive is mounted correctly
    !ls gdrive/'My Drive'

    # Install sphinx for Python 3 (instead of the default Python 2)
    !pip uninstall sphinx
    !pip3 install sphinx

    # Install the dependencies for sphinx and KeOps
    !pip install numpy GPUtil cmake ninja > install.log
    !pip install sphinx-gallery recommonmark sphinxcontrib-httpdomain sphinx_rtd_theme  >> install.log

    # Download KeOps...
    !pip install pykeops[full]  >> install.log

    # Download GeomLoss
    !git clone https://github.com/jeanfeydy/geomloss.git  >> install.log

    # Make sure that new scripts will have access to GeomLoss
    import os
    os.environ['PYTHONPATH'] += ":/content/geomloss/"
    !echo $PYTHONPATH

    # Put GeomLoss in the current environment
    import sys
    sys.path.append('/content/geomloss/')

    # First run, to compile everything
    %cd /content/geomloss/doc
    !make html

    # Second run, to get the correct timings without compile times
    !make clean
    !make html

    # Now, just download "documentation.zip" and upload it on the website :-)
    !zip -r geomloss_documentation.zip _build
    !cp geomloss_documentation.zip /content/gdrive/'My Drive'

That's it!
