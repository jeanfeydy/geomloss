Getting started
=================

Install
----------

Using pip (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~

Soon!


Using git
~~~~~~~~~~~

The **multiscale** backend of :meth:`geomloss.SamplesLoss` relies on KeOps routines
which have not yet been merged into the **master** branch.
As of today, to install GeomLoss and all its optional dependencies, 
you'll thus have to:

  1. Install the CUDA toolkit, including the **nvcc** compiler.
  2. Install `PyTorch <https://pytorch.org/>`_.
  3. Install the `KeOps library <http://www.kernel-operations.io/python/installation.html>`_.
  4. Clone the GeomLoss repository::
    
        git clone https://github.com/jeanfeydy/geomloss.git

  5. Add ``/path/to/geomloss`` to your ``$PYTHONPATH``.



A simple option may be to use a
`free Google Colab session <https://colab.research.google.com/notebooks/welcome.ipynb#recent=true>`_,
as discussed below.


Build the documentation
--------------------------

This website was generated on a `free Google Colab session <https://colab.research.google.com/notebooks/welcome.ipynb#recent=true>`_.
To reproduce our results and benchmarks, feel free to create
a new Colab notebook and type the following instructions in the first few cells::

    # Mount Google drive to save the final documentation
    from google.colab import drive
    drive.mount('/content/gdrive')

    # Make sure that GDrive is mounted correctly
    !ls gdrive/'My Drive'

    # Install the dependencies for sphinx and KeOps
    !pip install numpy GPUtil cmake ninja > install.log
    !pip install sphinx-gallery recommonmark sphinxcontrib-httpdomain sphinx_rtd_theme  >> install.log

    # Download KeOps... And don't forget to update pybind11.
    !git clone --recursive --branch cluster_sparsity https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops.git keops/  >> install.log

    # Download GeomLoss
    !git clone https://github.com/jeanfeydy/geomloss.git  >> install.log

    # Make sure that new scripts will have access to KeOps
    import os
    os.environ['PYTHONPATH'] += ":/content/geomloss/:/content/keops/"
    !echo $PYTHONPATH

    # Put KeOps in the current environment
    import sys
    sys.path.append('/content/geomloss/')
    sys.path.append('/content/keops/')

    # Test that KeOps is properly installed
    import torch
    import pykeops.torch as pktorch

    x = torch.arange(1, 10, dtype=torch.float32).view(-1, 3)
    y = torch.arange(3, 9, dtype=torch.float32).view(-1, 3)

    my_conv = pktorch.Genred('SqNorm2(x-y)', ['x = Vx(3)', 'y = Vy(3)'])
    print(my_conv(x, y))


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