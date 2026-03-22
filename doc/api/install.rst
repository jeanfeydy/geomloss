Getting started
=================

Install with pip (recommended)
---------------------------------

To install GeomLoss and all its (optional) dependencies, please:

  1. Install `PyTorch <https://pytorch.org/>`_.
  2. Install the `KeOps library <http://www.kernel-operations.io/keops/python/installation.html>`_.
  3. Install GeomLoss with::
    
      pip install geomloss

On `Google Colab <https://colab.research.google.com/>`_,
simply typing::

  !pip install geomloss[full]

should allow you to get a working setup in less than twenty seconds.

Install with git
-------------------


The simplest way of installing a specific version
of GeomLoss is to use `some advanced pip syntax <https://pip.pypa.io/en/stable/reference/pip_install/#git>`_:


.. code-block:: bash

    pip install git+https://github.com/jeanfeydy/geomloss.git@main#egg=project[full]



Alternatively, you may:

  1. Install `PyTorch <https://pytorch.org/>`_.
  2. Install the `KeOps library <http://www.kernel-operations.io/keops/python/installation.html>`_.
  3. Clone the GeomLoss repository::
    
        git clone https://github.com/jeanfeydy/geomloss.git

  4. Add ``/path/to/geomloss`` to your ``$PYTHONPATH``.




Build the documentation on Google Cloud
-----------------------------------------

To generate this website on a fresh `Google Compute session <https://cloud.google.com/compute>`_, first install CUDA and nvcc by following
`these instructions <https://askubuntu.com/questions/1077061/how-do-i-install-nvidia-and-cuda-drivers-into-ubuntu>`_.
You should then type the following lines in the SSH prompt::

    # Install pip and the necessary packages to run all tutorials:
    sudo apt install python3-pip
    pip3 install numpy GPUtil cmake ninja sklearn scipy imageio scikit-image
    pip3 install torch torchvision
    pip3 install dipy nibabel vtk pyvista SimpleITK
    pip3 install sphinx sphinx-gallery recommonmark sphinxcontrib-httpdomain sphinx_rtd_theme plyfile >> install.log
    
    # Clone the latest versions of keops and geomloss, adding them to the path
    git clone --quiet --recursive https://github.com/getkeops/keops.git keops/  >> install.log
    git clone --quiet  https://github.com/jeanfeydy/geomloss.git geomloss/  >> install.log
    echo "export PYTHONPATH=$PYTHONPATH:~/keops/:~/geomloss/" >> ~/.bashrc
    
    # Launch a "detachable" session:
    tmux
    cd geomloss/doc
    
    # Run the doc once to compile all the KeOps routines:
    make html
    # Clean up everything...
    make clean
    # And re-run all scripts to get the correct timings:
    make html

    # Zip everything into an archive:
    sudo apt install zip
    zip -r geomloss_documentation.zip _build/

    # On the *local* machine: download the zipped documentation.
    # Typically, with a properly configured "gcloud" command:
    gcloud compute scp root@nvidia-gpu-cloud-pytorch-image-1-vm:/home/jean_feydy/geomloss/doc/geomloss_documentation.zip ~



Build the documentation on Google Colab
-----------------------------------------

Alternatively, you may also generate this website on a `free Google Colab session <https://colab.research.google.com/>`_.
To reproduce our results (with longer runtimes), please create
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
    !pip install sphinx-gallery recommonmark sphinxcontrib-httpdomain sphinx_rtd_theme plyfile >> install.log

    # Download KeOps...
    !pip install pykeops  >> install.log

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
