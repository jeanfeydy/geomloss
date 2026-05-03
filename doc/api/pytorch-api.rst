API
===

:mod:`geomloss` - :doc:`Geometric Loss functions <geomloss>`, 
with full support of PyTorch's :mod:`autograd` engine:

.. currentmodule:: geomloss

The :class:`SamplesLoss` class implements a collection of geometric loss functions between point clouds, including the Sinkhorn divergence and kernel MMDs. It can be used as a loss function in a PyTorch training loop.

.. autosummary:: 

    SamplesLoss

The :mod:`geomloss.ot` submodule, introduced in 2026, now provides a modern interface to
our scalable optimal transport solvers. It is compatible with the
`Python Optimal Transport library <https://pythonot.github.io/>`_ and provides
a convenient access to optimal transport plans and potentials.

.. warning::
   The API of :mod:`geomloss.ot` is still in development and may change in future releases.
   We welcome any feedback and suggestions for improvement.

The :func:`ot.solve` and :func:`ot.solve_batch` functions handle user-defined cost matrices :math:`C_{ij}`.
They return an :class:`ot.OTResultMatrix` object which stores the optimal transport plan,
potentials, and other useful quantities.

.. autosummary:: 

    ot.solve
    ot.solve_batch
    ot.OTResultMatrix

The :func:`ot.solve_sample` handles the case where the cost matrix :math:`C_{ij}` 
derives from a cost function :math:`c(x_i, y_j)` that is evaluated on
points :math:`x_i` and :math:`y_j` in some vector space -- for instance,
the squared Euclidean distance :math:`c(x_i, y_j) = \|x_i-y_j\|^2`.
It is most similar to :class:`SamplesLoss` and returns an :class:`ot.OTResultSample` object.

.. autosummary:: 

    ot.solve_sample
    ot.OTResultSample


.. note::
    
    We will soon add support for measures defined on regular grids, such as 2D and 3D density images,
    as well as Wasserstein barycenters and solutions
    of the Gromov-Wasserstein problem.


.. autoclass:: geomloss.SamplesLoss

.. autofunction:: geomloss.ot.solve
.. autofunction:: geomloss.ot.solve_batch

.. autoclass:: geomloss.ot.OTResultMatrix
   :members:
   :undoc-members:

.. autofunction:: geomloss.ot.solve_sample

.. autoclass:: geomloss.ot.OTResultSample
   :members:
   :undoc-members:
