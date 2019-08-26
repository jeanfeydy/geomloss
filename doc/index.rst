Geometric Loss functions between sampled measures, images and volumes
---------------------------------------------------------------------

**N.B.:** This is still an **alpha** release! 
Please send me your feedback: I will polish the user interface,
implement Hausdorff divergences, add support for images/volumes
and clean the documentation over the summer of 2019.
Hopefully, a first stable release should be ready for
`Miccai 2019 <https://www.miccai2019.org/>`_, in October.

The **GeomLoss** library provides efficient GPU implementations for:

- `Kernel norms <https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space>`_ 
  (also known as `Maximum Mean Discrepancies <http://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf>`_).
- `Hausdorff divergences <https://hal.archives-ouvertes.fr/hal-01827184v2>`_, which are
  positive definite generalizations of the
  `ICP <https://en.wikipedia.org/wiki/Iterative_closest_point>`_ loss,
  analogous to **log-likelihoods** of Gaussian Mixture Models.
- `Unbiased Sinkhorn divergences <https://arxiv.org/abs/1810.08278>`_, which are
  cheap yet **positive definite** approximations of 
  `Optimal Transport <https://arxiv.org/abs/1803.00567>`_ 
  (`Wasserstein <https://en.wikipedia.org/wiki/Wasserstein_metric>`_) costs.


These loss functions, defined between positive measures, 
are available through the custom `PyTorch <https://pytorch.org/>`_ layers 
:class:`SamplesLoss <geomloss.SamplesLoss>`, 
:class:`ImagesLoss <geomloss.ImagesLoss>` and 
:class:`VolumesLoss <geomloss.VolumesLoss>`
which allow you to work with weighted **point clouds** (of any dimension),
**density maps** and **volumetric segmentation masks**.
Geometric losses come with three backends each:

- A simple ``tensorized`` implementation, for **small problems** (< 5,000 samples).
- A reference ``online`` implementation, with a **linear** (instead of quadratic)
  **memory footprint**, that can be used for finely sampled measures.
- A very fast ``multiscale`` code, which uses an
  **octree**-like structure for large-scale problems in dimension <= 3.


A typical sample of code looks like:

.. code-block:: python

    import torch
    from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss

    # Create some large point clouds in 3D
    x = torch.randn(100000, 3, requires_grad=True).cuda()
    y = torch.randn(200000, 3).cuda()

    # Define a Sinkhorn (~Wasserstein) loss between sampled measures
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

    L = loss(x, y)  # By default, use constant weights = 1/number of samples
    g_x, = torch.autograd.grad(L, [x])  # GeomLoss fully supports autograd!


GeomLoss is a simple interface for cutting-edge Optimal Transport
algorithms. It provides:

* Support for **batchwise** computations.
* **Linear** (instead of quadratic) **memory footprint** for large problems,
  relying on the `KeOps library <https://www.kernel-operations.io>`_
  for map-reduce operations on the GPU.
* Fast **kernel truncation** for small bandwidths, using an octree-based structure. 
* Log-domain stabilization of the Sinkhorn iterations,
  eliminating numeric **overflows** for small values of :math:`\varepsilon`.
* Efficient computation of the **gradients**, which bypasses the naive
  backpropagation algorithm.
* Support for `unbalanced  <https://link.springer.com/article/10.1007/s00222-017-0759-8>`_ 
  Optimal `Transport <https://arxiv.org/pdf/1506.06430.pdf>`_,
  with a softening of the marginal constraints
  through a maximum **reach** parameter.
* Support for the `ε-scaling heuristic <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.228.9750&rep=rep1&type=pdf>`_ 
  in the Sinkhorn loop, with `kernel truncation <https://arxiv.org/abs/1610.06519>`_
  in dimensions 1, 2 and 3.
  On typical 3D problems, our implementation is **50-100 times faster** than
  the standard `SoftAssign/Sinkhorn algorithm <https://arxiv.org/abs/1306.0895>`_.


Note, however, that :class:`SamplesLoss <geomloss.SamplesLoss>` does *not* implement the 
`Fast Multipole <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.129.7826&rep=rep1&type=pdf>`_ 
or `Fast Gauss <http://users.umiacs.umd.edu/~morariu/figtree/>`_ transforms.
If you are aware of a well-packaged implementation
of these algorithms on the GPU, please contact me!


The divergences implemented here are
all **symmetric**, **positive definite**
and therefore suitable for measure-fitting applications.
For positive input measures :math:`\alpha` and :math:`\beta`,
our :math:`\text{Loss}` functions are such that

.. math::
    \text{Loss}(\alpha,\beta) ~&=~ \text{Loss}(\beta,\alpha), \\
    0~=~\text{Loss}(\alpha,\alpha) ~&\leqslant~ \text{Loss}(\alpha,\beta), \\
    0~=~\text{Loss}(\alpha,\beta)~&\Longleftrightarrow~ \alpha = \beta.

**GeomLoss** can be used in a wide variety of settings, 
from **shape analysis** (LDDMM, optimal transport...)
to **machine learning** (kernel methods, GANs...)
and **image processing**.
Details and examples are provided below:

* :doc:`Maths and algorithms <api/geomloss>`
* :doc:`PyTorch API <api/pytorch-api>`
* `Source code <https://github.com/jeanfeydy/geomloss>`_
* :doc:`Examples <_auto_examples/index>`

**GeomLoss is licensed** under the `MIT license <https://github.com/jeanfeydy/geomloss/blob/master/LICENSE>`_.

Author and Contributors
-------------------------

Feel free to contact us for any **bug report** or **feature request**:

- `Jean Feydy <http://www.math.ens.fr/~feydy/>`_
- `Pierre Roussillon <https://proussillon.gitlab.io/en/>`_ (extensions to brain tractograms and normal cycles)

Licensing, academic use
---------------------------

This library is licensed under the permissive `MIT license <https://en.wikipedia.org/wiki/MIT_License>`_,
which is fully compatible with both **academic** and **commercial** applications.
If you use this code in a research paper, **please cite**:

::

    @inproceedings{feydy2019interpolating,
        title={Interpolating between Optimal Transport and MMD using Sinkhorn Divergences},
        author={Feydy, Jean and S{\'e}journ{\'e}, Thibault and Vialard, Fran{\c{c}}ois-Xavier and Amari, Shun-ichi and Trouve, Alain and Peyr{\'e}, Gabriel},
        booktitle={The 22nd International Conference on Artificial Intelligence and Statistics},
        pages={2681--2690},
        year={2019}
    }


Related projects
------------------

You may be interested by:

- The `KeOps library <http://www.kernel-operations.io/>`_, which provides 
  **efficient CUDA routines**
  for point cloud processing, with full `PyTorch <https://pytorch.org/>`_ support.
- Rémi Flamary and Nicolas Courty's 
  `Python Optimal Transport library <https://pot.readthedocs.io/en/stable/>`_,
  which provides a reference implementation of **OT-related methods** for small problems.
- Bernhard Schmitzer's `Optimal Transport toolbox <https://github.com/bernhard-schmitzer/optimal-transport/tree/master/v0.2.0>`_,
  which provides a reference **multiscale solver** for the OT problem, on the CPU.

Table of contents
-----------------

.. toctree::
   :maxdepth: 2

   api/install
   api/geomloss
   api/pytorch-api
   _auto_examples/index


