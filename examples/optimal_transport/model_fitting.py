"""
Optimization routines
============================================
"""

import os
import torch
import matplotlib
import matplotlib.pyplot as plt
from math import isnan
import numpy as np
from scipy.optimize import minimize

import warnings

warnings.filterwarnings(
    "ignore", ".*GUI is implemented.*"
)  # annoying warning with pyplot and pause...


def mypause(interval):
    """Pause matplotlib without stealing focus."""
    backend = plt.rcParams["backend"]
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


def model_to_numpy(model, grad=False):
    """
    The fortran routines used by scipy.optimize expect float64 vectors
    instead of the gpu-friendly float32 matrices: we need conversion routines.
    """
    if not all(param.is_contiguous() for param in model.parameters()):
        raise ValueError(
            "Scipy optimization routines are only compatible with parameters given as *contiguous* tensors."
        )

    if grad:
        tensors = [
            param.grad.data.view(-1).cpu().numpy() for param in model.parameters()
        ]
    else:
        tensors = [param.data.view(-1).cpu().numpy() for param in model.parameters()]
    return np.ascontiguousarray(np.hstack(tensors), dtype="float64")


def numpy_to_model(model, vec):
    i = 0
    for param in model.parameters():
        offset = param.numel()
        param.data = (
            torch.from_numpy(vec[i : i + offset])
            .view(param.data.size())
            .type(param.data.type())
        )
        i += offset

    if i != len(vec):
        raise ValueError(
            "The total number of variables in model is not the same as in 'vec'."
        )


def fit_model(
    Model,
    method="L-BFGS",
    tol=1e-10,
    nits=500,
    nlogs=10,
    lr=0.1,
    eps=0.01,
    maxcor=10,
    gtol=1e-10,
    display=False,
    **params
):
    """"""

    # Load parameters =====================================================================================================

    # We'll minimize the model's cost
    # with respect to the model's parameters using a standard gradient-like
    # descent scheme. As we do not perform any kind of line search,
    # this algorithm may diverge if the learning rate is too large !
    # For robust optimization routines, you may consider using
    # the scipy.optimize API with a "parameters <-> float64 vector" wrapper.
    use_scipy = False
    if method == "Adam":
        optimizer = torch.optim.Adam(Model.parameters(), lr=lr, eps=eps)
    elif method == "L-BFGS":
        optimizer = torch.optim.SGD(
            Model.parameters(), lr=1.0
        )  # We'll just use its "zero_grad" method...

        use_scipy = True
        method = "L-BFGS-B"
        options = dict(
            maxiter=nits,
            ftol=tol,  # Don't bother fitting the shapes to float precision
            gtol=gtol,
            maxcor=maxcor,  # Number of previous gradients used to approximate the Hessian
        )
    else:
        raise NotImplementedError(
            'Optimization method not supported : "' + method + '". '
            'Available values are "Adam" and "L-BFGS".'
        )

    costs = []
    # Define the "closures" associated to our model =======================================================================

    fit_model.nit = -1
    fit_model.breakloop = False

    def closure(final_it=False):
        """
        Encapsulates a problem + display iteration into a single callable statement.
        This wrapper is needed if you choose to use LBFGS-like algorithms, which
        (should) implement a careful line search along the gradient's direction.
        """
        fit_model.nit += 1
        it = fit_model.nit
        # Minimization loop --------------------------------------------------------------------
        optimizer.zero_grad()  # Reset the gradients (PyTorch syntax...).
        cost = Model.forward()
        costs.append(cost.item())  # Store the "cost" for plotting.
        cost.backward()  # Backpropagate to compute the gradient.
        # Break the loop if the cost's variation is below the tolerance param:
        if (
            len(costs) > 1 and abs(costs[-1] - costs[-2]) < tol
        ) or fit_model.nit == nits - 1:
            fit_model.breakloop = True

        if display:
            Model.plot(nit=fit_model.nit, cost=cost.item())
            # print("{}: {:2.4f}".format(fit_model.nit, cost.item()))
        return cost

    # Scipy-friendly wrapper ------------------------------------------------------------------------------------------------
    def numpy_closure(vec, final_it=False):
        """
        Wraps the PyTorch closure into a 'float64'-vector routine,
        as expected by scipy.optimize.
        """
        vec = lr * vec.astype(
            "float64"
        )  # scale the vector, and make sure it's given as float64
        numpy_to_model(Model, vec)  # load this info into Model's parameters
        c = closure(
            final_it
        ).item()  # compute the cost and accumulate the gradients wrt. the parameters
        dvec_c = lr * model_to_numpy(
            Model, grad=True
        )  # -> return this gradient, as a properly rescaled numpy vector
        return (c, dvec_c)

    # Actual minimization loop ===============================================================================================
    if use_scipy:
        res = minimize(
            numpy_closure,  # function to minimize
            model_to_numpy(Model),  # starting estimate
            method=method,
            jac=True,  # matching_problems also returns the gradient
            options=options,
        )
        numpy_closure(res.x, final_it=True)
        # print(res.message)
    else:
        for i in range(nits + 1):  # Fixed number of iterations
            optimizer.step(closure)  # "Gradient descent" step.
            if fit_model.breakloop:
                closure(final_it=True)
                break
