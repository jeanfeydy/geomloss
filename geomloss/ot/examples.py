# Example of code for the new API

from geomloss import ot

# exact ot
P = ot.solve(C, a, b).plan
wass = ot.solve(C, a, b).value

# Sinkhorn OT with uniform weights ("Robust point matching"):
P = ot.solve(C, reg=1).plan

# Vanilla Sinkhorn à la Cuturi:
# (N.B.: we use "entropy" instead of "relative entropy" = "KL".)
OT_reg = ot.solve(C, reg=1, reg_type="entropy").value

# "Sharp" Sinkhorn à la Luise:
SOT_reg = ot.solve(C, reg=1).value_linear


# Debiased unbalanced OT:
S_reg_unb = ot.solve(C, reg=1, unbalanced=10).value


# Compute a subset of a lazy, entropy-regularized and unbalanced transport plan
ot.solve_samples(xa, xb, blur=0.1, reach=5).lazy_plan[0:5, 1:3]


# Compute Sinkhorn divergence between empirical distributions
ot.solve_samples(xa, xb, reg=0.1, debias=true).value
