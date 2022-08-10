from geomloss import ot
from .check_ot_result import check_ot_result


def test_correct_value(dim, batchsize, reg):

    test_case = ot.tests.diracs_matrix(dim=dim, batchsize=batchsize)
    
    us = ot.solve(test_case["cost"], reg=reg, rtol=1e-2)
    gt = test_case["result"]

    check_ot_result(us, gt)