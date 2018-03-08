"""Tests for `bcc` package."""

import pytest
from bcc import *

# AAAAAAAAAAAAAHHHHHHHH!!!!!!!!!!!!!!
# import sys
# sys.path.insert(0, '/home/sheffler/rifsrc/buildR/lib.linux-x86_64-3.6')

try:
    from rif.numeric.lattice import *
    import rif
    import rif.util
    HAVE_RIF = True
except ImportError:
    HAVE_RIF = False

only_if_have_rif = pytest.mark.skipif('not HAVE_RIF')


def test_BCC_traits():
    with pytest.raises(tl.TraitError):
        BCC().sizes
    with pytest.raises(tl.TraitError):
        BCC(sizes=[1, 2, 3], lower=[1, 2]).width
    with pytest.raises(tl.TraitError):
        BCC(sizes=[1, 2, 3], upper=[1, 2]).width
    with pytest.raises(tl.TraitError):
        BCC(sizes=[[1, 2], [3, 4]]).upper
    assert np.allclose(BCC(sizes=(3, 4, 5)).lower, [0, 0, 0])
    assert np.allclose(BCC(sizes=(3, 4, 5)).upper, [1, 1, 1])
    assert np.allclose(BCC(sizes=(2, 4, 8)).width, [0.5, 0.25, 0.125])


@only_if_have_rif
def test_BCC_against_rifcpp():
    sizes = [3, 4, 7, 3, 2, 8]
    nside_points = 8
    binner = BCC(sizes=sizes, upper=sizes)
    linsp = [np.linspace(-1, sizes[i] + 1, nside_points)
             for i in range(len(sizes))]
    test_points = np.stack(np.meshgrid(*linsp), axis=-1)

    oracle = BCC6(sizes, lb=[0] * len(sizes), ub=sizes)
    i0 = oracle.index(test_points)
    i1 = binner.get_bin_index(test_points)
    assert np.all(i0 == i1)

    test_indices = np.arange(len(binner))
    c0 = oracle.center(test_indices)['raw']
    c1 = binner.get_bin_center(test_indices)
    # for i in range(10):
    # print(binner.get_bin_center(i))
    # print(c0[i])
    # print()
    assert np.allclose(c0, c1)
