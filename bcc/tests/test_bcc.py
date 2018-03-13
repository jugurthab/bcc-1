"""Tests for `bcc` package."""

import pytest
from bcc import *
from hypothesis import given, settings
import hypothesis.strategies as hs

try:
    from rif.numeric.lattice import *
    import rif
    import rif.util
    only_if_have_rif = lambda x: x
    oracles = [None, None, None, BCC3, BCC4, BCC5,
               BCC6, BCC7, BCC8, BCC9, BCC10]
except ImportError:
    only_if_have_rif = pytest.mark.skip


max_examples = 100


def len_or_m_one(x):
    try:
        return len(x)
    except TypeError:
        return -1


def point_grid(sizes, lb, ub):
    n = max(len_or_m_one(x) for x in (lb, ub, sizes))
    lb = [lb] * n if len_or_m_one(lb) is -1 else lb
    ub = [ub] * n if len_or_m_one(ub) is -1 else ub
    sizes = [sizes] * n if len_or_m_one(sizes) is -1 else sizes
    linsp = [np.linspace(lb[i], ub[i], sizes[i])
             for i in range(len(lb))]
    return np.stack(np.meshgrid(*linsp), axis=-1).reshape(-1, len(lb))


def index_samples(mx, nsamp):
    if mx < nsamp:
        return np.arange(mx)
    else:
        return np.random.randint(0, mx, nsamp)


@hs.composite
def sizes_lb_ub(draw, min_ndim=3, max_ndim=10):
    ndim = draw(hs.integers(min_value=min_ndim, max_value=max_ndim))
    sizes = draw(hs.lists(hs.integers(min_value=2, max_value=10),
                          min_size=ndim, max_size=ndim))
    lb0 = draw(hs.lists(hs.floats(min_value=-10, max_value=10,
                                  allow_nan=False, allow_infinity=False),
                        min_size=ndim, max_size=ndim))
    ub0 = draw(hs.lists(hs.floats(min_value=-10, max_value=10,
                                  allow_nan=False, allow_infinity=False),
                        min_size=ndim, max_size=ndim))
    return sizes, list(np.minimum(lb0, ub0)), list(np.maximum(ub0, ub0) + 1)


def test_BCC_traits():
    with pytest.raises(tl.TraitError):
        BCC().sizes
    with pytest.raises(tl.TraitError):
        BCC(sizes=[1, 2, 3], lower=[1, 2]).width
    with pytest.raises(tl.TraitError):
        BCC(sizes=[1, 2, 3], upper=[1, 2]).width
    with pytest.raises(tl.TraitError):
        BCC(sizes=[[1, 2], [3, 4]]).upper
    with pytest.raises(tl.TraitError):
        BCC(sizes=[1, 2],).upper
    with pytest.raises(tl.TraitError):
        BCC(sizes=[1, 2], lower=[1, 1], upper=[0, 2])
    assert np.allclose(BCC(sizes=(3, 4, 5)).lower, [0, 0, 0])
    assert np.allclose(BCC(sizes=(3, 4, 5)).upper, [1, 1, 1])
    assert np.allclose(BCC(sizes=(2, 4, 8)).width, [0.5, 0.25, 0.125])


def _test_BCC_against_rifcpp(inp):
    sizes, lb, ub = inp

    binr = BCC(sizes=sizes, lower=lb, upper=ub)
    oracle = oracles[len(sizes)](sizes, lb, ub)

    nside_points = (10000**(1 / len(sizes)))
    linsp = [np.linspace(lb[i], ub[i], nside_points)
             for i in range(len(sizes))]
    test_points = np.stack(np.meshgrid(*linsp), axis=-1)
    i0 = oracle.index(test_points)
    i1 = binr.get_bin_index(test_points)
    # print(i0[i0 != i1])
    # print(i1[i0 != i1])
    assert np.all(i0 == i1)

    test_indices = index_samples(len(binr), 10000)
    c0 = oracle.center(test_indices)['raw']
    c1 = binr.get_bin_center(test_indices)
    assert np.allclose(c0, c1)


@only_if_have_rif
def test_BCC_against_rifcpp_case0():
    sizes = [2, 2, 2, 2,  # 3,
             3, 3, 3]
    lb = [0.0, 0.0, 0.0, 0.0,  # 0.0,
          # 0.4687500000000000,  # works
          # 0.4687500000000001,  # fails
          0.46876,  # fails
          # 0.468751,  # works
          0.0,
          0.012699127197266515]
    ub = [1.0, 1.0, 1.0, 1.0,  # 1.0,
          1.46875, 1.0, 7.987300872802734]

    print(np.array(ub) - np.array(lb))
    # assert 0

    binr = BCC(sizes=sizes, lower=lb, upper=ub)
    oracle = oracles[len(sizes)](sizes, lb, ub)

    nside_points = (10000**(1 / len(sizes)))
    linsp = [np.linspace(lb[i], ub[i], nside_points)
             for i in range(len(sizes))]
    test_points = np.stack(np.meshgrid(*linsp), axis=-1)
    i0 = oracle.index(test_points)
    i1 = binr.get_bin_index(test_points)
    print(i0.shape)
    print(sizes, lb, ub)
    # print(i0[i0 != i1])
    # print(i1[i0 != i1])
    assert np.all(i0 == i1)

    test_indices = index_samples(len(binr), 10000)
    c0 = oracle.center(test_indices)['raw']
    c1 = binr.get_bin_center(test_indices)
    assert np.allclose(c0, c1)


@only_if_have_rif
@given(sizes_lb_ub())
@settings(max_examples=max_examples)
def test_BCC_against_rifcpp(inp):
    _test_BCC_against_rifcpp(inp)


@given(sizes_lb_ub())
@settings(max_examples=max_examples)
def test_BCC_invertibility_by_index(inp):
    sizes, lb, ub = inp
    binr = BCC(sizes=sizes, lower=lb, upper=ub)
    idx0 = index_samples(len(binr), 50000)
    cen = binr.get_bin_center(idx0)
    cen_inbounds = 0 == (np.sum(cen < binr.lower, axis=-1) +
                         np.sum(cen > binr.upper, axis=-1))
    idx0 = idx0[cen_inbounds]
    cen = cen[cen_inbounds]
    idx1 = binr.get_bin_index(cen)
    all_idx0_idx1_same = np.all(idx0 == idx1)
    print('fraction wrong:', sum(idx0 != idx1) / len(idx0))
    print(cen[idx0 != idx1])
    print(idx0[idx0 != idx1])
    print(idx1[idx0 != idx1])
    assert all_idx0_idx1_same


@given(sizes_lb_ub())
@settings(max_examples=max_examples)
def test_BCC_invertibility_by_center(inp):
    sizes, lb, ub = inp
    binr = BCC(sizes=sizes, lower=lb, upper=ub)
    pts = (binr.lower
           + (np.random.rand(10000, len(sizes))
               * (binr.upper - binr.lower)))
    assert np.all(binr.lower <= pts)
    assert np.all(binr.upper >= pts)
    idx0 = binr.get_bin_index(pts)
    assert np.all(idx0 >= 0)
    cen = binr.get_bin_center(idx0)
    idx1 = binr.get_bin_index(cen)
    print('fraction wrong:', sum(idx0 != idx1) / len(idx0))
    print(idx0[idx0 != idx1])
    print(idx1[idx0 != idx1])
    assert np.all(idx0 == idx1)


@given(sizes_lb_ub())
@settings(max_examples=max_examples)  # , max_shrinks=1, timeout=36000)
def test_BCC_covrad(inp):
    sizes, lb, ub = (np.array(x) for x in inp)
    ndim = len(sizes)
    binr = BCC(sizes=sizes, lower=lb, upper=ub)

    pts = binr.lower + np.random.rand(50000,
                                      ndim) * (binr.upper - binr.lower)
    assert np.all(binr.lower <= pts)
    assert np.all(binr.upper >= pts)
    idx = binr.get_bin_index(pts)
    assert np.all(idx >= 0)
    cen = binr.get_bin_center(idx)
    err = np.linalg.norm((pts - cen) / binr.width, axis=-1)

    # from testing, not math... should figure out?
    unit_cov_rad = [0.559,  # ndim == 3
                    0.705,  # ndim == 4
                    0.748,  # ndim == 5
                    0.854,  # ndim == 6
                    0.888,  # ndim == 7
                    0.984,  # ndim == 8
                    1.004,  # ndim == 9
                    1.068,  # ndim == 10
                    ]
    correct = np.all(err < unit_cov_rad[ndim - 3])
    if not correct:
        print('test_BCC_covrad_ndim', ndim, 'err', np.max(err))
    assert correct
