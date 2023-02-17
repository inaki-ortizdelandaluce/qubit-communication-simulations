from qt.classical import *
from qt.random import vector3


def test_heaviside():
    a = np.array([-1e-9, 1e-9, -1., +1.])
    assert np.allclose(np.array([0, 1, 0, 1]), heaviside(a))


def test_theta():
    a = np.array([-1e-9, 1e-9, -1., +1.])
    assert np.allclose(np.array([0, 1e-9, 0, 1]), theta(a))


def test_prepare():
    np.random.seed(0)
    lambda1, lambda2 = vector3(), vector3()
    bits = prepare(lambda1, lambda2)

    # lamda1 -> array([ 0.82760922, -0.03589592,  0.56015575])
    # lambda2 -> array([ 0.79972854, -0.38493927,  0.46071251]))
    # x -> array([-0.36250052, 0.43579601, 0.82381746])

    assert (bits[0] == 1 and bits[1] == 0)


def test_measure_pvm():
    np.random.seed(0)
    lambda1, lambda2 = vector3(), vector3()
    p = measure_pvm(lambda1, lambda2, 1, 0)
    print(p)
    assert True

