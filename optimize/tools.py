from optimize.constants import *
import numpy as np
import matplotlib.pyplot as plt


def estimate_gradient(f, x, h=DERIV_TOL):
    n = x.size
    df = np.zeros(n)
    for k in range(n):
        delta = np.zeros(n)
        delta[k] = h
        df[k] = (f(x + delta) - f(x - delta)) / (2 * h)
    return df


def line_along(f, x, d):
    def g(alpha):
        return f(x + alpha * d)

    def dg(alpha):
        return estimate_gradient(f, x + alpha * d).dot(d)

    return g, dg


def cubic_interpolation(g, dg, s=1, cubic_tol=CUBIC_TOl):
    # Determine initial interval
    a = 0
    b = s
    while not (dg(b) >= 0 and g(b) >= g(a)):
        a = b
        b *= 2

    # Update current interval
    alpha = None
    while abs(a - b) > cubic_tol:
        ga = g(a)
        gb = g(b)
        dga = dg(a)
        dgb = dg(b)
        z = 3 * (ga - gb) / (b - a) + dga + dgb
        w = np.sqrt(z ** 2 - dga * dgb)
        alpha = b - (dgb + w - z) / (dgb - dga + 2 * w) * (b - a)

        if dg(alpha) >= 0 or g(alpha) >= g(a):
            b = alpha
        elif dg(alpha) < 0 or g(alpha) < g(a):
            a = alpha
    return alpha  # step size


def armijo(f, x, d, dfx, sigma=SIGMA, b=B, s=S):
    fx = f(x)
    m = 0
    while fx - f(x + b ** m * s * d) < -sigma * b ** m * s * dfx.dot(d):
        m += 1
    alpha = b ** m * s
    return alpha  # step size


def stopping_criteria(i, dfx0, dfx, max_iter=MAX_ITER, grad_tol=GRAD_TOL):
    return (np.linalg.norm(dfx) / np.linalg.norm(dfx0)) > grad_tol and i < max_iter


def steepest_descent(f, df, x0, line_search='armijo'):
    dfx0 = df(x0)

    x = x0
    dfx = df(x)
    d = -dfx

    i = 0
    while stopping_criteria(i, dfx0, dfx):
        # Determine step size
        alpha = None
        if line_search == 'armijo':
            alpha = armijo(f, x, d, dfx)
        elif line_search == 'cubic_interpolation':
            g, dg = line_along(f, x, d)
            alpha = cubic_interpolation(g, dg)

        # Update current point
        x += alpha * d

        # Update iteration number
        i += 1

        # Update gradient and descent direction
        dfx = df(x)
        d = -dfx
    return x, f(x), i


def test():
    # Test quadratic function
    def quadratic(x):
        return x[0] ** 2 + x[1] ** 2 + 5.

    # Test quadratic function gradient
    def dquadratic(x):
        return np.array([2 * x[0],
                         2 * x[1]])

    # Test rosenbrock function
    def rosenbrock(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    # Test rosenbrock function gradient
    def drosenbrock(x):
        return np.array([-2 + 2 * x[0] - 400 * x[0] * (x[1] - x[0] ** 2),
                         200 * (x[1] - x[0] ** 2)])

    # Test steepest descent
    print('\nMinimizing quadratic function with armijo...')
    x0 = np.array([5., 5.])
    x_min, fx_min, niter = steepest_descent(quadratic, dquadratic, x0, line_search='armijo')
    print('xmin: {}\nfmin: {}\nNumber of iterations: {}'.format(x_min, fx_min, niter))

    # # Test quadratic function
    # print('\nMinimizing quadratic function with cubic interpolation...')
    # x0 = np.array([5., 5.])
    # x_min, fx_min, niter = steepest_descent(quadratic, dquadratic, x0, line_search='cubic_interpolation')
    # print('xmin: {}\nfmin: {}\nNumber of iterations: {}'.format(x_min, fx_min, niter))

    print('\nMinimizing rosenbrock function with armijo...')
    x0 = np.array([5., 5.])
    x_min, fx_min, niter = steepest_descent(rosenbrock, drosenbrock, x0, line_search='armijo')
    print('xmin: {}\nfmin: {}\nNumber of iterations: {}'.format(x_min, fx_min, niter))

    print('\nMinimizing rosenbrock function cubic interpolation...')
    x0 = np.array([5., 5.])
    x_min, fx_min, niter = steepest_descent(rosenbrock, drosenbrock, x0, line_search='cubic_interpolation')
    print('xmin: {}\nfmin: {}\nNumber of iterations: {}'.format(x_min, fx_min, niter))

    # scipy.optimize.minimize test
    print('\nMinimizing rosenbrock function with scipy.optimize.minimize...')
    from scipy.optimize import minimize
    res = minimize(rosenbrock, x0)
    print(res)

    # Test line_along by drawing curves
    x0 = np.array([0., 2.])
    d = np.array([1, 0])
    g, dg = line_along(rosenbrock, x0, d)
    t = np.arange(-2., 2., 0.01)
    gt = np.zeros(t.size)
    for n in range(t.size):
        gt[n] = g(t[n])

    plt.plot(t, gt)
    plt.title('Rosenbrock function 1D curve')
    plt.show()


if __name__ == '__main__':
    test()
    pass
