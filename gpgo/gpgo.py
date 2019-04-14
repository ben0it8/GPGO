__author__ = "Oliver Atanaszov, Pippo Aiko, Lukas Radke"
__contact__ = "oliver.atanaszov@gmail.com"

"""Implementation of Osborne et al. 2019 "Gaussian Processes for Global Optimization" paper

http://www.robots.ox.ac.uk/~mosb/public/pdf/115/Osborne%20et%20al.%20-%202009%20-%20Gaussian%20processes%20for%20global%20optimization.pdf
"""

import numpy as np
from scipy.stats import multivariate_normal
import time
import scipy.optimize as scop
import matplotlib.pyplot as plt
from .utils import kernel, d_kernel, get_moments
from .plot_utils import plot_onestep_1d

def compute_rhos(
    theta_prior=np.array([1, 1, 1, 1, 0]),  # = nu in the paper (6)
    theta_dist_cov=np.eye(5),  # = lambda.T * lambda in the paper (6)
    theta_space_cov=np.eye(5),  # = w.T * w in the paper (6)
    theta_sample_count=10,
    x0=np.zeros(0),
    y0=np.zeros(0),
    theta_S=None,
    periodic=False):
    """
    Compute rhos for eq 5: p(y*|x*,I0) = sum_i [rho_i * N(y*; m_i, C_i)]

    x0 must be k-d array
    y0 must be k-1 array
    returns : rhos & theta_S (each of size 'theta_sample_count')
    """

    # function returns the rho weights and the corresponding theta values
    # ---------------------------------------------------------------------------------------

    S = theta_sample_count
    theta_dim = theta_prior.shape[0]

    # draw S random samples for theta
    if theta_S == None:
        theta_S = multivariate_normal.rvs(theta_prior, theta_dist_cov, S)

    # compute weights from equation (13) ----------------------------------------------------

    K_theta = np.zeros((S, S))
    FRAK_N = np.zeros((S, S))

    for i in range(S):
        for j in range(S):

            # equation (11)
            K_theta[i, j] = multivariate_normal.pdf(
                theta_S[i, :],
                mean=theta_S[j, :],
                cov=theta_dist_cov,
                allow_singular=True,
            )

            # equation (12)
            theta_stack = np.hstack((theta_S[i, :], theta_S[j, :]))
            mean_stack = np.hstack((theta_prior, theta_prior))

            cov_stack = np.zeros((2 * theta_dim, 2 * theta_dim))
            cov_stack[:theta_dim, :theta_dim] = theta_dist_cov + theta_space_cov
            cov_stack[theta_dim:, theta_dim:] = theta_dist_cov + theta_space_cov
            cov_stack[theta_dim:, :theta_dim] = theta_dist_cov
            cov_stack[:theta_dim, theta_dim:] = theta_dist_cov

            FRAK_N[i, j] = multivariate_normal.pdf(
                theta_stack, mean=mean_stack, cov=cov_stack, allow_singular=True
            )

    K_theta_Inv = np.linalg.inv(K_theta)
    M = K_theta_Inv.dot(FRAK_N).dot(K_theta_Inv)

    # sample rS -------------------------------------------------------------------------
    rS = np.ones((S, 1))

    if x0.shape[0] > 1:
        for i in range(S):

            K00 = kernel(x0, x0, theta_S[i, :], periodic)

            rS[i, :] = multivariate_normal.pdf(
                y0.flatten(),
                mean=theta_S[i, 4] * np.ones(y0.shape[0]),
                cov=K00,
                allow_singular=True,
            )

    # weights rho -----------------------------------------------------------------------
    weighted_rS = M.dot(rS)
    onesT = np.ones(S).reshape((1, S))

    rho = weighted_rS / (onesT.dot(weighted_rS))

    return rho, theta_S

def V(x, eta, x0, y0, theta):
    """
    Eq. (8)
    x must be 1-d array
    x0 must be k-d array
    y0 must be k-1 array
    rho must be an array of length S
    theta must be an array of length >= 4
    """

    x = x.reshape((1, -1))  # x is a single  sample

    mi, Ci = get_moments(x, x0, y0, theta)

    Phi = multivariate_normal.cdf(eta, mean=mi, cov=Ci, allow_singular=True)
    N = multivariate_normal.pdf(eta, mean=mi, cov=Ci, allow_singular=True)

    return eta + (mi - eta) * Phi - Ci * N

def dV(x, eta, x0, y0, theta, periodic=False):
    """
    x must be 1-d array (single point) or array of length d
    x0 must be k-d array
    y0 must be k-1 array or array of length k
    returns : vector of length d
    """

    if len(x.shape) == 1:
        x = x.reshape((1, -1))
    if len(y0.shape) == 1:
        y0 = y0.reshape((-1, 1))

    assert len(x.shape) == 2 and x.shape[0] == 1
    assert len(x0.shape) == 2 and x0.shape[1] == x.shape[1]
    assert len(y0.shape) == 2 and y0.shape[0] == x0.shape[0] and y0.shape[1] == 1

    sigma = theta[5] if len(theta) == 6 else 0.0
    E0 = np.eye(x0.shape[0])

    x = x.reshape((1, -1))  # x is a single  sample
    mu0 = theta[4] * np.ones(y0.shape)

    mi, Ci = get_moments(x, x0, y0, theta)

    # Kx0K00Inv = kernel(x, x0, theta).dot(np.linalg.inv(kernel(x0, x0, theta) + sigma ** 2 * E0))
    K00Inv = np.linalg.inv(kernel(x0, x0, theta, periodic) + sigma ** 2 * E0)

    Kx0Grad = d_kernel(x, x0, theta, periodic)
    dCi = -2 * Kx0Grad.dot(K00Inv).dot(kernel(x0, x, theta, periodic))
    dmi = Kx0Grad.dot(K00Inv).dot(y0 - mu0)

    # Ci must be positive semidefinite
    Ci = np.maximum(Ci, 0)

    # Phi = multivariate_normal.cdf(eta, mean=mi, cov=Ci, allow_singular=True)
    # N = multivariate_normal.pdf(eta, mean=mi, cov=Ci, allow_singular=True)

    out = dmi * multivariate_normal.cdf(
        eta, mi, Ci, allow_singular=True
    ) - 0.5 * dCi * multivariate_normal.pdf(eta, mi, Ci, allow_singular=True)

    assert len(out.shape) == 2 and out.shape[0] == x.shape[1] and out.shape[1] == 1

    return out.flatten()

def expected_loss(x, eta, rhos, theta_S, x0, y0, test_box=None, epsilon=1):

    eta_arr = np.zeros(theta_S.shape[0])

    if len(theta_S.shape) < 6:
        eta_arr = eta * np.ones(theta_S.shape[0])
    else:
        for i in range(theta_S.shape[0]):
            mi, Ci = get_moments(x0, x0, y0, theta_S[i, :])
            confident_evals = mi[1 / (2 * np.diag(Ci)) < epsilon]
            if confident_evals.shape[0] > 0:
                eta_arr[i] = confident_evals.min()
            else:
                eta_arr[i] = mi.min()

    if test_box is not None:
        bmin, bmax = test_box[0], test_box[1]
        if not np.all((x < bmax) * (x > bmin)):
            return np.inf

    exp_loss = 0
    for i in range(theta_S.shape[0]): # eq (7)
        exp_loss += rhos[i] * V(x, eta_arr[i], x0, y0, theta_S[i, :])

    return exp_loss

def d_expected_loss(x, eta, rhos, theta_S, x0=np.zeros(0), y0=np.zeros(0), epsilon=1):

    eta_arr = np.zeros(theta_S.shape[0])

    if len(theta_S.shape) < 6:
        eta_arr = eta * np.ones(theta_S.shape[0])
    else:
        for i in range(theta_S.shape[0]):
            mi, Ci = get_moments(x0, x0, y0, theta_S[i, :])
            confident_evals = mi[1 / (2 * np.diag(Ci)) < epsilon]
            if confident_evals.shape[0] > 0:
                eta_arr[i] = confident_evals.min()
            else:
                eta_arr[i] = mi.min()

    exp_loss = np.zeros(x.shape)
    for i in range(theta_S.shape[0]):
        exp_loss += rhos[i] * dV(x, eta_arr[i], x0, y0, theta_S[i, :])

    assert exp_loss.shape == x.shape

    return exp_loss

def one_step_lookahead(x0, y0, theta_prior, test_box, theta_sample_count=16, disp=True, optimizer=scop.fmin_bfgs,
                      theta_S=None, epsilon=1, init_guess_count=1, periodic=False):
    """One-step lookahead as described in the paper """
    # theta prior handling
    theta_dist_cov = np.eye(len(theta_prior))
    theta_space_cov = np.eye(len(theta_prior))

    if disp:
        print("theta prior:", theta_prior)

    eta = y0.min()

    # select a random (uniformly distributed) point in the box spanned by x0
    x_init = np.random.uniform(test_box[0, :], test_box[1, :])
    if disp:
        print("initial guess:", x_init)

    # if x_init is too close to any of the points in x0, we add some noise to "kick" it away
    for i, d in enumerate([np.linalg.norm(x_init - x) for x in x0]):
        if d < 1e-9:
            x0[i] = x0[i] + np.random.randn()

    rhos, theta_S = compute_rhos(
        theta_prior,
        theta_dist_cov=theta_dist_cov,
        theta_space_cov=theta_space_cov,
        theta_sample_count=theta_sample_count,
        x0=x0,
        y0=y0,
        theta_S=theta_S,
    )

    exp_loss = lambda x: expected_loss(
        x,
        eta=eta,
        rhos=rhos,
        theta_S=theta_S,
        x0=x0,
        y0=y0,
        test_box=test_box,
        epsilon=epsilon,
    )
    d_exp_loss = lambda x: d_expected_loss(
        x, eta=eta, rhos=rhos, theta_S=theta_S, x0=x0, y0=y0, epsilon=epsilon
    )

    x_next = optimizer(exp_loss, fprime=d_exp_loss, x0=x_init, disp=disp)

    for i in range(init_guess_count - 1):
        x_init = np.random.uniform(test_box[0, :], test_box[1, :])
        for i, d in enumerate([np.linalg.norm(x_init - x) for x in x0]):
            if d < 1e-9:
                x0[i] = x0[i] + np.random.randn()
        optimizeResult = optimizer(exp_loss, fprime=d_exp_loss, x0=x_init, disp=disp)
        if exp_loss(optimizeResult) < exp_loss(x_next):
            x_next = optimizeResult

    return x_next, eta, rhos, theta_S, x0, y0

def run_experiment(test_fn, x0, test_box, max_iter=5, noise=None, disp=False, optimizer=scop.fmin_bfgs, 
                   init_guess_count=1, periodic=False, epsilon=1):
    """TODO: ADD DOCUMENTATION"""

    for k in range(max_iter):

        if noise is not None:
            theta_prior = np.array([1, 1, 1, 1, np.mean(np.mean(test_fn(x0))), noise])
        else:
            theta_prior = np.array([1, 1, 1, 1, np.mean(np.mean(test_fn(x0)))])

        print(f"iter {k+1}/{max_iter}")
        x_next, eta, rhos, theta_S, x0, y0 = one_step_lookahead(
            x0, test_fn(x0), theta_prior, test_box
        )
        print("next eval at x =", x_next)

        if np.isscalar(x_next):
            x_next = np.array([x_next])
        x_next = x_next.reshape((1, -1))
        plt.figure(x0.shape[0], figsize=(8.5, 6))
        plot_onestep_1d(
            200,
            test_fn,
            test_box,
            x_next,
            eta,
            rhos,
            theta_S,
            x0,
            y0,
            smoothGaussians=False,
        )
        x0 = np.vstack((x0, x_next))
    return x0
