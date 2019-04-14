import numpy as np
from scipy.stats import multivariate_normal
from .utils import kernel, d_kernel, get_moments

def sample_posterior(
    x, theta=np.array([1, 1, 1, 1, 0]), x0=np.zeros(0), y0=np.zeros(0), samples=1):
    """
    x must be n-d array
    x0 must be k-d array
    y0 must be k-1 array
    returns : samples-1 array
    """

    if x0.shape[0] < 1:  # no prior functions evaluations

        # mu is constant prior
        mu = theta[4] * np.ones(x.shape[0])
        return multivariate_normal.rvs(mu, kernel(x, x, theta), samples)

    else:  # prior functions evaluations

        mTheta, CTheta = get_moments(x, x0, y0, theta)

        return multivariate_normal.rvs(mTheta.flatten(), CTheta, samples)

def sample_marginalized_posterior(
    x, rho, theta_S, x0=np.zeros(0), y0=np.zeros(0), samples=1):
    """
    x must be n-d array
    x0 must be k-d array
    y0 must be k-1 array
    rho must be an array of length S
    theta_S must be S-m array
    """

    Y = np.zeros((samples, x.shape[0]))

    for j in range(theta_S.shape[0]):
        Y += rho[j] * sample_posterior(x, theta_S[j, :], x0, y0, samples=samples)

    return Y