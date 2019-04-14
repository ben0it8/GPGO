import numpy as np

def kernel(x1, x2, theta = np.array([1, 1, 1, 1]), periodic = False):
    """
    x1 must be n-d array
    x2 must be k-d array
    returns : n-k array
    """
    assert len(x1.shape) == 2 and len(x2.shape) == 2
    assert x1.shape[1] == x2.shape[1]
    
    hA, hB, wA, wB = theta[0], theta[1], theta[2], theta[3]
    
    d = x1.shape[1]
    n1 = x1.shape[0]
    n2 = x2.shape[0]

    X_diffs = np.zeros((d, n1, n2))

    for i in range(d):
        X_diffs[i,:,:] = x1[:,i,None] - x2[:,i,None].T
        
    wA2Inv = 1 / (wA * wA)
    r2A = np.sum(np.power(X_diffs, 2), axis = 0) * wA2Inv
    
    wBInv = 1 / wB
    r2B = np.sum(np.power(np.sin(np.pi * X_diffs) * wBInv, 2), axis = 0)
    
    out = (hA**2) * np.exp(-r2A * 0.5) + periodic * ((hB**2) * np.exp(-r2B * 0.5))
    
    assert len(out.shape) == 2 and out.shape[0] == x1.shape[0] and out.shape[1] == x2.shape[0]
    
    return out

def d_kernel(x1, x2, theta = np.array([1, 1, 1, 1]), periodic = False):
    """
    x1 must be 1-d array (single point) or array of length d
    x2 must be k-d array
    returns : d-k array (each entry derived for all dimension)
    """
    
    if len(x1.shape) == 1:
        x1 = x1.reshape((1, -1))
    
    assert len(x1.shape) == 2 and len(x2.shape) == 2
    assert x1.shape[1] == x2.shape[1]
    assert x1.shape[0] == 1    # only one samples allowed!
    
    hA = theta[0]
    hB = theta[1]
    wA = theta[2]
    wB = theta[3]
    
    d = x1.shape[1]
    n1 = x1.shape[0]
    n2 = x2.shape[0]

    X_diffs = np.zeros((d, n1, n2))

    for i in range(d):
        X_diffs[i,:,:] = x1[:,i,None] - x2[:,i,None].T
        
    wA2Inv = 1 / (wA * wA)
    r2A = np.sum(np.power(X_diffs, 2), axis = 0) * wA2Inv
    
    wBInv = 1 / wB
    r2B = np.sum(np.power(np.sin(np.pi * X_diffs) * wBInv, 2), axis = 0)
    
    out = np.zeros((d, n2))
    for i in range(d):
        out[i, :] = - X_diffs[i,:,:] * (hA**2 / (wA**2)) * np.exp(-r2A * 0.5) + \
                  periodic * (-np.pi * (hB**2)/(2*wB**2)*np.sin(2*np.pi*X_diffs[i,:,:]) * np.exp(-r2B * 0.5))
    
    assert len(out.shape) == 2 and out.shape[0] == x1.shape[1] and out.shape[1] == x2.shape[0]
    
    return out

def approx_grad(x, y):

    grad = np.zeros(y.shape)
    for i in range(y.shape[0]):
        if i == 0:
            grad[i] = (y[1] - y[0]) / (x[1] - x[0])
        elif i == y.shape[0] - 1:
            grad[i] = (y[i] - y[i - 1]) / (x[i] - x[i - 1])
        else:
            grad[i] = ((y[i + 1] - y[i - 1])) / (x[i + 1] - x[i - 1])

    return grad

def GAP(obj_fn, y_opt, x_0, x_i):
    """Describes the reduction of the gap between starting value and 
    global minima after i iterations of optimization 
    
    G_i = f(x_0) - f(x_i) / f(x_0) - f*
    
    obj_fn: callable, the function being optimized
    y_opt: scalar-valued optimum of obj_fn
    x_0: d_1 array with first evaluation point
    x_i dx1 array with last evaluation point
    """
    assert np.isscalar(y_opt)
    assert (x_0.shape) == (x_i.shape)
    x_0, x_i = np.array([x_0]), np.array([x_i])
    return round(
        float(np.squeeze((obj_fn(x_0) - obj_fn(x_i)) / (obj_fn(x_0) - y_opt))), 4)

def get_moments(x, x0, y0, theta):
    """
    This computes m_theta and C_theta.
    Assumes, that length(theta) is 5 in noiseless and 6 in noisy scenario.
    x must be n x d array
    x0 must be k x d array
	y0 must be k x 1 array
    returns : m (n-1 array); C (n x n array)
    """
    assert len(x.shape) == 2 and len(x0.shape) == 2
    assert x.shape[1] == x0.shape[1]

    mu = theta[4] * np.ones((x.shape[0], 1))
    mu0 = theta[4] * np.ones(y0.shape)
    sigma = (
        theta[5] if len(theta) == 6 else 0.0
    )  # if not passed, will be left out of m and C

    E, E0 = np.eye(x.shape[0]), np.eye(x0.shape[0])

    Kx0K00Inv = kernel(x, x0, theta).dot(
        np.linalg.inv(kernel(x0, x0, theta) + sigma ** 2 * E0)
    )

    mTheta = mu + Kx0K00Inv.dot(y0 - mu0)
    CTheta = kernel(x, x, theta) + sigma ** 2 * E - Kx0K00Inv.dot(kernel(x0, x, theta))

    # CTheta must be positive semidefinite!
    try:
        np.linalg.cholesky(CTheta)
    except np.linalg.LinAlgError:
        CTheta += np.eye(CTheta.shape[0]) * 1e-6

    assert mTheta.shape[0] == x.shape[0] and mTheta.shape[1] == 1
    assert CTheta.shape[0] == x.shape[0] and CTheta.shape[1] == x.shape[0]

    return mTheta, CTheta

