import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

def plot_samples(x, Y, x0, y0, figure, title="plotSamples"):

    assert x.shape[0] == Y.shape[1]

    y_mean = np.mean(Y, axis=0)
    y_std = np.std(Y, axis=0)
    fig = figure
    fig.set_size_inches(12.5, 7.5)

    plt.rc("xtick", labelsize=15)
    plt.rc("ytick", labelsize=15)

    if x0.shape[0] > 0:
        plt.scatter(x0, y0, color="#10FF00", s=100, zorder=3)

    plt.fill_between(
        x.flatten(), y_mean - y_std, y_mean + y_std, alpha=0.1, color="k", zorder=1
    )
    plt.fill_between(
        x.flatten(),
        y_mean - 2 * y_std,
        y_mean + 2 * y_std,
        alpha=0.1,
        color="k",
        zorder=1,
    )
    plt.fill_between(
        x.flatten(),
        y_mean - 3 * y_std,
        y_mean + 3 * y_std,
        alpha=0.1,
        color="k",
        zorder=1,
    )
    plt.plot(x, y_mean, color="red", zorder=2)
    plt.title(title)

    ax = plt.gca()
    ax.set_xlim(left=x[0], right=x[-1])
    ax.patch.set_facecolor("#EAEAF2")
    plt.setp(ax.spines.values(), color=None)
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color="w")
    ax.grid(True, color="w")

def get_plotting_mean_std(x, rhos, theta_S, x0, y0):
    """
    x must be n-d array
    x0 must be k-d array
    y0 must be k-1 array or array of length k
    """

    if len(y0.shape) == 1:
        y0 = y0.reshape((-1, 1))

    assert len(x.shape) == 2

    mean = np.zeros(x.shape[0])
    std = np.zeros(x.shape[0])

    for j in range(rhos.shape[0]):

        theta = theta_S[j, :]

        mu = theta[4] * np.ones((x.shape[0], 1))
        mu0 = theta[4] * np.ones(y0.shape)

        sigma = (
            theta[5] if len(theta) == 6 else 0.0
        )  # if not passed, will be left out of m and C
        E, E0 = np.eye(x.shape[0]), np.eye(x0.shape[0])

        Kx0K00Inv = K(x, x0, theta).dot(
            np.linalg.inv(K(x0, x0, theta) + sigma ** 2 * E0)
        )

        mi = mu + Kx0K00Inv.dot(y0 - mu0)
        mean += rhos[j] * mi.flatten()

        for i in range(x.shape[0]):
            xc = x[i, :].reshape((1, -1))
            Kx0K00Inv = K(xc, x0, theta).dot(np.linalg.inv(K(x0, x0, theta)))
            Ci = K(xc, xc, theta) + sigma ** 2 * E - Kx0K00Inv.dot(K(x0, xc, theta))
            std[i] += rhos[j] ** 2 * np.maximum(np.zeros(Ci.shape), Ci)[0, 0]

    std = np.sqrt(std)
    assert (
        len(mean.shape) == 1
        and mean.shape[0] == x.shape[0]
        and len(std.shape) == 1
        and std.shape[0] == x.shape[0]
    )
    return mean, std

def plot_onestep_1d(
    res,
    function,
    test_box,
    x_next,
    eta,
    rhos,
    theta_S,
    x0,
    y0,
    smoothGaussians=True,
    plotGradient=False,):

    x_max = test_box[0, 0]
    x_min = test_box[1, 0]
    X = np.linspace(x_min, x_max, res)

    y_func = function(X)
    plt.plot(X, y_func, "--k", label="objective function")
    y_max = y_func.max() * 1.2
    y_min = y_func.min() * 1.2

    y_loss = np.zeros(X.shape[0])
    y_loss_grad = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        y_loss[i] = expectedLoss(
            X[i].reshape((1, -1)), eta, rhos, theta_S, x0=x0, y0=y0
        ).flatten()
        y_loss_grad[i] = expectedLossGrad(
            X[i].reshape((1, -1)), eta, rhos, theta_S, x0=x0, y0=y0
        )

    mean, std = get_plotting_mean_std(X.reshape((-1, 1)), rhos, theta_S, x0, y0)

    plt.plot(X, y_loss, color="blue", label="expected loss")
    if plotGradient:
        plt.plot(X, y_loss_grad, color="green", label="expected loss grad")

    plt.plot(X, mean, "r", label="mean")
    if smoothGaussians:
        y_space = np.linspace(
            y_min, y_max, int(X.shape[0] * (y_max - y_min) / (x_max - x_min))
        )
        M = np.flip(
            -np.abs(y_space[:, None] - mean[None, :]) / std[None, :] + 3, axis=0
        )
        cm = LinearSegmentedColormap.from_list("myMap", [(1, 1, 1), (1, 0.5, 0.5)])
        plt.imshow(
            M,
            extent=[x_min, x_max, y_min, y_max],
            interpolation="bicubic",
            vmin=0,
            vmax=3,
            cmap=cm,
        )
    else:
        plt.fill_between(
            X, mean - std, mean + std, alpha=0.1, color="red", label="+- 1SD"
        )
    plt.scatter(x0, y0, marker="+", color="black", s=300, label="observation")

    x_next_loss = expectedLoss(x_next, eta, rhos, theta_S, x0=x0, y0=y0)
    plt.scatter(x_next, x_next_loss, color="blue", marker="D", label="next evaluation")

    plt.hlines(
        eta,
        x_min,
        x_max,
        linestyles="--",
        linewidths=0.8,
        color="green",
        label=r"$\eta$",
    )

    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max)
    plt.title("One Step Lookahead: function evaluation #" + str(x0.shape[0]))
    plt.grid(True)
    plt.legend(loc="best")
    plt.show()

def plot_onestep_2d(res, function, test_box, x_next, eta, rhos, theta_S, x0, y0):

    x_max = test_box[1, 0].flatten()[0] * 1.2
    x_min = test_box[0, 0].flatten()[0] * 1.2

    y_max = test_box[1, 1].flatten()[0] * 1.2
    y_min = test_box[0, 1].flatten()[0] * 1.2

    print(x_max)
    print(x_min)
    print(y_max)
    print(y_min)

    X = np.linspace(x_min, x_max, res)
    Y = np.linspace(y_min, y_max, res)

    z_func = np.zeros((X.shape[0], Y.shape[0]))
    z_loss = np.zeros((X.shape[0], Y.shape[0]))
    z_loss_grad = np.zeros((X.shape[0], Y.shape[0], 2))
    z_variance = np.zeros((X.shape[0], Y.shape[0]))
    z_mean = np.zeros((X.shape[0], Y.shape[0]))

    X_line = np.zeros((X.shape[0] * Y.shape[0], 2))

    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):

            x = np.array([X[i], Y[j]]).reshape((1, -1))
            X_line[i + Y.shape[0] * j] = x

            z_func[i, j] = function(x)

            # print (X[i], '^2 + ', Y[j], '^2 = ', z_func[i, j])

            for k in range(theta_S.shape[0]):
                mi, Ci = getMoments(x, x0, y0, theta_S[k, :])
                z_variance[i, j] += rhos[k] ** 2 * Ci.flatten()[0]
                z_mean[i, j] += rhos[k] * mi.flatten()[0]
            z_variance[i, j] = np.sqrt(z_variance[i, j])

            z_loss[i, j] = expectedLoss(x, eta, rhos, theta_S, x0=x0, y0=y0).flatten()
            z_loss_grad[i, j, :] = expectedLossGrad(
                x.reshape((1, -1)), eta, rhos, theta_S, x0=x0, y0=y0
            )

    fig = plt.figure(x0.shape[0], figsize=(15, 15))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    imFunc = ax1.imshow(
        np.flip(z_func.T, axis=0),
        extent=[x_min, x_max, y_min, y_max],
        interpolation="bicubic",
        vmin=z_func.min(),
        vmax=z_func.max(),
    )
    imExpLoss = ax2.imshow(
        np.flip(z_loss.T, axis=0),
        extent=[x_min, x_max, y_min, y_max],
        interpolation="bicubic",
        cmap="Blues",
    )
    imMean = ax3.imshow(
        np.flip(z_mean.T, axis=0),
        extent=[x_min, x_max, y_min, y_max],
        interpolation="bicubic",
        vmin=z_func.min(),
        vmax=z_func.max(),
    )
    imVar = ax4.imshow(
        np.flip(z_variance.T, axis=0),
        extent=[x_min, x_max, y_min, y_max],
        interpolation="bicubic",
        cmap="Reds_r",
    )

    fig.colorbar(imFunc, ax=ax1)
    fig.colorbar(imExpLoss, ax=ax2)
    fig.colorbar(imVar, ax=ax3)
    fig.colorbar(imMean, ax=ax4)

    # mean, std = getMoments(X_line, rhos, theta_S, x0, y0)

    axes = [ax1, ax2, ax3, ax4]

    for i in range(4):
        axes[i].scatter(
            x0[:, 0], x0[:, 1], marker="+", color="black", s=300, label="observation"
        )
        axes[i].scatter(
            x_next[:, 0],
            x_next[:, 1],
            color="blue",
            marker="D",
            label="next evaluation",
        )
        axes[i].legend(loc="best")
        axes[i].set_ylim(y_min, y_max)
        axes[i].set_xlim(x_min, x_max)
        axes[i].grid(True)

    ax1.set_title("Objective Function")
    ax2.set_title("Expected Loss")
    ax3.set_title("GP Mean")
    ax4.set_title("Standard Deviation")

    plt.show()
