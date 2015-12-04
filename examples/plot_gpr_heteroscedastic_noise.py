# Authors: Jan Hendrik Metzen <janmetzen@mailbox.org>
#
# License: BSD 3 clause
"""
==============================================================
Illustration how HeteroscedasticKernel can learn a noise model
==============================================================

A heteroscedastic kernel allows adapting to situations where different regions
in the data space exhibit different noise levels. For this, the kernel learns
for a set of prototypes values from the data space explicit noise levels.
These exemplary noise levels are then generalized to the entire data space by
means for kernel regression.

In the shown example, a homoscedastic and heteroscedastic noise kernel are
compared. The function to be learned is a simple linear relationship; however,
the noise level grows quadratically with the input. Both kernels allow
capturing the mean equally well; however, the heteroscedastic kernel can
considerably better explain the observed data (according to the log-marginal
likelihood LML) and provide better noise estimates.
"""
print __doc__

import numpy as np

import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, ConstantKernel as C
from sklearn.cluster import KMeans

from gp_extras.kernels import HeteroscedasticKernel

np.random.seed(0)

# Generate data
n_samples = 100
def f(X):
    # target function is just a linear relationship + heteroscadastic noise
    return X + 0.5*np.random.multivariate_normal(np.zeros(X.shape[0]),
                                                 np.diag(X**2), 1)[0]

X = np.random.uniform(-7.5, 7.5, n_samples)  # input data
y = f(X)  # Generate target values by applying function to manifold

# Gaussian Process with RBF kernel and homoscedastic noise level
kernel_homo = C(1.0, (1e-10, 1000)) * RBF(1, (0.01, 100.0)) \
    + WhiteKernel(1e-3, (1e-10, 50.0))
gp_homoscedastic = GaussianProcessRegressor(kernel=kernel_homo, alpha=0)
gp_homoscedastic.fit(X[:, np.newaxis], y)
print "Homoscedastic kernel: %s" % gp_homoscedastic.kernel_
print "Homoscedastic LML: %.3f" \
    % gp_homoscedastic.log_marginal_likelihood(gp_homoscedastic.kernel_.theta)
print

# Gaussian Process with RBF kernel and heteroscedastic noise level
prototypes = KMeans(n_clusters=10).fit(X[:, np.newaxis]).cluster_centers_
kernel_hetero = C(1.0, (1e-10, 1000)) * RBF(1, (0.01, 100.0)) \
    + HeteroscedasticKernel.construct(prototypes, 1e-3, (1e-10, 50.0),
                                      gamma=5.0, gamma_bounds="fixed")
gp_heteroscedastic = GaussianProcessRegressor(kernel=kernel_hetero, alpha=0)
gp_heteroscedastic.fit(X[:, np.newaxis], y)
print "Heteroscedastic kernel: %s" % gp_heteroscedastic.kernel_
print "Heteroscedastic LML: %.3f" \
    % gp_heteroscedastic.log_marginal_likelihood(gp_heteroscedastic.kernel_.theta)


# Plot result
X_ = np.linspace(-7.5, 7.5, 100)
y_ = X_
noise_std = 0.5 * X_

plt.subplot(1, 2, 1)
plt.scatter(X, y)
plt.plot(X_, y_, 'b', label="true function")
plt.fill_between(X_, y_ - noise_std, y_ + noise_std,
                 alpha=0.5, color='b')
y_mean, y_std = gp_homoscedastic.predict(X_[:, None], return_std=True)
plt.plot(X_, y_mean, 'k', lw=3, zorder=9, label="predicted mean")
plt.fill_between(X_, y_mean - y_std, y_mean + y_std,
                 alpha=0.5, color='k')
plt.xlim(-7.5, 7.5)
plt.legend(loc="best")
plt.title("Homoscedastic noise model")

plt.subplot(1, 2, 2)
plt.scatter(X, y)
plt.plot(X_, y_, 'b', label="true function")
plt.fill_between(X_, y_ - noise_std, y_ + noise_std,
                 alpha=0.5, color='b')
y_mean, y_std = gp_heteroscedastic.predict(X_[:, None], return_std=True)
plt.plot(X_, y_mean, 'k', lw=3, zorder=9, label="predicted mean")
plt.fill_between(X_, y_mean - y_std, y_mean + y_std,
                 alpha=0.5, color='k')
plt.xlim(-7.5, 7.5)
plt.legend(loc="best")
plt.title("Heteroscedastic noise model")
plt.show()

