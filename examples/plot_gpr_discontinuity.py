# Authors: Jan Hendrik Metzen <janmetzen@mailbox.org>
#
# License: BSD 3 clause
"""
============================================================================
Illustration how the ManifoldKernel can be used to deal with discontinuities
============================================================================

The ManifoldKernel allows to learn a mapping from low-dimensional input space
(1d in this case) to a higher-dimensional manifold (2d in this case). Since this
mapping is non-linear, this can be effectively used for turning a stationary
base kernel into a non-stationary kernel, where the non-stationarity is
learned. In this example, this used to learn a function which is sinusoidal but
with a discontinuity at x=0. Using an adaptable non-stationary kernel allows
to model uncertainty better and yields a better extrapolation beyond observed
data in this example.
"""

print(__doc__)

import numpy as np
import pylab

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from gp_extras.kernels import ManifoldKernel

np.random.seed(1)

# Specify Gaussian Process
kernel = C(1.0, (0.01, 100)) \
    * ManifoldKernel.construct(base_kernel=RBF(0.1), architecture=((1, 2),),
                               transfer_fct="tanh", max_nn_weight=1)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5,
                              n_restarts_optimizer=10)

X_ = np.linspace(-7.5, 7.5, 100)
y_ = np.sin(X_) + (X_ > 0)

# Visualization of prior
pylab.figure(0, figsize=(10, 8))
X_nn = gp.kernel.k2._project_manifold(X_[:, None])
pylab.subplot(3, 2, 1)
for i in range(X_nn.shape[1]):
    pylab.plot(X_, X_nn[:, i], label="Manifold-dim %d" % i)
pylab.legend(loc="best")
pylab.xlim(-7.5, 7.5)
pylab.title("Prior mapping to manifold")

pylab.subplot(3, 2, 2)
y_mean, y_cov = gp.predict(X_[:, None], return_cov=True)
pylab.plot(X_, y_mean, 'k', lw=3, zorder=9, label="mean")
pylab.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                   y_mean + np.sqrt(np.diag(y_cov)),
                   alpha=0.5, color='k')
y_samples = gp.sample_y(X_[:, None], 10)
pylab.plot(X_, y_samples, color='b', lw=1)
pylab.plot(X_, y_samples[:, 0], color='b', lw=1, label="samples") # just for the legend
pylab.legend(loc="best")
pylab.xlim(-7.5, 7.5)
pylab.ylim(-4, 3)
pylab.title("Prior samples")


# Generate data and fit GP
X = np.random.uniform(-5, 5, 40)[:, None]
y = np.sin(X[:, 0]) + (X[:, 0] > 0)
gp.fit(X, y)

# Visualization of posterior
X_nn = gp.kernel_.k2._project_manifold(X_[:, None])

pylab.subplot(3, 2, 3)
for i in range(X_nn.shape[1]):
    pylab.plot(X_, X_nn[:, i], label="Manifold-dim %d" % i)
pylab.xlim(-7.5, 7.5)
pylab.legend(loc="best")
pylab.title("Posterior mapping to manifold")

pylab.subplot(3, 2, 4)
y_mean, y_cov = gp.predict(X_[:, None], return_cov=True)
pylab.plot(X_, y_mean, 'k', lw=3, zorder=9, label="mean")
pylab.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                   y_mean + np.sqrt(np.diag(y_cov)),
                   alpha=0.5, color='k')
y_samples = gp.sample_y(X_[:, None], 10)
pylab.plot(X_, y_samples, color='b', lw=1)
pylab.plot(X_, y_samples[:, 0], color='b', lw=1, label="samples") # just for the legend
pylab.scatter(X[:, 0], y, c='r', s=50, zorder=10)
pylab.plot(X_, y_, 'r', lw=1, zorder=9, label="true function")
pylab.xlim(-7.5, 7.5)
pylab.ylim(-4, 3)
pylab.legend(loc="best")
pylab.title("Posterior samples")

# For comparison a stationary kernel
kernel = C(1.0, (0.01, 100)) * RBF(0.1)
gp_stationary = GaussianProcessRegressor(kernel=kernel, alpha=1e-5,
                                         n_restarts_optimizer=1)
gp_stationary.fit(X, y)

pylab.subplot(3, 2, 6)
y_mean, y_cov = gp_stationary.predict(X_[:, None], return_cov=True)
pylab.plot(X_, y_mean, 'k', lw=3, zorder=9, label="mean")
pylab.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                   y_mean + np.sqrt(np.diag(y_cov)),
                   alpha=0.5, color='k')
y_samples = gp_stationary.sample_y(X_[:, None], 10)
pylab.plot(X_, y_samples, color='b', lw=1)
pylab.plot(X_, y_samples[:, 0], color='b', lw=1, label="samples") # just for the legend
pylab.scatter(X[:, 0], y, c='r', s=50, zorder=10)
pylab.plot(X_, y_, 'r', lw=1, zorder=9, label="true function")
pylab.xlim(-7.5, 7.5)
pylab.ylim(-4, 3)
pylab.legend(loc="best")
pylab.title("Stationary kernel")

pylab.tight_layout()
pylab.show()