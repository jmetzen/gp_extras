# Authors: Jan Hendrik Metzen <janmetzen@mailbox.org>
#
# License: BSD 3 clause
"""
==============================================================================
Illustration how ManifoldKernel can exploit data on lower-dimensional manifold
==============================================================================

This example illustrates how the ManifoldKernel allows exploiting when the
function to be learned has a lower effective input dimensionality (2d in the
example) than the actual observed data (5d in the example). For this, a
non-linear mapping (represented using an MLP) from data space onto
manifold is learned. A stationary GP is used to learn the function on this
manifold.

In the example, the ManifoldKernel is able to nearly perfectly recover the
original square 2d structure of the function input space and correspondingly
learns to model the target function better than a stationary, anisotropic GP
in the 5d data space.
"""
print __doc__

import numpy as np

import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, ConstantKernel as C
from sklearn_kernels import ManifoldKernel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve


np.random.seed(0)

n_samples = 100
n_features = 5
n_dim_manifold = 2
n_hidden = 3

# Generate data
def f(X_nn):  # target function
    return np.sqrt(np.abs(X_nn[:, 0] * X_nn[:, 1]))

X_ = np.random.uniform(-5, 5, (n_samples, n_dim_manifold))  # data on manifold
A = np.random.random((n_dim_manifold, n_features))  # mapping from manifold to data space
X = X_.dot(A)  # X are the observed values
y = f(X_)  # Generate target values by applying function to manifold

# Gaussian Process with anisotropic RBF kernel
kernel = C(1.0, (1e-10, 100)) * RBF([1] * n_features,
                                    [(0.1, 100.0)] * n_features) \
    + WhiteKernel(1e-3, (1e-10, 1e-1))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                              n_restarts_optimizer=3)

# Gaussian Process with Manifold kernel (using an isotropic RBF kernel on
# manifold for learning the target function)
# Use an MLP with one hidden-layer for the mapping from data space to manifold
architecture=((n_features, n_hidden, n_dim_manifold),)
kernel_nn = C(1.0, (1e-10, 100)) \
    * ManifoldKernel.construct(base_kernel=RBF(0.1, (1.0, 100.0)),
                               architecture=architecture,
                               transfer_fct="tanh", max_nn_weight=1.0) \
    + WhiteKernel(1e-3, (1e-10, 1e-1))
gp_nn = GaussianProcessRegressor(kernel=kernel_nn, alpha=0,
                                 n_restarts_optimizer=3)

# Fit GPs and create scatter plot on test data
gp.fit(X, y)
gp_nn.fit(X, y)

print "Initial kernel: %s" % gp_nn.kernel
print "Log-marginal-likelihood: %s" \
    % gp_nn.log_marginal_likelihood(gp_nn.kernel.theta)

print "Learned kernel: %s" % gp_nn.kernel_
print "Log-marginal-likelihood: %s" \
    % gp_nn.log_marginal_likelihood(gp_nn.kernel_.theta)

X_test_ = np.random.uniform(-5, 5, (1000, n_dim_manifold))
X_nn_test = X_test_.dot(A)
y_test = f(X_test_)
plt.figure(0)
plt.scatter(y_test, gp.predict(X_nn_test), c='b', label="GP RBF")
plt.scatter(y_test, gp_nn.predict(X_nn_test), c='r', label="GP NN")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.legend(loc=0)
plt.title("Scatter plot on test data")

print "RMSE of stationary anisotropic kernel: %s" \
    % mean_squared_error(y_test, gp.predict(X_nn_test))
print "RMSE of stationary anisotropic kernel: %s" \
    % mean_squared_error(y_test, gp_nn.predict(X_nn_test))

plt.figure(1)
X_gp_nn_test = gp_nn.kernel_.k1.k2._project_manifold(X_nn_test)
plt.scatter(X_gp_nn_test[:, 0], X_gp_nn_test[:, 1], c=y_test)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Function value', rotation=270)
plt.xlabel("Manifold dimension 1")
plt.ylabel("Manifold dimension 2")
plt.title("Learned 2D Manifold")
plt.show()
