# Authors: Jan Hendrik Metzen <janmetzen@mailbox.org>
#
# License: BSD 3 clause
"""
=====================================================================
Illustration how the LocalLengthScalesKernel can with discontinuities
=====================================================================

The LocalLengthScalesKernel allows learning local length scales in the data
space and thus can identify areas, in which broader and more narrow
generalization is appropriate. This is illustrated on a simple sinusoidal
function with a discontinuity at X=0. Because of this discontinuity, a
stationary Matern kernel is forced to reduce the global length-scale
considerably. A LocalLengthScalesKernel, on the other hand, needs to reduce
only the length-scale close to the discontinuity, and achieves a considerably
larger log-marginal-likelihood accordingly.

The example illustrates also how a custom optimizer based on differential
evolution can be used for GP hyperparameter-tuning. This is required here
because the log-marginal-likelihood for the LocalLengthScalesKernel is highly
multi-modal, which is problematic for gradient-based methods like L-BFGS.
"""
print __doc__

import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import ConstantKernel as C, Matern
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve

from gp_extras.kernels import LocalLengthScalesKernel

np.random.seed(42)

n_samples = 50

# Generate data
def f(X):  # target function
    return np.sin(5*X) + np.sign(X)

X = np.random.uniform(-1, 1, (n_samples, 1))  # data
y = f(X)[:, 0]

# Define custom optimizer for hyperparameter-tuning of non-stationary kernel
def de_optimizer(obj_func, initial_theta, bounds):
    res = differential_evolution(lambda x: obj_func(x, eval_gradient=False),
                                 bounds, maxiter=20, disp=False, polish=False)
    return res.x, obj_func(res.x, eval_gradient=False)

# Specify stationary and non-stationary kernel
kernel_matern = C(1.0, (1e-10, 1000)) \
    * Matern(length_scale_bounds=(1e-1, 1e3), nu=1.5)
gp_matern = GaussianProcessRegressor(kernel=kernel_matern)

kernel_lls = C(1.0, (1e-10, 1000)) \
  * LocalLengthScalesKernel.construct(X, l_L=0.1, l_U=2.0, l_samples=5)
gp_lls = GaussianProcessRegressor(kernel=kernel_lls, optimizer=de_optimizer)

# Fit GPs
gp_matern.fit(X, y)
gp_lls.fit(X, y)

print "Learned kernel Matern: %s" % gp_matern.kernel_
print "Log-marginal-likelihood Matern: %s" \
    % gp_matern.log_marginal_likelihood(gp_matern.kernel_.theta)


print "Learned kernel LLS: %s" % gp_lls.kernel_
print "Log-marginal-likelihood LLS: %s" \
    % gp_lls.log_marginal_likelihood(gp_lls.kernel_.theta)

# Compute GP mean and standard deviation on test data
X_ = np.linspace(-1, 1, 500)

y_mean_lls, y_std_lls = gp_lls.predict(X_[:, np.newaxis], return_std=True)
y_mean_matern, y_std_matern = \
    gp_matern.predict(X_[:, np.newaxis], return_std=True)

plt.figure(figsize=(7, 7))
plt.subplot(2, 1, 1)
plt.plot(X_, f(X_), c='k', label="true function")
plt.scatter(X[:, 0], y, color='k', label="samples")
plt.plot(X_, y_mean_lls, c='r', label="GP LLS")
plt.fill_between(X_, y_mean_lls - y_std_lls, y_mean_lls + y_std_lls,
                 alpha=0.5, color='r')
plt.plot(X_, y_mean_matern, c='b', label="GP Matern")
plt.fill_between(X_, y_mean_matern - y_std_matern, y_mean_matern + y_std_matern,
                 alpha=0.5, color='b')
plt.legend(loc="best")
plt.title("Comparison of learned models")
plt.xlim(-1, 1)

plt.subplot(2, 1, 2)
plt.plot(X_, gp_lls.kernel_.k2.theta_gp
             * 10**gp_lls.kernel_.k2.gp_l.predict(X_[:, np.newaxis]),
         c='r', label="GP LLS")
plt.plot(X_, np.ones_like(X_) * gp_matern.kernel_.k2.length_scale,
         c='b', label="GP Matern")
plt.xlim(-1, 1)
plt.ylabel("Length-scale")
plt.legend(loc="best")
plt.title("Comparison of length scales")
plt.tight_layout()
plt.show()
