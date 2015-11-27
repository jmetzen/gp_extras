
import numpy as np
import pylab

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels_non_stationary import ManifoldKernel

np.random.seed(0)

# Specify Gaussian Process
kernel = (1e-10, 1.0, 100) \
    * ManifoldKernel(base_kernel=RBF(0.1), architecture=((1, 2),),
                     transfer_fct="tanh", max_nn_weight=1)
gp = GaussianProcessRegressor(kernel=kernel)


X_ = np.linspace(-7.5, 7.5, 100)

# Visualization of prior
X_nn = kernel.k2._project_manifold(X_[:, None])
pylab.figure(0, figsize=(8, 8))
pylab.subplot(2, 1, 1)
for i in range(X_nn.shape[1]):
    pylab.plot(X_, X_nn[:, i])
pylab.title("Prior mapping to manifold")

pylab.subplot(2, 1, 2)
y_mean, y_cov = gp.predict(X_[:, None], return_cov=True)
pylab.plot(X_, y_mean, 'k', lw=3, zorder=9)
pylab.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                   y_mean + np.sqrt(np.diag(y_cov)),
                   alpha=0.5, color='k')
y_samples = gp.sample(X_[:, None], 10)
pylab.plot(X_, y_samples, color='b', lw=1)
pylab.xlim(-7.5, 7.5)
pylab.ylim(-3, 3)
pylab.title("Prior samples")


# Generate data and fit GP
X = np.random.uniform(-5, 5, 30)[:, None]
y = np.sin(X[:, 0]) + (X[:, 0] > 0)
old_params = kernel.params
gp.fit(X, y)

# Visualization of posterior
X_nn = kernel.k2._project_manifold(X_[:, None])

pylab.figure(1, figsize=(8, 8))
pylab.subplot(2, 1, 1)
for i in range(X_nn.shape[1]):
    pylab.plot(X_, X_nn[:, i])
pylab.title("Posterior mapping to manifold")

pylab.subplot(2, 1, 2)
y_mean, y_cov = gp.predict(X_[:, None], return_cov=True)
pylab.plot(X_, y_mean, 'k', lw=3, zorder=9)
pylab.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                   y_mean + np.sqrt(np.diag(y_cov)),
                   alpha=0.5, color='k')
y_samples = gp.sample(X_[:, None], 10)
pylab.plot(X_, y_samples, color='b', lw=1)
pylab.scatter(X[:, 0], y, c='r', s=50, zorder=10)
pylab.xlim(-7.5, 7.5)
#pylab.ylim(-3, 3)
pylab.title("Posterior samples")

pylab.show()