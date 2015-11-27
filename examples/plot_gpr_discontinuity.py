
import numpy as np
import pylab

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn_kernels import ManifoldKernel

np.random.seed(1)

# Specify Gaussian Process
kernel = C(1.0, (0.01, 100)) \
    * ManifoldKernel.construct(base_kernel=RBF(0.1), architecture=((1, 2),),
                               transfer_fct="tanh", max_nn_weight=1)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5,
                              n_restarts_optimizer=10)

X_ = np.linspace(-7.5, 7.5, 100)

# Visualization of prior
X_nn = gp.kernel.k2._project_manifold(X_[:, None])
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
y_samples = gp.sample_y(X_[:, None], 10)
pylab.plot(X_, y_samples, color='b', lw=1)
pylab.xlim(-7.5, 7.5)
pylab.ylim(-3, 3)
pylab.title("Prior samples")


# Generate data and fit GP
X = np.random.uniform(-5, 5, 40)[:, None]
y = np.sin(X[:, 0]) + (X[:, 0] > 0)
gp.fit(X, y)

# Visualization of posterior
X_nn = gp.kernel_.k2._project_manifold(X_[:, None])

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
y_samples = gp.sample_y(X_[:, None], 10)
pylab.plot(X_, y_samples, color='b', lw=1)
pylab.scatter(X[:, 0], y, c='r', s=50, zorder=10)
pylab.xlim(-7.5, 7.5)
#pylab.ylim(-3, 3)
pylab.title("Posterior samples")

pylab.show()