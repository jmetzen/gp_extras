""" Non-stationary kernels that can be used with sklearn's GP module. """

import numpy as np

from scipy.special import gamma, kv

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.gaussian_process.kernels import Kernel, _approx_fprime


class ManifoldKernel(Kernel):
    """ Non-stationary correlation model based on manifold learning.

    This non-stationary kernel consists internally of two parts:
    a mapping from the actual data space onto a manifold and a stationary
    kernel on this manifold. The mapping is realized by a neural
    network whose architecture can be specified externally. The parameters of
    this network are learned along with the length scales of the Gaussian
    process, typically such that the marginal likelihood or the posterior
    probability of the GP are maximized. Any common stationary kernel
    can then be used on top of this manifold.

    Parameters
    ----------
    base_kernel: Kernel
        The top-level, stationary kernel returning

    architecture: sequence of tuples
        Defines the structure of the internal neural network architecture
        mapping the data from the original data space onto a manifold. Note
        that different data dimensions can be processed by different networks
        and that the networks can have different number of layers. For
        instance, the architecture ((1, 2),(2, 4, 5)) would map a 3-dimensional
        input space onto a 7-dimensional manifold. For this, the first input
        dimension would be processed by the network (1, 2) with 1 inputs,
        2 outputs, and no hidden layer yielding the first two manifold
        dimensions. The other two input dimensions would be processed by a
        network (2, 4, 5) with 2 inputs, 4 hidden units, and 5 outputs
        yielding the remaining five manifold dimensions.

    transfer_fct: str, default="tanh"
        The transfer function used in the hidden and output units. Supported
        are "tanh" and the rectified linear unit ("relu"). Defaults is "tanh"

    max_nn_weight: float, default=5.0
        The maximum absolute value a weight of the neural network might have.

    .. seealso::

    "Manifold Gaussian Process for Regression",
    Roberto Calandra, Jan Peters, Carl Edward Rasmussen, Marc Peter Deisenroth,
    http://arxiv.org/abs/1402.5876
    """

    def __init__(self, base_kernel, architecture, transfer_fct="tanh",
                 max_nn_weight=5.0):
        self.base_kernel = base_kernel

        self.architecture = architecture
        self.transfer_fct = transfer_fct
        self.max_nn_weight = max_nn_weight

        n_outputs, self.theta_nn_size = determine_network_layout(architecture)

        theta0 = \
            list(np.random.uniform(-max_nn_weight, max_nn_weight,
                                   self.theta_nn_size)) \
                + list(self.base_kernel.theta)
        thetaL = [-max_nn_weight] * self.theta_nn_size \
            + list(self.base_kernel.bounds[:, 0])
        thetaU = [max_nn_weight] * self.theta_nn_size \
            + list(self.base_kernel.bounds[:, 1])

        self.params = np.array(theta0)
        self.params_bounds = np.vstack((thetaL, thetaU)).T

        self.theta_vars = [("params", len(self.params))]

    @property
    def theta(self):
        return self.params

    @theta.setter
    def theta(self, theta):
        self.params = np.asarray(theta, dtype=np.float)
        self.base_kernel.theta = theta[self.theta_nn_size:]
        #if self.theta.ndim == 2:
        #    self.theta = self.theta[:, 0]

        ## XXX:
        #if np.any(self.theta == 0):
        #    self.theta[np.where(self.theta == 0)] \
        #        += np.random.random((self.theta == 0).sum()) * 2e-5 - 1e-5

    @property
    def bounds(self):
        return self.params_bounds

    @bounds.setter
    def bounds(self, bounds):
        1 / 0

    def __call__(self, X, Y=None, eval_gradient=False):
        X_nn = self._project_manifold(X)
        if Y is None:
            K = self.base_kernel(X_nn)
            if not eval_gradient:
                return K
            else:
                # approximate gradient numerically
                # XXX: Analytic expression for gradient based on chain rule and
                #      backpropagation?
                def f(theta):  # helper function
                    # return self.clone_with_theta(theta)(X, Y)
                    import copy  # XXX: Avoid deepcopy
                    kernel = copy.deepcopy(self)
                    kernel.theta = theta
                    return kernel(X)
                return K, _approx_fprime(self.theta, f, 1e-10)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            Y_nn = self._project_manifold(Y)
            return self.base_kernel(X_nn, Y_nn)

    def _project_manifold(self, X, theta=None):
        # Lazily fetch transfer function (to keep object pickable)
        if self.transfer_fct == "tanh":
            transfer_fct = np.tanh
        elif self.transfer_fct == "sin":
            transfer_fct = np.sin
        elif self.transfer_fct == "relu":
            transfer_fct = lambda x: np.maximum(0, x)
        elif self.transfer_fct == "linear":
            transfer_fct = lambda x: x
        elif hasattr(self.transfer_fct, "__call__"):
            transfer_fct = self.transfer_fct

        if theta is None:
            theta = self.theta

        y = []
        for subnet in self.architecture:
            y.append(X[:, :subnet[0]])
            for layer in range(len(subnet) - 1):
                W = theta[:subnet[layer]*subnet[layer+1]]
                W = W.reshape((subnet[layer], subnet[layer+1]))
                b = theta[subnet[layer]*subnet[layer+1]:
                                 (subnet[layer]+1)*subnet[layer+1]]
                a = y[-1].dot(W) + b
                y[-1] = transfer_fct(a)

                # chop off weights of this layer
                theta = theta[(subnet[layer]+1)*subnet[layer+1]:]

            X = X[:, subnet[0]:]  # chop off used input dimensions

        return np.hstack(y)


def determine_network_layout(architecture):
    """ Determine number of outputs and params of given architecture."""
    n_outputs = 0
    n_params = 0
    for subnet in architecture:
        for layer in range(len(subnet) - 1):
            n_params += (subnet[layer] + 1) * subnet[layer+1]

        n_outputs += subnet[-1]

    return n_outputs, n_params


class LocalLengthScalesKernel(Kernel):
    """ Non-stationary kernel based on local smoothness estimates.

    This non-stationary correlation model learns internally point estimates of
    local smoothness using a second-level Gaussian Process. For this, it
    selects a subset of the training data and learns length-scales at this
    specific points. These length scales are generalized using the second-level
    Gaussian Process. Furthermore, global (isotropic or anisotropic) length
    scales are learned for both the top-level GP and the length-scale GP.

    The kernel of the second-level GP is an RBF kernel.

    Parameters
    ----------
    isotropic : bool, default=True
        Whether the global length-scales of the top-level GP are isotropic or
        anisotropic

    l_isotropic : bool, default=True
        Whether the global length-scales of the length-scale GP are isotropic
        or anisotropic

    l_samples: int, default=10
        How many datapoints from the training data are selected as support
        points for learning the length-scale GP

    .. seealso::

    "Nonstationary Gaussian Process Regression using Point Estimates of Local
    Smoothness", Christian Plagemann, Kristian Kersting, and Wolfram Burgard,
    ECML 2008
    """
    def __init__(self, X, nu=1.5,
                 isotropic=True, theta0=1e-1, thetaL=1e-3, thetaU=1.0,
                 l_isotropic=True, theta_l_0=1e-1, theta_l_L=1e-3, theta_l_U=1e1,
                 l_samples=10, l_0=1.0, l_L=0.1, l_U=10.0):
        self.nu = nu
        self.isotropic = isotropic
        self.theta0 = theta0
        self.thetaL = thetaL
        self.thetaU = thetaU
        self.l_isotropic = l_isotropic
        self.theta_l_0 = theta_l_0
        self.theta_l_L = theta_l_L
        self.theta_l_U = theta_l_U
        self.l_samples = l_samples
        self.l_0 = l_0
        self.l_L = l_L
        self.l_U = l_U

        assert X is not None
        self.X = np.asarray(X)
        if X.shape[0] > self.l_samples:
            kmeans = KMeans(n_clusters=self.l_samples)
            self.X_ = kmeans.fit(self.X).cluster_centers_
        else:
            self.X_ = np.asarray(self.X)

        # Determine how many entries in theta belong to the different
        # categories (used later for parsing theta)
        self.theta_gp_size = 1 if self.isotropic else X.shape[1]
        self.theta_l_size = 1 if self.l_isotropic else X.shape[1]
        self.theta_size = \
            self.theta_gp_size + self.theta_l_size + self.l_samples

        theta0 = [theta0] * self.theta_gp_size
        thetaL = [thetaL] * self.theta_gp_size
        thetaU = [thetaU] * self.theta_gp_size

        theta0 += [theta_l_0] * self.theta_l_size
        thetaL += [theta_l_L] * self.theta_l_size
        thetaU += [theta_l_U] * self.theta_l_size

        theta0 += [l_0] * self.l_samples
        thetaL += [l_L] * self.l_samples
        thetaU += [l_U] * self.l_samples

        self.params = theta0
        self.params_bounds = np.vstack((thetaL,  thetaU)).T

        self.theta_vars = [("params", len(self.params))]

    @property
    def theta(self):
        return self.params

    @theta.setter
    def theta(self, params):
        from . import GaussianProcessRegressor
        from .kernels import RBF

        self.params = np.asarray(params, dtype=np.float)

        # Parse theta into its components
        self.theta_gp, self.theta_l, self.length_scales = \
            self._parse_theta(self.params)

        # Train length-scale Gaussian Process
        kernel = RBF(self.theta_l)
        self.gp_l = GaussianProcessRegressor(kernel=kernel, optimizer=None)
        self.gp_l.fit(self.X_, np.log10(self.length_scales))

    def __call__(self, X, Y=None, eval_gradient=False):
        l_train = 10 ** self.gp_l.predict(X)

        # Prepare distances and length scale information for any pair of
        # datapoints, whose correlation shall be computed
        if Y is not None:
            # Get pairwise componentwise L1-differences to the input training
            # set
            d = Y[:, np.newaxis, :] - X[np.newaxis, :, :]
            d = d.reshape((-1, Y.shape[1]))
            # Predict length scales for query datapoints
            l_query = 10 ** self.gp_l.predict(Y)
            l = np.transpose([np.tile(l_train, len(l_query)),
                              np.repeat(l_query, len(l_train))])
        else:
            # No external datapoints given; auto-correlation of training set
            # is used instead
            d = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            d = d.reshape((-1, X.shape[1]))
            l = np.transpose([np.tile(l_train, len(l_train)),
                              np.repeat(l_train, len(l_train))])  # XXX: check

        # Compute general Matern kernel
        if d.ndim > 1 and self.theta_gp.size == d.ndim:
            activation = \
                np.sum(self.theta_gp.reshape(1, d.ndim) * d ** 2, axis=1)
        else:
            activation = self.theta_gp[0] * np.sum(d ** 2, axis=1)
        tmp = 0.5 * (l**2).sum(1)
        tmp2 = np.maximum(2*np.sqrt(self.nu * activation / tmp), 1e-5)
        k = np.sqrt(l[:, 0]) * np.sqrt(l[:, 1]) \
            / (gamma(self.nu) * 2**(self.nu - 1))
        k /= np.sqrt(tmp)
        k *= tmp2**self.nu * kv(self.nu, tmp2)

        # Convert correlations to 2d matrix
        if Y is not None:
            return k.reshape(-1, X.shape[0]).T
        else:  # exploit symmetry of auto-correlation
            K = k.reshape(X.shape[0], X.shape[0])
            if not eval_gradient:
                return K
            else:
                # approximate gradient numerically
                def f(theta):  # helper function
                    import copy  # XXX: Avoid deepcopy
                    kernel = copy.deepcopy(self)
                    kernel.theta = theta
                    return kernel(X)
                return K, _approx_fprime(self.params, f, 1e-5)

    def _parse_theta(self, theta):
        """ Parse parameter vector theta into its components.

        Parameters
        ----------
        theta : array_like
            An array containing all hyperparameters.

        Returns
        -------
        theta_gp : array_like
            An array containing the hyperparameters of the main GP.
        theta_l : array_like
            An array containing the hyperparameters of the length-scale GP.
        length_scales : array_like
            An array containing the length-scales for the length-scale GP.
        """
        theta = np.asarray(theta, dtype=np.float)

        assert (theta.size == self.theta_size), \
            "theta does not have the expected size (expected: %d, " \
            "actual size %d). Expected: %d entries for main GP, " \
            "%d entries for length-scale GP, %d entries containing the "\
            "length scales, and %d entries for nu." \
            % (self.theta_size, theta.size, self.theta_gp_size,
               self.theta_l_size, self.l_samples, self.nu_size)

        # Split theta in its components
        theta_gp = theta[:self.theta_gp_size]
        theta_l = \
            theta[self.theta_gp_size:][:self.theta_l_size]
        length_scales = \
            theta[self.theta_gp_size+self.theta_l_size:][:self.l_samples]

        return theta_gp, theta_l, length_scales


class HeteroscedasticKernel(Kernel):
    """Kernel which learns a heteroscedastic noise model.

    This kernel learns for a set of prototypes values from the data space
    explicit noise levels. These exemplary noise levels are then generalized to
    the entire data space by means for kernel regression.

    Parameters
    ----------
    prototypes : array-like, shape = (n_prototypes, n_X_dims)
        Prototypic samples from the data space for which noise levels are
        estimated.

    sigma_2 : float, default: 1.0
        Parameter controlling the initial noise level

    sigma_2_bounds : pair of floats >= 0, default: (0.1, 10.0)
        The lower and upper bound on sigma_2

    gamma : float, default: 1.0
        Length scale of the kernel regression on the noise level

    gamma_bounds : pair of floats >= 0, default: (1e-2, 1e2)
        The lower and upper bound on gamma
    """
    def __init__(self, prototypes, sigma_2=1.0, sigma_2_bounds=(0.1, 10.0),
                 gamma=1.0, gamma_bounds=(1e-2, 1e2)):
        assert len(prototypes) == len(sigma_2)
        self.prototypes = np.asarray(prototypes)
        self.prototypes_std = self.prototypes.std(0)
        self.n_prototypes = self.prototypes.shape[0]

        self.sigma_2 = np.asarray(sigma_2)
        self.sigma_2_bounds = sigma_2_bounds

        self.gamma = gamma
        self.gamma_bounds = gamma_bounds

        self.theta_vars = []
        if sigma_2_bounds is not "fixed":
            self.theta_vars.append(("sigma_2", self.n_prototypes))
        if gamma_bounds is not "fixed":
            self.theta_vars.append("gamma")

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K= np.eye(X.shape[0]) * self.diag(X)
            if eval_gradient:
                K_gradient = \
                    np.zeros((K.shape[0], K.shape[0], self.n_prototypes))
                K_pairwise = \
                    pairwise_kernels(self.prototypes / self.prototypes_std,
                                     X / self.prototypes_std,
                                     metric="rbf", gamma=self.gamma)
                for i in range(self.n_prototypes):
                    for j in range(K.shape[0]):
                        K_gradient[j, j, i] = \
                            self.sigma_2[i] * K_pairwise[i, j] \
                            / K_pairwise[:, j].sum()
                return K, K_gradient
            else:
                return K
        else:
            K = np.zeros((X.shape[0], Y.shape[0]))
            return K   # XXX: similar entries?

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        # kernel regression of noise levels
        K_pairwise = \
            pairwise_kernels(self.prototypes / self.prototypes_std,
                             X / self.prototypes_std,
                             metric="rbf", gamma=self.gamma)

        return (K_pairwise * self.sigma_2[:, None]).sum(axis=0) \
                / K_pairwise.sum(axis=0)

    def __repr__(self):
        return "{0}(sigma_2=[{1}], gamma={2})".format(self.__class__.__name__,
            ", ".join(map("{0:.3g}".format, self.sigma_2)), self.gamma)
