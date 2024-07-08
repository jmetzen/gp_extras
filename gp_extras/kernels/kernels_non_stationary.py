# Authors: Jan Hendrik Metzen <janmetzen@mailbox.org>
#
# License: BSD 3 clause

""" Non-stationary kernels that can be used with sklearn's GP module. """

import numpy as np

from scipy.special import gamma, kv

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import Kernel, _approx_fprime, Hyperparameter, RBF


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

    def __init__(self, w, w_bounds, base_kernel, architecture, theta_nn_size,
                 transfer_fct="tanh", max_nn_weight=5.0):
        self.w = w
        self.w_bounds = w_bounds
        self.base_kernel = base_kernel
        self.architecture = architecture
        self.theta_nn_size = theta_nn_size
        self.transfer_fct = transfer_fct
        self.max_nn_weight = max_nn_weight

        self.hyperparameter_w = \
                Hyperparameter("w", "numeric", self.w_bounds,
                               self.w.shape[0])

    @classmethod
    def construct(cls, base_kernel, architecture, transfer_fct="tanh",
                  max_nn_weight=5.0):
        n_outputs, theta_nn_size = determine_network_layout(architecture)

        w = np.array(list(np.random.uniform(-max_nn_weight, max_nn_weight,
                                            theta_nn_size))
                     + list(base_kernel.theta))
        wL = [-max_nn_weight] * theta_nn_size \
            + list(base_kernel.bounds[:, 0])
        wU = [max_nn_weight] * theta_nn_size \
            + list(base_kernel.bounds[:, 1])
        w_bounds = np.vstack((wL, wU)).T
        return cls(w, w_bounds, base_kernel=base_kernel,
                   architecture=architecture, theta_nn_size=theta_nn_size,
                   transfer_fct=transfer_fct, max_nn_weight=max_nn_weight)

    @property
    def theta(self):
        return self.w

    @theta.setter
    def theta(self, theta):
        self.w = np.asarray(theta, dtype=float)
        self.base_kernel.theta = theta[self.theta_nn_size:]

    @property
    def bounds(self):
        return self.w_bounds

    @bounds.setter
    def bounds(self, bounds):
        self.w_bounds = bounds

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
                    return self.clone_with_theta(theta)(X, Y)
                return K, _approx_fprime(self.theta, f, 1e-5)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            Y_nn = self._project_manifold(Y)
            return self.base_kernel(X_nn, Y_nn)

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
        return np.diag(self(X)) # XXX

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return False

    def _project_manifold(self, X, w=None):
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

        if w is None:
            w = self.w

        y = []
        for subnet in self.architecture:
            y.append(X[:, :subnet[0]])
            for layer in range(len(subnet) - 1):
                W = w[:subnet[layer]*subnet[layer+1]]
                W = W.reshape((subnet[layer], subnet[layer+1]))
                b = w[subnet[layer]*subnet[layer+1]:
                      (subnet[layer]+1)*subnet[layer+1]]
                a = y[-1].dot(W) + b
                y[-1] = transfer_fct(a)

                # chop off weights of this layer
                w = w[(subnet[layer]+1)*subnet[layer+1]:]

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
    def __init__(self, X_, nu=1.5,
                 isotropic=True, theta0=1e-1, thetaL=1e-3, thetaU=1.0,
                 l_isotropic=True, theta_l_0=1e-1, theta_l_L=1e-3, theta_l_U=1e1,
                 l_samples=10, l_0=1.0, l_L=0.1, l_U=10.0):
        self.X_ = X_
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

        # Determine how many entries in theta belong to the different
        # categories (used later for parsing theta)
        self.theta_gp_size = 1 if self.isotropic else X_.shape[1]
        self.theta_l_size = 1 if self.l_isotropic else X_.shape[1]
        self.theta_size = \
            self.theta_gp_size + self.theta_l_size + self.l_samples

        self.weights = [theta0] * self.theta_gp_size
        weightsL = [thetaL] * self.theta_gp_size
        weightsU = [thetaU] * self.theta_gp_size

        self.weights += [theta_l_0] * self.theta_l_size
        weightsL += [theta_l_L] * self.theta_l_size
        weightsU += [theta_l_U] * self.theta_l_size

        self.weights += [l_0] * self.l_samples
        weightsL += [l_L] * self.l_samples
        weightsU += [l_U] * self.l_samples

        self.weights_bounds = np.vstack((weightsL, weightsU)).T
        self.theta = np.log(self.weights)

        self.hyperparameter_weights = \
                Hyperparameter("weights", "numeric", self.weights_bounds,
                               len(self.weights))

    @classmethod
    def construct(cls, X, nu=1.5, isotropic=True,
                  theta0=1e-1, thetaL=1e-3, thetaU=1.0, l_isotropic=True,
                  theta_l_0=1e-1, theta_l_L=1e-3, theta_l_U=1e1,
                  l_samples=10, l_0=1.0, l_L=0.1, l_U=10.0):
        assert X is not None
        # Select points on which length-scale GP is defined
        if X.shape[0] > l_samples:
            kmeans = KMeans(n_clusters=l_samples)
            X_ = kmeans.fit(X).cluster_centers_
        else:
            X_ = np.asarray(X)
            l_samples = X.shape[0]
        return cls(X_=X_, nu=nu, isotropic=isotropic, theta0=theta0,
                   thetaL=thetaL, thetaU=thetaU, l_isotropic=l_isotropic,
                   theta_l_0=theta_l_0, theta_l_L=theta_l_L, theta_l_U=theta_l_U,
                   l_samples=l_samples, l_0=l_0, l_L=l_L, l_U=l_U)

    @property
    def theta(self):
        return np.log(self.weights)

    @theta.setter
    def theta(self, weights):
        self.weights = np.exp(np.asarray(weights, dtype=float))

        # Parse weights into its components
        self.theta_gp, self.theta_l, self.length_scales = \
            self._parse_weights(self.weights)

        # Train length-scale Gaussian Process
        kernel = RBF(self.theta_l, length_scale_bounds="fixed")
        self.gp_l = GaussianProcessRegressor(kernel=kernel)
        self.gp_l.fit(self.X_, np.log10(self.length_scales))

    @property
    def bounds(self):
        return np.log(self.weights_bounds)

    @bounds.setter
    def bounds(self, bounds):
        self.weights_bounds = np.exp(bounds)

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
        if d.ndim > 1 and self.theta_gp.size == d.shape[-1]:
            activation = \
                np.sum(self.theta_gp.reshape(1, -1) * d ** 2, axis=1)
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
                # XXX: computed gradient analytically?
                def f(theta):  # helper function
                    return self.clone_with_theta(theta)(X, Y)
                return K, _approx_fprime(self.weights, f, 1e-7)

    def _parse_weights(self, weights):
        """ Parse parameter vector weights into its components.

        Parameters
        ----------
        weights : array_like
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
        weights = np.asarray(weights, dtype=float)

        assert (weights.size == self.theta_size), \
            "weights does not have the expected size (expected: %d, " \
            "actual size %d). Expected: %d entries for main GP, " \
            "%d entries for length-scale GP, %d entries containing the "\
            "length scales, and %d entries for nu." \
            % (self.theta_size, weights.size, self.theta_gp_size,
               self.theta_l_size, self.l_samples, self.nu_size)

        # Split theta in its components
        theta_gp = weights[:self.theta_gp_size]
        theta_l = \
            weights[self.theta_gp_size:][:self.theta_l_size]
        length_scales = \
            weights[self.theta_gp_size+self.theta_l_size:][:self.l_samples]

        return theta_gp, theta_l, length_scales

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
        return np.diag(self(X)) # XXX

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return False

    def __repr__(self):
        return "{0}(theta_gp={1}, theta_l={2}, length_scales={3})".format(
            self.__class__.__name__, self.theta_gp, self.theta_l,
            self.length_scales)


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
        assert prototypes.shape[0] == sigma_2.shape[0]
        self.prototypes = prototypes

        self.sigma_2 = np.asarray(sigma_2)
        self.sigma_2_bounds = sigma_2_bounds

        self.gamma = gamma
        self.gamma_bounds = gamma_bounds

        self.hyperparameter_sigma_2 = \
                Hyperparameter("sigma_2", "numeric", self.sigma_2_bounds,
                               self.sigma_2.shape[0])

        self.hyperparameter_gamma = \
                Hyperparameter("gamma", "numeric", self.gamma_bounds)

    @classmethod
    def construct(cls, prototypes, sigma_2=1.0, sigma_2_bounds=(0.1, 10.0),
                  gamma=1.0, gamma_bounds=(1e-2, 1e2)):
        prototypes = np.asarray(prototypes)
        if prototypes.shape[0] > 1 and len(np.atleast_1d(sigma_2)) == 1:
            sigma_2 = np.repeat(sigma_2, prototypes.shape[0])
            sigma_2_bounds = np.vstack([sigma_2_bounds] *prototypes.shape[0])
        return cls(prototypes, sigma_2, sigma_2_bounds, gamma, gamma_bounds)

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
        prototypes_std = self.prototypes.std(0)
        n_prototypes = self.prototypes.shape[0]
        n_gradient_dim = \
            n_prototypes + (0 if self.hyperparameter_gamma.fixed else 1)

        X = np.atleast_2d(X)
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K= np.eye(X.shape[0]) * self.diag(X)
            if eval_gradient:
                K_gradient = \
                    np.zeros((K.shape[0], K.shape[0], n_gradient_dim))
                K_pairwise = \
                    pairwise_kernels(self.prototypes / prototypes_std,
                                     X / prototypes_std,
                                     metric="rbf", gamma=self.gamma)
                for i in range(n_prototypes):
                    for j in range(K.shape[0]):
                        K_gradient[j, j, i] = \
                            self.sigma_2[i] * K_pairwise[i, j] \
                            / K_pairwise[:, j].sum()
                if not self.hyperparameter_gamma.fixed:
                    # XXX: Analytic expression for gradient?
                    def f(gamma):  # helper function
                        theta = self.theta.copy()
                        theta[-1] = gamma[0]
                        return self.clone_with_theta(theta)(X, Y)
                    K_gradient[:, :, -1] = \
                        _approx_fprime([self.theta[-1]], f, 1e-5)[:, :, 0]
                return K, K_gradient
            else:
                return K
        else:
            K = np.zeros((X.shape[0], Y.shape[0]))
            return K   # XXX: similar entries?

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return False

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
        prototypes_std = self.prototypes.std(0)
        n_prototypes = self.prototypes.shape[0]

        # kernel regression of noise levels
        K_pairwise = \
            pairwise_kernels(self.prototypes / prototypes_std,
                             X / prototypes_std,
                             metric="rbf", gamma=self.gamma)

        return (K_pairwise * self.sigma_2[:, None]).sum(axis=0) \
                / K_pairwise.sum(axis=0)

    def __repr__(self):
        return "{0}(sigma_2=[{1}], gamma={2})".format(self.__class__.__name__,
            ", ".join(map("{0:.3g}".format, self.sigma_2)), self.gamma)
