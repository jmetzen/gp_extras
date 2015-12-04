Summary
-------

Starting from version 0.18 (already available in the post-0.17 master branch), scikit-learn will ship a completely revised [Gaussian process module](http://scikit-learn.org/dev/modules/gaussian_process.html), supporting among other things kernel engineering. While scikit-learn only ships the most [common kernels](http://scikit-learn.org/dev/modules/gaussian_process.html#kernels-for-gaussian-processes), this project contains some more advanced, non-standard kernels that can seamlessly be used with scikit-learns [GaussianProcessRegressor](http://scikit-learn.org/dev/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor). The following kernels are included at the moment:
 * **ManifoldKernel**: Non-stationary correlation model based on manifold learning. This non-stationary kernel consists internally of two parts: a mapping from the actual data space onto a manifold and a stationary kernel on this manifold. The mapping is realized by a neural network whose architecture can be specified externally. The parameters of this network are learned along with the length scales of the Gaussian process, typically such that the marginal likelihood or the posterior probability of the GP are maximized. Any common stationary kernel can then be used on top of this manifold.
 * **LocalLengthScalesKernel**: Non-stationary kernel based on local smoothness estimates. This non-stationary correlation model learns internally point estimates of local smoothness using a second-level Gaussian Process. For this, it selects a subset of the training data and learns length-scales at this specific points. These length scales are generalized using the second-level Gaussian Process. Furthermore, global (isotropic or anisotropic) length scales are learned for both the top-level GP and the length-scale GP.
 * **HeteroscedasticKernel**: Kernel which learns a heteroscedastic noise model. This kernel learns for a set of prototypes values from the data space explicit noise levels. These exemplary noise levels are then generalized to the entire data space by means for kernel regression.

For examles on how these kernels can be used and where they might be useful, please take a look at the examples in the `examples` subdirectory.

Installation
------------

You will need the new Gaussian process implementation from scikit-learn. For this, install the current development version of scikit-learn (or scikit-learn version 0.18 once this is available)

    git clone git@github.com:scikit-learn/scikit-learn.git
    cd sklearn
    sudo python setup.py install

Now, you can install `kernel_extras`

    git clone git@github.com:jmetzen/gp_extras.git
    cd gp_extras/gp_extras
    sudo python setup.py install

