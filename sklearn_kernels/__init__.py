""" Additional kernels that can be used with sklearn's GP module. """

# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#
# License: BSD 3 clause

from .kernels_non_stationary import ManifoldKernel, LocalLengthScalesKernel

__all__ = ['ManifoldKernel', 'LocalLengthScalesKernel']
