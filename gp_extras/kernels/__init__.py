""" Additional kernels that can be used with sklearn's GP module. """

# Authors: Jan Hendrik Metzen <janmetzen@mailbox.org>
#
# License: BSD 3 clause

from .kernels_non_stationary \
    import ManifoldKernel, LocalLengthScalesKernel, HeteroscedasticKernel

__all__ = ['ManifoldKernel', 'LocalLengthScalesKernel',
           'HeteroscedasticKernel']
