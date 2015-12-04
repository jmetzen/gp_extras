Installation
============

You will need the new Gaussian process implementation from sklearn. For this, install the current development version of scikit-learn (or sklearn version 0.18 once this is available)

    git clone git@github.com:scikit-learn/scikit-learn.git
    cd sklearn
    sudo python setup.py install

Install `kernel_extras`

    git clone git@github.com:jmetzen/kernels_extra.git
    cd kernel_extras
    sudo python setup.py install

Take a look at the examples in the `examples` subdirectory.
