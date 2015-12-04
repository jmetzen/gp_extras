Installation
============

You will need the new Gaussian process implementation from scikit-learn. For this, install the current development version of scikit-learn (or scikit-learn version 0.18 once this is available)

    git clone git@github.com:scikit-learn/scikit-learn.git
    cd sklearn
    sudo python setup.py install

Now, you can install `kernel_extras`

    git clone git@github.com:jmetzen/gp_extras.git
    cd gp_extras/gp_extras
    sudo python setup.py install

Take a look at the examples in the `examples` subdirectory.
