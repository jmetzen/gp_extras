#! /usr/bin/env python
# adapted from sklearn

import sys
import os
import shutil
import glob
from distutils.command.clean import clean as Clean


if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins

class CleanCommand(Clean):
    description = "Remove build directories"

    def run(self):
        Clean.run(self)
        if os.path.exists('build'):
            # setup.py creates build/lib.* and build/tmp.*
            # everything else in 'build' is created by cmake and should be
            # ignored
            for path in glob.glob("build/lib.*"):
                print("Removing '" + path +"'")
                shutil.rmtree(path)
            for path in glob.glob("build/temp.*"):
                print("Removing '" + path + "'")
                shutil.rmtree(path)


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration("gp_extras", parent_package, top_path)

    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage("kernels")
    return config


def setup_package():
    metadata = dict(name="gp_extras",
                    maintainer="Jan Hendrik Metzen",
                    maintainer_email="janmetzen@mailbox.org",
                    description="Additional resources for sklearn Gaussian processes",
                    license="BSD 3-clause",
                    cmdclass={'clean': CleanCommand},
                    )

    if (len(sys.argv) >= 2
            and ('--help' in sys.argv[1:] or sys.argv[1]
                 in ('--help-commands', 'egg_info', '--version', 'clean'))):
        try:
            from setuptools import setup
            #install_requires is only available if setuptools is used
            metadata["install_requires"] = ["scikit-learn"]
        except ImportError:
            from distutils.core import setup

    else:
        from numpy.distutils.core import setup

        metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
