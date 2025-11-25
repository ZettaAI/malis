"""Setup script for the MALIS package.

This script builds the malis package with Cython extensions for efficient
computation of MALIS loss and related graph operations.
"""

import os
import sys
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


def get_include_dirs():
    """Get include directories for building C extensions."""
    include_dirs = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "malis"),
    ]

    # Add Python include directories
    python_inc = sysconfig.get_path('include')
    if python_inc:
        include_dirs.append(python_inc)
        include_dirs.append(os.path.dirname(python_inc))

    return include_dirs


def get_library_dirs():
    """Get library directories for building C extensions."""
    library_dirs = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "malis"),
    ]

    # Add system library directory
    libdir = sysconfig.get_config_var("LIBDIR")
    if libdir:
        library_dirs.append(libdir)

    return library_dirs


class build_ext(_build_ext):
    """Custom build_ext command that adds numpy include directory."""

    def finalize_options(self):
        """Finalize options, adding numpy include directory."""
        _build_ext.finalize_options(self)
        # Add numpy include directory
        import numpy
        self.include_dirs.append(numpy.get_include())


setup(
    name='malis',
    version='1.0',
    description='MALIS segmentation loss function',
    url='https://github.com/TuragaLab/malis',
    author='Srinivas Turaga',
    author_email='turagas@janelia.hhmi.org',
    cmdclass={'build_ext': build_ext},
    license='MIT',
    install_requires=[
        'cython>=0.29',
        'numpy>=1.19',
        'h5py>=3.0',
        'scipy>=1.5',
        'torch>=1.9.0',
    ],
    setup_requires=[
        'cython>=0.29',
        'numpy>=1.19',
        'scipy>=1.5',
    ],
    packages=['malis'],
    ext_modules=[
        Extension(
            "malis.malis",
            ["malis/malis.pyx", "malis/malis_cpp.cpp"],
            include_dirs=get_include_dirs(),
            library_dirs=get_library_dirs(),
            language='c++',
            extra_link_args=["-std=c++11"],
            extra_compile_args=["-std=c++11", "-w"],
        )
    ],
    zip_safe=False,
    python_requires='>=3.8',
)
