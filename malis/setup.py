"""Setup script for building malis Cython extension in-place.

This is a helper script for building the malis extension module directly
within the malis package directory. For normal installation, use the
setup.py in the parent directory.
"""


def setup_cython():
    """Build the malis Cython extension."""
    from setuptools import Extension, setup
    from Cython.Build import cythonize
    import numpy

    ext_modules = [
        Extension(
            "malis",
            ["malis.pyx", "malis_cpp.cpp"],
            language='c++',
            extra_link_args=["-std=c++11"],
            extra_compile_args=["-std=c++11", "-w"],
        )
    ]

    setup(
        ext_modules=cythonize(ext_modules),
        include_dirs=[numpy.get_include()],
    )


if __name__ == '__main__':
    setup_cython()
