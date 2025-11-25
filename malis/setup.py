def setup_cython():
    from setuptools import setup, Extension
    from Cython.Build import cythonize
    import numpy

    ext_modules = [
        Extension(
            "malis",
            ["malis.pyx", "malis_cpp.cpp"],
            language='c++',
            extra_link_args=["-std=c++11"],
            extra_compile_args=["-std=c++11", "-w"]
        )
    ]

    setup(
        ext_modules=cythonize(ext_modules),
        include_dirs=[numpy.get_include()]
    )


if __name__ == '__main__':
    setup_cython()
