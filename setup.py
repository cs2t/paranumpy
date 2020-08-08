import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="paranumpy", 
    version="1.0.1",
    author="Fabio Caruso",
    author_email="caruso@physik.uni-kiel.de",
    description="A collection of functions for the parallel handling of numpy arrays",
    long_description='`paranumpy` is conceived as a tool to facilitate the parallelization of numpy arrays in python.
It consists of a set of functions to handle numpy arrays in a MPI (mpi4py) parallel environment. Further information is 
available at [https://paranumpy.readthedocs.io](https://paranumpy.readthedocs.io).',
    long_description_content_type="text/markdown",
    url="https://github.com/cs2t/paranumpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'mpi4py',
    ],
    python_requires='>=3.6',
)
