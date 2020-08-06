import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="paranumpy", # Replace with your own username
    version="1.0.1",
    author="Fabio Caruso",
    author_email="caruso@physik.uni-kiel.de",
    description="A set of functions for the parallel handling of numpy arrays",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
