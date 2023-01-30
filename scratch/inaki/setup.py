import setuptools


class NoNumpy(Exception):
    pass


try:
    from numpy.distutils.core import Extension
    from numpy.distutils.core import setup
except ImportError:
    raise NoNumpy('Numpy needs to be installed for extensions to be compiled.')


if __name__ == "__main__":
    with open("README.md", "r") as fh:
        long_description = fh.read()

    setup(name="qt",
          version="0.0.1",
          author="",
          author_email="",
          description="Classical and quantum simulations of qubit transmission",
          long_description=long_description,
          long_description_content_type="text/markdown",
          url="https://github.com/inaki-ortizdelandaluce/qubit-transmission-simulations",
          packages=setuptools.find_packages(),
          license="MIT",
          classifiers=[
              "Programming Language :: Python :: 3",
              "License :: OSI Approved :: MIT License",
          ],
          install_requires=['numpy', 'matplotlib', 'healpy',],
          python_requires='>=3.10',
          )

