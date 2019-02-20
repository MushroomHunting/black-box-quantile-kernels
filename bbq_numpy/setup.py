from setuptools import setup, find_packages

setup(
    name='bbq',
    version='1.0.0',
    description='Black Box Quantile Kernels',
    url='https://github.com/MushroomHunting/black-box-quantile-kernels',
    author='Anthony Tompkins, Ransalu Senanayake, Philippe Morere',
    license='MIT',
    packages=[package for package in find_packages() if
              (package.startswith("bbq"))],
    install_requires=[
        'numpy',
        'matplotlib',
        'ghalton',
        "scipy",
        "tqdm",
        "nlopt",
        'scikit-learn'
    ],
    zip_safe=False
)
