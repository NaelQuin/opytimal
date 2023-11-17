from setuptools import setup, find_packages

VERSION = '0.9.0'
DESCRIPTION = 'Optimal control PDE-based solver'
LONG_DESCRIPTION = 'Opytimal is a Python/FEniCS framework that have the main goal solve Optimal Control problems considering multiple and mixed controls based to linear and nonlinear PDEs, in addition to can also solve PDEs simply and clearly'

setup(
    name="opytimal",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Natanael Quintin",
    author_email="natanael.quintino@ipiaget.pt",
    license='CC0 1.0 Universal',
    packages=find_packages(),
    install_requires=[
        "dolfin==2019.2.0",
    ],
    keywords='optimalcontrol, FEniCS, FuildDynamics, NavierStokes, Stokes',
    classifiers= [
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ]
)