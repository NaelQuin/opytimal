from setuptools import setup, find_packages

VERSION = '0.9.9.1'
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
    packages=find_packages(
        include=['opytimal', 'opytimal.*', 'demos', 'demos.*']
        ),
    install_requires=[
        "fenics-dijitso==2019.2.0.dev0",
        "fenics-dolfin==2019.2.0.dev0",
        "fenics-ffc==2019.2.0.dev0",
        "fenics-fiat==2019.2.0.dev0",
        "fenics-ufl==2023.2.0",
        "fenics-ufl-legacy==2022.3.0",
        "screeninfo>=0.8.1",
        "termcolor>=2.2.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.2",
        "scipy>=1.10.1",
        "pandas>=1.5.3",
        "sympy>=1.11.1",
        "tikzplotlib>=0.8.2",
        "easygui>=0.98.1",
    ],
    keywords='optimalcontrol, FEniCS, FuildDynamics, NavierStokes, Stokes',
    classifiers= [
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ]
)