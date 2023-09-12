'''
Module for define types
'''

# __all__ = []

from typing import *

from sympy.core import power, symbol, add, mul
import numpy as np
import dolfin as df
import matplotlib
import ufl
import mpl_toolkits.mplot3d as matplotlib3d
from ufl_legacy import (cell, form, algebra, tensoralgebra, differentiation,
                        integral, measure, indexsum)
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import parasite_axes
from dolfin.cpp.mesh import (MeshFunctionSizet, MeshFunctionBool,
                             MeshFunctionDouble)

# Set any types
Class = TypeVar('Class')
Function = TypeVar('Function')
File = TypeVar('File')
Number = Union[
    int, float, complex
    ]
Iterable = Union[
    str, tuple, list
    ]

Axe = TypeVar('Axe')
Colorbar = TypeVar('Colorbar')
Plot = Union[
    matplotlib.lines.Line2D,
    matplotlib.collections.PathCollection,
    matplotlib3d.art3d.Line3D,
    matplotlib3d.art3d.Poly3DCollection,
    matplotlib3d.art3d.Path3DCollection,
    ]
Collection = Union[
    matplotlib.collections.PathCollection,
    matplotlib3d.art3d.Poly3DCollection,
    matplotlib3d.art3d.Path3DCollection,
    ]
HostAxes = parasite_axes.HostAxes
Color = Union[
    matplotlib.colors.LinearSegmentedColormap,
    matplotlib.colors.ListedColormap
    ]

Array = Union[
    TypeVar('Array'),
    np.ndarray
    ]
Matrix = Union[
    np.matrix,
    df.Matrix,
    df.PETScMatrix
    ]
Vector = Union[
    df.GenericVector,
    df.Vector,
    df.PETScVector
    ]

Mesh = df.Mesh
MeshFunction = Union[
    MeshFunctionSizet,
    MeshFunctionBool,
    MeshFunctionDouble
    ]
Element = Union[
    df.FiniteElement,
    df.VectorElement,
    df.MixedElement
    ]
Solver = Union[
    df.PETScKrylovSolver,
    df.PETScLUSolver,
    df.KrylovSolver,
    df.LUSolver,
    df.NewtonSolver,
    df.PETScSNESSolver
    ]

MyExpression = TypeVar('MyExpression')
Argument = ufl.Argument
Equation = ufl.equation.Equation
Form = form.Form
Cell = cell.Cell
Integrand = Union[
    tensoralgebra.Dot,
    tensoralgebra.Inner,
    algebra.Product,
    algebra.Sum,
    algebra.Power,
    algebra.Division,
    algebra.Conj,
    differentiation.Div,
    indexsum.IndexSum
    ]
Integral = integral.Integral
Measure = measure.Measure

AnalyticalFunction = TypeVar('AnalyticalFunction')

SpExpression = Union[
    power.Pow,
    add.Add,
    mul.Mul,
    symbol.Symbol
]

class Map(type):

    def __class_getitem__(self, *args):
        return 'Map[%s]' % str(args).strip('(),')
