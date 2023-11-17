'''
Opytimal module
'''

from opytimal.numeric import getPow10
from opytimal.profiler import ProgressBar
from opytimal.tests import testLoop
from opytimal.parallel import parallel
from opytimal.types import Tuple, Union
from opytimal.files import createFolder
from opytimal.profiler import tic, toc, ticRAM, tocRAM
from opytimal.mathFunctions import sin, cos, tan, exp, log, ln
from opytimal.symbols import x, y, z, t, symbols
from opytimal.analytical import (AnalyticalFunction, AnalyticalVectorFunction)
from opytimal.arrays import (identity, zeros, getComplement, array, concatenate,
                             ravel, hstack, vstack, arange, norm as arrNorm,
                             flatten)
from opytimal.string import (showInfo, splitPathFile, replaceProgressive,
                             basename, showErrors)
from opytimal.meshes import (readXDMFMesh, getInnerNodes, getSubMeshes,
                             getCoordinates, getNormal, getBoundaryDofsWithout,
                             getBoundaryDofs)
from opytimal.plots import (plotMesh, plotComparison, adjustFiguresInScreen,
                            show, figure, dynamicComparison, imageTypes,
                            setPltStyle, plt)
from opytimal.fenics import (setExpression, extractElements, calculeNnz,
                             mySolve, appendErrorsByTime, getInputExactData,
                             setSolver, getErrorFormula, gradJ, copySolver,
                             getFormTerm, emptyForm, gradientDescent, showError,
                             getDomainLabels, replaceBoundNameByMark,
                             getMeasure, evaluateErrors, Zero, getAdjointSystem,
                             getOptimalConditionsSystem, evaluateCost,
                             getLocal, setLocal, showProblemData,
                             getFunctionExpressions)


def __getattr__(attr):
    # Warn for expired attributes
    import warnings

    if attr == "demos":
        import opytimal.demos as demos
        return demos

    raise AttributeError("module {!r} has no attribute "
                            "{!r}".format(__name__, attr))

def __dir__():
    public_symbols = (
        {'demos'}
    )
    return list(public_symbols)
