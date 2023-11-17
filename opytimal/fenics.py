'''
Module for fenics proccessment methods
'''

__all__ = ['setExpression', 'extractElements', 'mySolve', 'setSolver',
           'dot', 'inner', 'grad', 'div', 'isTensor', 'calculeNnz',
           'getErrorFormula', 'gradJ', 'replaceForm', 'getFormTerm',
           'gradientDescent', 'getDomainLabels', 'replaceBoundNameByMark',
           'evaluateErrors', 'Zero', 'showError', 'getAdjointSystem',
           'getOptimalConditionsSystem', 'getFunctionExpressions',
           'appendErrorsByTime']

import ufl
import ufl_legacy
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

from .meshes import getBoundaryData, getBoundaryArea
from .dicts import getMultipleKeys
from .string import replaceProgressive, convertInt
from .settings import luSolverSettings, krylovSolverSettings, descentSettings
from .functions import getVars
from .tests import testLoop
from .profiler import tic, toc
from .string import showInfo, splitParcels
from .arrays import contains, array, vectorize, norm as arrNorm
from .dicts import revertKeysValues
from .tables import table, setOption
from .plots import show, figure, title, adjustFiguresInScreen
from .types import (Function, Element, Class, Equation, Union, Array,
                    Argument, Any, Tuple, Form, Cell, Integrand, Integral,
                    Measure, Matrix, Vector, Solver, MeshFunction,
                    MyExpression, File)
from .symbols import x, y, z, t, toccode


def setExpression(
    func: Function,
    element: Element = df.FiniteElement('CG', None, 1),
    degree: int = None,
    name: str = None,
    t: bool = False
        ) -> (df.UserExpression):

    # Generate the correspondently MyExpression class
    MyExpression = generateMyExpression(func, element, degree, t)

    # Set the respective expression
    expression = MyExpression(name=name)

    if 'Analytical' in type(func).__name__:
        # Get the analytical function formula
        expression._def = func.definition

    else:
        # Add the t variable to func args
        kw = {'t': t}\
            if 't' in getVars(func)\
            else {}

        # Add the analytical function formula to expression class
        expression._def = toccode(func(x, y, z, **kw))

    return expression


def generateMyExpression(
    func: Function,
    element: Element = None,
    degree: int = None,
    t: bool = False,
    dim: int = None
        ) -> (Class):

    if element is None and degree is not None and dim is not None:
        # Set the respective lagrange vector element
        element = df.VectorElement('P', None, degree, dim)

    elif element is None and degree is not None:
        # Set the respective lagrange element
        element = df.FiniteElement('P', None, degree)

    elif element is None and degree is None:
        raise KeyError('Must supply element or degree')

    class MyExpression(df.UserExpression):

        def __init__(self, **kwargs):
            self.t = 0  # Initial value
            self.element = element
            super().__init__(**kwargs)
            return None

        def eval(self, values, x):
            values[:] = func(*x, t=self.t)
            return None

        def value_shape(self):
            return self.element.value_shape()

    if t:  # and not any(element.value_shape()):
        # Update the MyExpression eval method
        class MyExpressionUpdate(MyExpression):

            def eval(self, values, x):
                values[:] = func(*x, t=self.t)
                return None

    else:  # if not t and any(element.value_shape()):
        # Update the MyExpression eval method
        class MyExpressionUpdate(MyExpression):

            def eval(self, values, x):
                values[:] = func(*x)
                return None

    # elif any(element.value_shape()):
    #     # Update the MyExpression eval method
    #     class MyExpressionUpdate(MyExpression):

    #         def eval(self, values, x):
    #             values[:] = func(*x)
    #             return None

    if 'MyExpressionUpdate' in locals():
        # Rename the updated class
        MyExpression = MyExpressionUpdate

    return MyExpression


def extractElements(
    Th: str,
    cell: Union[Cell, str]
) -> list[Element]:
    # Remove the whitespaces and split by comma
    Th = Th.replace(' ', '').split(',')

    # Init a elements list
    elements = []

    # Looping in the elements given
    for element in Th:
        # Verify if It is vector
        vector = '[' in element and ']' in element

        # Apply a split on vector elements
        element = element.strip(']').split('[')

        # Get the vector dimension (If there is)
        dim = int(element[1]) \
            if len(element) > 1 and element[1] not in [':', '']\
            else getTopologicalDimension(cell)

        # Split the element space and basis degree
        P_deg = list(element[0])
        P = ''.join(P_deg[:-1])
        deg = P_deg[-1]

        # Set the respective function element
        # and your extra args
        if vector:
            FuncElement = df.VectorElement
            args = (dim,)
        else:
            FuncElement = df.FiniteElement
            args = ()

        # Add to elements list
        elements.append(
            FuncElement(P, cell, int(deg), *args)
        )

    return elements


def mySolve(
    equation: Equation,
    w: df.Function,
    bcs: list[df.DirichletBC],
    solver: Solver,
    A: Matrix = df.PETScMatrix(),
    b: Vector = df.PETScVector(),
    P: Matrix = None,
    identZeros: bool = False,
    runtime: bool = False,
    nnz: int = None,
    dofs: int = None,
    copyTo: File = None
        ) -> (Tuple[df.Matrix, df.Vector]):

    if runtime:
        # Init a timer
        tic()

    # Split the variational formulation side
    a = equation.lhs
    L = equation.rhs

    # Assemble matrix and vector
    df.assemble(a, tensor=A, keep_diagonal=identZeros)
    df.assemble(L, tensor=b)

    if bcs is not None:
        # Applying the boundary conditions
        [bc.apply(A, b) for bc in bcs]

    if identZeros:
        # Turn zeros to one in diagonal
        A.ident_zeros()

    # Set the solver operators
    solver.set_operator(A)\
        if P is None or type(solver) is df.LUSolver\
        else solver.set_operators(A, P)

    if runtime:
        # Get the solver label and split it
        solverLbl = solver.label().split('_')

        # Split the linear_solver and preconditioner
        linearSolver = solverLbl[1]
        preconditioner = '_'.join(solverLbl[2:])\
            if len(solverLbl) > 2\
            else None

        # Print the info
        showInfo(
            'System data', copyTo=copyTo
        )
        showInfo(
            f'Nonzeros: {convertInt(nnz)}' if nnz is not None else '',
            f'DOFS: {convertInt(dofs)}' if dofs is not None else '',
            f'Linear Solver: {linearSolver}',
            f'Preconditioner: {preconditioner}',
            alignment=':', delimiters=False, breakStart=False,
            ignoreEmptyLines=True, copyTo=copyTo
        )

    # Solve the matricial system
    solver.solve(w.vector(), b)

    if runtime:
        # Stop the timer
        runtime = toc(printer=False, convert=True)

        # Show the runtime
        showInfo(
            f'Runtime: {runtime}',
            alignment=':', delimiters=False, breakStart=False,
            ignoreEmptyLines=True, copyTo=copyTo
        )

    return A, b


def setSolver(
    linearSolver: str,
    preconditioner: str = None,
    settings: dict[str: Union[str, float, bool]] = None
) -> Union[df.KrylovSolver, df.LUSolver]:
    '''Set the best fenics solver'''

    # Set the correct solver function
    if df.has_krylov_solver_method(linearSolver):
        # Set the Krylov solver
        solver = df.PETScKrylovSolver(linearSolver, preconditioner)

        # Set the solver name
        solverName = f'Krylov_{linearSolver}_{preconditioner}'

        # Update solver parameters
        solver.parameters.update(
            settings
            if settings is not None
            else krylovSolverSettings
        )

    else:
        # Set the LU solver
        solver = df.PETScLUSolver(linearSolver)

        # Set the solver name
        solverName = f'LU_{linearSolver}'

        # Update solver parameters
        solver.parameters.update(
            settings
            if settings is not None
            else luSolverSettings
        )

    # Apply the solver name
    solver.rename(solverName, solverName)

    return solver


def grad(
    u: Union[Argument, df.Function, Array]
) -> (Array):
    if any(u.ufl_shape):
        output = [
            array([ui.dx(i) for i in range(ui.geometric_dimension())])
            for ui in u
        ]
    else:
        output = array([u.dx(i) for i in range(u.geometric_dimension())])
    return output


def div(
    u: Union[Argument, df.Function, Array]
) -> Union[Argument, df.Function]:
    return grad(u).sum()


def dot(
    u: Union[Argument, df.Function, Array],
    v: Union[Argument, df.Function, Array]
) -> (Union[Argument, df.Function]):
    if type(u) is np.ndarray or any(u.ufl_shape):
        output = sum([ui * vi for ui, vi in zip(u, v)])
    else:
        output = u*v
    return output


def inner(
    u: Union[Argument, df.Function, Array],
    v: Union[Argument, df.Function, Array]
) -> Union[Argument, df.Function]:
    if isTensor(u) and isTensor(v):
        output = sum(
            [[uij * vij for uij, vij in zip(ui, vi)]
                for ui, vi in zip(u, v)]
        )
    else:
        output = dot(u, v)

    return output


def isTensor(
    u: Any
) -> bool:
    '''Verify if u is a tensor'''

    # Verify if is array of the vector functions or vector arguments
    output = isArray(u) & any(u[0].ufl_shape)

    if not isArray(u):
        # Verify if is a fenics tensor
        output |= any(u.ufl_shape) & any(u[0].ufl_shape)

    return output


def isArray(
    u: Any
) -> bool:
    return type(u) is np.ndarray


def calculeNnz(
    V: df.FunctionSpace,
    dx: Measure = None
) -> int:
    # Get the trial and test functions
    u = df.TrialFunction(V)
    v = df.TestFunction(V)

    # Assemble the mass matrix
    mass = df.assemble(sum([df.dot(ui, vi) for ui, vi in zip(u, v)])*dx)\
        if len(u.ufl_shape) > 0\
        else df.assemble(df.dot(u, v)*dx)

    # Get the nonzero values amount
    nnz = mass.nnz()

    return nnz


def getErrorFormula(
    ud: df.Function,
    dx: Measure = df.dx,
    relative: bool = True
        ) -> dict[str:Form]:
    '''
    class Error
    error = Error(ud, dx)
    error.evaluate(U, relative)
    error.relativize(v=self.ud)
    error.show()
    '''

    # Get the respective function space
    V = ud.function_space()

    # Set a trial function in the same function space
    u = df.TrialFunction(V)

    # Get the error formula
    error = getNormFormula(u - ud, dx)

    if relative:
        # Get the norm to relative the error
        norm = getNormFormula(ud, dx, evaluate=True)

        # Replace zeros to one
        norm = replaceNullToOne(norm)

        # Relativize the errors and add to error dict
        error = {k: (v, 1/norm[k]*v) for k, v in error.items()}

    return error


def replaceNullToOne(
    norm: dict[str: float]
        ) -> (dict[str: float]):

    # Get norm dict items
    normItems = array(list(norm.items()), dtype=object)

    # Find the null norms
    nullNorms = normItems[:, 1] == 0

    # Adjustment to null norms
    if nullNorms.any():
        # Replace null values to one
        normItems[:, 1][nullNorms] = 1

        # Turn back to dict
        norm = dict(normItems)

    return norm


def getNormFormula(
    u: df.Function,
    dx: Measure = df.dx,
    evaluate: bool = False
        ) -> (dict[str:Form]):

    # Set the L² formula
    norm = {'L²': u**2*dx}

    # Set the H¹ formula
    norm['H¹'] = norm['L²'] + df.inner(df.grad(u), df.grad(u))*dx

    if getFiniteElementDegree(u) > 1:
        # Set the H² formula
        norm['H²'] = norm['H¹'] + df.div(df.grad(u))**2*dx

    if evaluate:
        # Evaluate the norm formulas
        norm = {k: df.assemble(v)**0.5 for k, v in norm.items()}

    return norm


def evaluateErrors(
    *formFunc: Tuple[Form, df.Function],
    labels: list[str] = None,
    relative: bool = False
) -> dict[str:dict[str:float]]:

    if labels is None:
        # Get the function names
        labels = [S.name() for S, _ in formFunc]

    # Init a dict to output
    errors = {}

    if relative:
        # Looping in pair (Func, errorForm)
        for k, (S, error) in enumerate(formFunc):
            # Add to respective func label the errors
            errors[labels[k]] = {
                eType: (df.assemble(df.action(e[0], S))**0.5,
                        df.assemble(df.action(e[1], S))**0.5)
                for eType, e in error.items()
            }
    else:
        # Looping in pair (Func, errorForm)
        for k, (S, error) in enumerate(formFunc):
            # Add to respective func label the errors
            errors[labels[k]] = {
                eType: df.assemble(df.action(e, S))**0.5
                for eType, e in error.items()
            }

    return errors


def getTopologicalDimension(cell: Union[Cell, str]) -> int:
    tdim = {'interval': 1, 'triangle': 2, 'tetrahedron': 3}[cell]\
        if type(cell) is str\
        else cell.topological_dimension()
    return tdim


def gradJ(J, var, z=None, f=None, ug=None, h=None, v=None):
    # Set the epsilon scalar
    epsilon = df.Constant(1e-1, name='ε')

    # Get cost args
    costArgs = getVars(J)

    # Replace the arg to respective differentiation
    costArgsDiff = str(costArgs).replace(var, f'{var} + epsilon*v')

    # Evaluate cost args content
    costArgs = map(eval, costArgs)

    # Evaluate cost differentiation args
    costArgsDiff = eval(costArgsDiff)

    # Evaluate args content
    costArgsDiff = map(eval, costArgsDiff)

    # Calcule the differentiation
    diff = 1/epsilon*(J(*costArgsDiff) - J(*costArgs))

    return diff


def replaceForm(
    form: Form,
    *oldNew: Tuple[Union[df.Constant, df.Function],
                   Union[Argument, df.Constant, df.Function]]
) -> Form:

    # Looping in (old, new) pair
    for old, new in dict(oldNew).items():

        if old.ufl_shape == tuple():
            # Make the respective replacement
            form = ufl.replace(form, {old: new})
        else:
            # Loopin in coordinates
            for old_i, new_i in zip(old, new):
                # Make the respective replacement
                form = ufl.replace(form, {old_i: new_i})
                input('Done')

    return form


def getFormTermString(
    form: str,
    *terms: Union[Argument, df.Constant, df.Function],
    timeDependent: bool = False
        ) -> (dict[str: Form]):

    # Set the output dict forms
    formByTerm = {term: [] for term in terms}

    # Split the form parcels
    parcels = array(
        splitParcels(form)
    )

    # Looping in terms
    for term in terms:
        # Verify if term in parcels
        verification = contains(parcels, term)

        if timeDependent:
            verification *= ~contains(parcels, f'{term}n')

        if any(verification):
            # Get the respective parcels
            formByTerm[term].extend(
                parcels[verification]
            )

    # Looping in term and parcels
    for term, parcels in formByTerm.items():
        # Join the parcels of the respective form
        formByTerm[term] = ' + '.join(parcels)

    return formByTerm


def getFormTerm(
    form: Union[Form, str],
    *terms: Union[Argument, df.Constant, df.Function],
    timeDependent: bool = False
        ) -> (dict[str: Form]):

    if type(form) is str:
        # Call specific function
        return getFormTermString(form, *terms, timeDependent=timeDependent)

    # Set the output dict forms
    formByTerm = {term.name(): emptyForm() for term in terms}

    # Get form integrals
    integrals = form.integrals()

    # Looping in integrals
    for integral in integrals:
        # Get respective integrand in string
        integrand = integral.integrand()

        # Turn integrand to string
        integrandStr = str(integrand)

        # Get respective measure
        dm = getMeasure(integral)

        # Looping in terms
        for term in terms:

            # Verify if operands contains the term
            if term.name() in integrandStr:
                # Add to respective control term
                formByTerm[term.name()] += integrand*dm

                # Go to next integrand
                break

    return formByTerm


def measureByType(type: str) -> df.Measure:
    # Get the respective measure
    dm = {
        'cell': df.dx,
        'exterior_facet': df.ds,
        'interior_facet': df.dS
    }[type]

    return dm


def getMeasure(
    integral: Integral
) -> Measure:

    # Get the respective measure
    dm = measureByType(integral.integral_type())

    # Get the subdomain id
    i = integral.subdomain_id()

    # Get the subdomain data
    dmSubDomainData = integral.subdomain_data()

    # Get the integral meta data
    dmMetadata = integral.metadata()

    # Set the resepctive measure
    dm = dm(i, metadata=dmMetadata, subdomain_data=dmSubDomainData)

    return dm


def getOperands(
    integrand: Integrand
) -> Tuple[list[df.Constant], list[Argument]]:
    # Init a output operand list
    scalars = []
    others = []

    # Get the operands
    operands = integrand.ufl_operands

    if len(operands) > 1:

        if 'constant' in operands[0].__module__.lower():
            scalars.append(float(operands[0]))
        elif 'algebra' in operands[0].__module__.lower():
            nextOperands = getOperands(operands[0])
            scalars.extend(nextOperands[0])
            others.extend(nextOperands[1])
        else:
            others.append(operands[0])

        if 'indexsum' not in operands[1].__module__.lower():
            nextOperands = getOperands(operands[1])
            scalars.extend(nextOperands[0])
            others.extend(nextOperands[1])
        else:
            others.append(operands[1])

    elif len(operands) == 1:
        if 'constant' in operands[0].__module__.lower():
            scalars.append(float(operands[0]))
        else:
            others.append(operands[0])

    elif 'constant' in integrand.__module__.lower():
        scalars.append(integrand)

    else:
        others.append(integrand)

    return scalars, others


def reduceOperands(
    *forms: ufl.Form,
    conjugate: bool = False,
    separateSignal: bool = False,
    fixBoundaryMean: bool = False
) -> (ufl.Form):
    # Set the measures by type
    measuresByType = {
        'cell': df.dx,
        'exterior_facet': df.ds,
        'interior_facet': df.dS
    }

    # Init the output list
    reducedForms = []

    # Looping in forms
    for form in forms:
        # Get the mesh geometric dimension
        dim = form.geometric_dimension()

        # Set the product function convenently
        prod = inner\
            if dim == 3\
            else dot

        # Init the reduced variational formulation
        reducedForm = emptyForm()

        # Get the integrals
        integrals = form.integrals()

        # Looping in integrals
        for integral in integrals:
            # Get the respective operands
            scalars, others = getOperands(integral.integrand())

            if 0 in scalars:
                continue

            # Make the scalars product
            scalars = np.prod(scalars)

            if separateSignal:
                # Get the scalars signal
                signal = scalars/abs(scalars)

                # Remove the scalar signal
                scalars = abs(scalars)

            # Remount the integrand
            integrand = prod(scalars*others[0], others[1])\
                if len(others) == 2\
                else scalars*others[0]

            # Recover the integral measure
            dm = getMeasure(integral)
            # dm = measuresByType[integral.integral_type()]
            # dmId = integral.subdomain_id()
            # dmMetadata = integral.metadata()
            # dmSubDomainData = integral.subdomain_data()
            # dm = dm(dmId, metadata=dmMetadata,
            #         subdomain_data=dmSubDomainData)

            if conjugate:
                # Get the integrand's conjugated
                integrand = ufl.conj(integrand)

            input(integrand)
            input(dm)
            input(integrand*dm)

            if fixBoundaryMean\
                    and 'facet' in dm.integral_type():
                # Integrate by complement
                reducedIntegral = integralByComplement(
                    integrand, dm, integral.subdomain_id(),
                    set(integral.subdomain_data().array())
                )

            else:
                # Finalize reduced integral
                reducedIntegral = integrand*dm

            # print(
            #     reducedIntegral,
            #     integrand,
            #     dm,
            #     sep='\n'
            #     )
            # input()

            if separateSignal:
                # Put the signal
                reducedIntegral = signal*(reducedIntegral)

            # Add to reduced variational formulation
            reducedForm += reducedIntegral

        # Add to output list
        reducedForms.append(reducedForm)

    if len(reducedForms) == 1:
        # Get the unique form
        reducedForms = reducedForms[0]

    return reducedForms


def emptyForm() -> (Form):
    return ufl_legacy.form.Form([])


def gradientDescent(
    controls: list[str],
    costFunc: Function,
    a_s: dict[str: Tuple[df.Constant, df.Constant]],
    dm: dict[str: df.Measure],
    stateSystem: Tuple[ufl.form.Form, ufl.form.Form, df.Function],
    adjointSystem: Tuple[ufl.form.Form, ufl.form.Form, df.Function],
    optimalSystem: Tuple[ufl.form.Form, ufl.form.Form, df.Function],
    bcs: dict[str: list[df.DirichletBC]],
    solver: Solver,
    exactData: dict[str: df.Function],
    errorForm: dict[str:Form],
    A: Union[Matrix, Tuple[Matrix, Matrix]] = 2*(df.PETScMatrix(),),
    b: Union[Vector, Tuple[Vector, Vector]] = 2*(df.PETScVector(),),
    P: dict[str: Matrix] = {eq:None for eq in ['state', 'adjoint', 'optimal']},
    initialControl: dict[str: Function] = None,
    rho: float = 1,
    gamma: str = '1',
    descentSettings: dict[str: Union[str, float, bool]] = descentSettings,
    copyTo: File = None
        ) -> (tuple[float, float], tuple[int, int], float):
    '''
    Solve the optimal system by Descent Gradient Method
    '''

    # Get the domain labels
    omg = getDomainLabels(
        {k: getMeasure(v['L²'][0].integrals()[0])
            for k, v in errorForm.items()}
    )

    # Turn to local, the descent settings
    iteractivePlots = descentSettings.get('iteractivePlots', False)
    maxIter = descentSettings.get('maxIter', 2000)
    costTol = descentSettings.get('costTol', 1e-6)
    costTolRel = descentSettings.get('costTolRel', 1e-9)
    rhoTol = descentSettings.get('rhoTol', 1e-6)
    stateTolPercent = descentSettings.get('stateTolPercent', 1)
    controlTolPercent = descentSettings.get('controlTolPercent', 3)
    solutionsTolRelation = descentSettings.get('solutionsTolRelation', 'and')

    if hasattr(solver, '__len__'):
        # Split the solver and your copy
        solver, solverOpt = solver

    else:
        # Make a copy to the respective solver
        solverOpt = copySolver(solver)

    # Init empties Matrix and Vector
    A = df.PETScMatrix()
    b = df.PETScVector()
    AOpt = [df.PETScMatrix() for _ in controls]
    bOpt = [df.PETScVector() for _ in controls]

    if initialControl is None:
        # Set an initial null control function
        C0 = Function(VOpt)

        # Split the function
        C0Split = C0.split()\
            if len(controls) > 1\
            else [C0]

        # Split the initial controls by controls
        C0 = {c: C0Split()[i] for i, c in enumerate(controls)}\
            if len(controls) > 1\
            else {controls[0]: C0}

    else:
        # Put a alias to initial control
        C0 = initialControl

    # Split the variational formulation by side
    aState, LState, staSol = stateSystem
    aAdjoint, LAdjoint, adjSol = adjointSystem
    aOptimal, LOptimal, optSol = optimalSystem

    # Get the solution function spaces
    VSta = staSol.function_space()
    VOpt = [opt.function_space() for opt in optSol]

    # Split the controls vectors
    Cs = {c: C for c, C in zip(controls, optSol)}\
        if len(controls) > 1\
        else {controls[0]: optSol[0]}

    # Rename controls vector solutions
    [Cs[c].rename(c, c) for c in controls]

    # Set the optimal solution as initial control
    [Cs[c].assign(C0[c]) for c in controls]

    # Reduce the scalars and adjust the boundary means of the
    # aOptimal and LOptimal formulations
    # aOptimal, LOptimal = reduceOperands(
    #     aOptimal, LOptimal,
    #     fixBoundaryMean='g' in (''.join(controls)) or 'h' in (''.join(controls))
    #     )

    # Set default initial default values
    relCostError = None               # Relative cost
    iter = iterSub = iterSubTotal = 0 # Iteration counters
    Jk_1 = 0                          # Previous cost value

    # Set the initial gamma weight like function
    gammak = eval(
        f"lambda dJk, dJk_1: {gamma}",
        {'norm': arrNorm},
        locals()
        )

    # Init a list to store learn rates
    rhos = [rho]

    # Verify if state function space is mixed
    mixedSpace = VSta.num_sub_spaces() > 0

    # Set the current state solution
    Uk = df.Function(VSta, name='Uk')

    # Set the current controls, gradient and direction
    Ck, dJk, dk = [
        [df.Function(VOpti, name=f'{name}k')
            for VOpti in VOpt]
        for name in ['C', 'dJ', 'd']
    ]

    # Set the previous gradient and direction
    dJk_1, dk_1 = [
        [df.Function(VOpti, name=f'{name}k_1')
            for VOpti in VOpt]
        for name in ['dJ', 'd']
    ]

    # Collapse the optimal function spaces
    VOpt = {c: V for c, V in zip(controls, VOpt)}

    # Set the storage to solutions by iterations
    solutionsStore = {
        'U': {},
        'C': {c: {} for c in controls}
    }

    # Set a temporary function
    U = df.Function(VSta, name='U')\
        if not mixedSpace\
        else df.Function(VSta, name='U').split(deepcopy=True)

    if mixedSpace:
        # Split the subspaces
        VSta = VSta.split()

        # Collapse the subspaces
        VStaC = [V.collapse() for V in VSta]

        # Add the others state solutions to store
        solutionsStore.update(
            {f'U-{i}': {} for i in range(len(VSta))}
        )

    # Apply the solutions tolerance relation for state and
    # control variables
    stateControlRelation = '{}'\
        + len(controls)*(' %s {}' % solutionsTolRelation)

    # Gradient Descent Algorithm
    while iter < maxIter:

        if iteractivePlots and iter > 0:
            # Some plots
            [(figure(f'optimalState_{controls[i]}', clear=True),
              title(f'Optimal control "{controls[i]}" used in State'),
              df.plot(c)) for i, c in enumerate(optSol)]
            figure('StateBefore', clear=True)
            df.plot(U.copy(deepcopy=True))\
                if not mixedSpace else None
            title('State Before')

        # Solve the current state equation
        mySolve(
            aState == LState, staSol, bcs['state'],
            solver, A, b, P['state']
        )

        # Put a alias or get the subsolutions
        Z = staSol\
            if not mixedSpace\
            else staSol.split(deepcopy=True)

        # Calcule the cost and show it
        Jk = evaluateCost(
            costFunc, Z, exactData['ud'], Cs, a_s, dm,
            iters=(iter, iterSub), rho=rho,
            previousCost=Jk_1 if iter > 0 else None,
            copyTo=copyTo
        )

        if not mixedSpace:
            # Fixing the lifting
            U.assign(Z)

            if 'ug' in controls:
                # Add the lifting contribution
                setLocal(U, getLocal(U) + getLocal(Cs['ug']))

        else:
            # Update the solutions
            for Ui, Zi in zip(U, Z):
                # Set the Ui value
                Ui.assign(Zi)

            if 'ug' in controls:
                # Add the lifting contribution
                setLocal(U[0], getLocal(U[0]) + getLocal(Cs['ug']))

        if iteractivePlots:
            # Any plot
            figure('StateAfter', clear=True)
            df.plot(U.copy(deepcopy=True)
                    if not mixedSpace
                    else U[0].copy(deepcopy=True))
            title('State After')

        if not mixedSpace:
            # Calcule approach errors
            errors = evaluateErrors(
                (U, errorForm['u']),
                *[(Cs[c], errorForm[c]) for c in controls],
                labels=['u', *controls],
                relative=type(errorForm['u']['L²']) is tuple
            )

        else:
            # Calcule approach errors
            errors = evaluateErrors(
                (U[0], errorForm['u']),
                (U[1], errorForm['p']),
                *[(Cs[c], errorForm[c]) for c in controls],
                labels=['u', 'p', *controls],
                relative=type(errorForm['u']['L²']) is tuple
            )

        # Show the errors
        showError(errors, omg, copyTo=copyTo)

        # Stop criterion 4 [State or control solutions are expected]
        if stateControlRelation is not None \
                and eval(
                    stateControlRelation.format(
                        errors['u']['L²'][1]*100 < stateTolPercent,
                        *[errors[c]['L²'][1]*100 < controlTolPercent
                            for c in controls]
                    )):
            # Store the current state and control solutions
            Uk.assign(staSol)
            [Ck_i.assign(opt_i) for Ck_i, opt_i in zip(Ck, optSol)]

            # Store any gradient descent elements
            storeGradientDescentElements(
                solutionsStore, U=staSol, C=optSol,
                Jk=Jk, Jk_1=Jk_1, rho=rho, iter=iter,
                iterSubTotal=iterSubTotal, controls=controls,
                storeKey=iter
            )

            showInfo(
                stopCriterion := 'State/control solutions are expected',
                color='green', end='\n\n', copyTo=copyTo
            )

            # Stop the loop
            break

        # Stop criterion 1 [Cost tolerance]
        if Jk < costTol:
            # Store the current state and control solutions
            Uk.assign(staSol)
            Ck.assign(optSol)

            # Store any gradient descent elements
            storeGradientDescentElements(
                solutionsStore, U=staSol, C=optSol,
                Jk=Jk, Jk_1=Jk_1, rho=rho, iter=iter,
                iterSubTotal=iterSubTotal, controls=control,
                storeKey=iter
            )

            # Report this situation
            showInfo(
                stopCriterion := 'Cost tolerance is reached',
                color='green', end='\n\n', copyTo=copyTo
            )

            # Stop the loop
            break

        # Stop criterion 2 [Learn rate is smaller]
        elif rho < rhoTol:
            # Report this situation
            showInfo(
                stopCriterion := 'Learn rate is smaller',
                err=True, end='\n\n', copyTo=copyTo
            )

            # Stop the loop
            break

        # Stop criterion 3 [Gradient change is minimal]
        elif (relCostError := abs(Jk - Jk_1)/Jk) < costTolRel:
            # Report this situation
            showInfo(
                stopCriterion := 'Gradient change is minimal',
                err=True, end='\n\n', copyTo=copyTo
            )

            # Stop the loop
            break

        # Criterion to do gradient descent iterations
        elif Jk < Jk_1 or iter == 0:
            # Store any gradient descent elements
            storeGradientDescentElements(
                solutionsStore, U=staSol, C=optSol,
                Jk=Jk, Jk_1=Jk_1, rho=rho, iter=iter,
                iterSubTotal=iterSubTotal, controls=controls,
                storeKey=iter
            )

            # Store the current gradient as previous
            [dJk_1_i.assign(dJk_i)
                for dJk_1_i, dJk_i in zip(dJk_1, dJk)]

            # Store the current state and control solutions
            Uk.assign(staSol)
            [Ck_i.assign(opt_i)
                for Ck_i, opt_i in zip(Ck, optSol)
                if any(getLocal(opt_i))]

            # Store the current cost value as previous
            Jk_1 = Jk

            # Store the current rho
            rhos.append(rho)

            # Initialize the learn rate
            rho = {
                1: np.mean(rhos),
                2: rhos[0]
            }[1]

            if iter > 0:
                # Cummulate the subloop iterations
                iterSubTotal += iterSub

                # Store the current direction as previous
                [dk_1_i.assign(dk_i) for dk_1_i, dk_i in zip(dk_1, dk)]

            if iteractivePlots:
                # Any plot
                figure('AdjointBefore', clear=True)
                df.plot(adjSol)
                title('Adjoint Before')

            # Solve the current adjoint equation
            mySolve(
                aAdjoint == LAdjoint, adjSol, bcs['adjoint'],
                solver, A, b, P['adjoint']
            )

            if iteractivePlots:
                # Some plots
                figure('AdjointAfter', clear=True)
                df.plot(adjSol)
                title('Adjoint After')

                [(figure(f'dJkBefore_{controls[i]}', clear=True),
                  title(f'Optimal Gradient "{controls[i]}" Before'),
                  df.plot(c)) for i, c in enumerate(dJk)]

            # Looping in optimality conditions systems
            for i in range(len(controls)):
                # Solve the current optimal equation
                mySolve(
                    aOptimal[i] == LOptimal[i], dJk[i], None,
                    solverOpt, AOpt[i], bOpt[i], identZeros=True
                )

            if iteractivePlots:
                # Any plot
                [(figure(f'dJkAfter_{controls[i]}', clear=True),
                  title(f'Optimal Gradient "{controls[i]}" After'),
                  df.plot(c)) for i, c in enumerate(dJk)]
                adjustFiguresInScreen(
                    *[figure(name) for name in [
                        *[f'optimalState_{c}' for c in controls],
                        'StateBefore', 'StateAfter',
                        'AdjointBefore', 'AdjointAfter',
                        *[f'dJkBefore_{c}' for c in controls],
                        *[f'dJkAfter_{c}' for c in controls]]]
                )
                show()

            # Update the gradient iteration counter
            iter += 1

            # Restart the subloop counter
            iterSub = 0

            # Take the current gradient vector
            dJkVec = array([dJk_i.vector()[:] for dJk_i in dJk])

            if iter > 1:
                # Take the previous gradient vector
                dJk_1Vec = array([dJk_1_i.vector()[:] for dJk_1_i in dJk_1])

                # Take the previous direction vector
                dk_1Vec = array([dk_1_i.vector()[:] for dk_1_i in dk_1])

                # Add the previous gradient contribution
                dk_1Contrib = gammak(dJkVec, dJk_1Vec)
                dk_1Contrib *= dk_1Vec

            else:
                # Init the direction variable
                dk_1Contrib = 0

        else:  # [Cost didn't decrease]
            # Update the learn rate
            rho *= 0.5

            # Update the counter of the subloops
            iterSub += 1

        # Calcule the new direction candidate
        dkCandidate = dk_1Contrib - rho*dJkVec

        # Update the control perturbations
        [setLocal(dk_i, dkCand_i)
            for dk_i, dkCand_i in zip(dk, dkCandidate)]

        # Update the control
        [opt_i.assign(Ck_i + dk_i)
            for opt_i, Ck_i, dk_i in zip(optSol, Ck, dk)]

        # Update the each optimal solutions
        [Cs[c].assign(optSol[i])
            for i, c in enumerate(controls)]

    # ------------------------------
    # DESCENT IS BREAKED OR FINISHED
    # ------------------------------
    # Get the optimal solution
    staSol.assign(Uk)
    [opt_i.assign(Ck_i) for Ck_i, opt_i in zip(Ck, optSol)]

    # # Add the relative error between current and previous cost values
    # errors['costErrorRel'] = relCostError

    # Compact the iterators amount
    iters = (iter, iterSubTotal)

    return (solutionsStore, Jk_1, errors, iters, rho, stopCriterion)


def getMeasureNameByType(type: str) -> str:
    dmName = {
        'cell': 'dx',
        'exterior_facet': 'ds',
        'interior_facet': 'dS'
    }[type]
    return dmName


def getMeasureName(form: Union[Form, Integral]) -> str:

    if type(form) is Form:
        # Get the first integral
        form = form.integrals()[0]

    # Get the measure tyep
    dmType = form.integral_type()

    # Get the measure name
    dmName = getMeasureNameByType(dmType)

    return dmName


def domainLabel(
    dm: Union[str, df.Measure],
    markByNum: dict[int: str] = None
) -> str:

    if type(dm) is not str:
        # Get measure type and tag
        dmType = getMeasureNameByType(dm.integral_type())
        dmTag = dm.subdomain_id()

        if dmTag == 'everywhere':
            # Set a empty tag
            dmTag = ''

        elif markByNum is not None:
            # Replace number to respective label
            dmTag = markByNum[dmTag]

    else:
        # Get measure type and tag
        dmType = dm[:2]
        dmTag = dm[2:].strip('()')

    # Get the domain label
    domainLbl = {
        'dx': 'Ω',
        'ds': 'Γ'
    }[dmType]

    # Add the domain tag
    domainLbl += str(dmTag).replace('let', '')

    return domainLbl


def getDomainLabels(
    dm: dict[str: Union[str, df.Measure]],
    numByMark: dict[str: int] = None
        ) -> (dict[str: str]):

    # Revert the dictionary
    markByNum = revertKeysValues(numByMark)\
        if numByMark is not None\
        else None

    # Replace each measure to respective domain label
    domainLbls = {k: domainLabel(v, markByNum) for k, v in dm.items()}

    return domainLbls


def copySolver(solver: Solver) -> (Solver):
    # Get the solver name
    solverName = solver.name()

    # Count the char "_" in solver name
    sepAmount = solverName.count('_')

    # Check the solver name format
    if sepAmount == 0:
        raise ValueError("Isn't able to copy this solver")

    # Split the solver name
    solverNameSplitted = solverName.split('_')

    # Get the solver type and linear solver
    solverType, linearSolver = solverNameSplitted[:2]

    # Get the preconditioner
    preconditioner = '_'.join(solverNameSplitted[2:])\
        if sepAmount >= 2\
        else None

    # Build the solver copy
    solverCopy = df.PETScKrylovSolver(linearSolver, preconditioner)\
        if solverType == 'Krylov'\
        else df.PETScLUSolver(linearSolver)

    # Set the solver copy name
    solverCopy.rename(solver.name()+'_copy', solver.name()+'_copy')

    return solverCopy


def integralByComplement(
    integrand: Integrand,
    dm: Measure,
    idAim: int,
    allIds: list[int],
    removeZero: bool = True
) -> (ufl.Integral):

    # Remove respective idAim
    otherIds = set(allIds) - {idAim}

    if removeZero:
        # Remove the id zero
        otherIds -= {0}

    # Mount the measure sum without respective id
    dmOthers = '+'.join([f"dm({i})" for i in otherIds])

    # Finalize the integral
    integral = integrand*dm(None) - (integrand*eval(dmOthers))

    return integral


def storeGradientDescentElements(
    store: dict[str: Any],
    **kwargs
) -> (None):

    try:
        U = kwargs['U'].copy(deepcopy=True)
        C = kwargs['C'].copy(deepcopy=True)
        if 'P' in kwargs and kwargs['P'] is not None:
            P = kwargs['P'].copy(deepcopy=True)
    except TypeError:
        U = deepcopy(kwargs['U'])
        C = deepcopy(kwargs['C'])
        if 'P' in kwargs and kwargs['P'] is not None:
            P = deepcopy(kwargs['P'])

    # Store the respective solutions
    store['U'][kwargs['storeKey']] = U
    store['C'][kwargs['storeKey']] = C
    if 'P' in kwargs and kwargs['P'] is not None:
        store['P'][kwargs['storeKey']] = P

    # Store respectives values
    store['Jk'] = kwargs['Jk']
    store['Jk_1'] = kwargs['Jk_1']
    store['rho'] = kwargs['rho']
    store['iters'] = (kwargs['iter'], kwargs['iterSubTotal'])

    return None


def deepcopy(
    u: [df.Function, list[df.Function]],
    unique: bool = True
) -> ([df.Function, list[df.Function]]):
    'Return a deepcopy of the fenics function'

    # Turn input data to list (If need)
    us = u if type(u) in iterableTypes() else [u]

    # Init the output list
    output = []

    # Looping in elements
    for u in us:
        # Add the respective copy
        output.append(
            u.copy(deepcopy=True)
            if type(u) is not dict
            else deepcopyDict(u)
        )

    if unique and len(output) == 1:
        # Get the unique value
        output = output[0]

    return output


def iterableTypes() -> (list[type]):
    return [list, tuple, np.ndarray]


def deepcopyDict(dic: dict[Any: Any]) -> dict[Any: Any]:
    # Init the output dict
    output = {}

    # Looping in dictionary items
    for k, v in dic.items():
        # Make the deepcopy convenently
        output[k] = deepcopyDict(v)\
            if type(v) is dict\
            else v.copy(deepcopy=True)

    return output


def replaceBoundNameByMark(
    dm: str,
    boundMark: MeshFunction
) -> str:

    if '(' in dm and ')' in dm:
        # Get the boundary tag/numbering
        boundTag = dm[2:].strip('()')

        # Get the respective boundary numbering
        boundNum = str(boundMark[boundTag])\
            if not boundTag.isdigit()\
            else boundTag

        # Replace the boundary tag to your respective numbering
        dm = dm.replace(boundTag, boundNum)

    return dm


def getMax(*arr: Array, mesh: df.Mesh = None) -> (float):
    '''Get the max from array'''

    # Init a max value
    theMax = 0

    # Looping in 'arr's
    for a in arr:
        # Get the max convenently
        if type(a) in [df.Function, df.Expression]:

            if type(a) is df.Expression and mesh is None:
                # Get any mesh
                mesh = [a.function_space().mesh()
                        for a in arr if hasattr(a, 'function_space')]

                if any(mesh):
                    # Get any mesh
                    mesh = mesh[0]
                else:
                    raise ValueError(
                        f'The expression "{a.name()}" need one mesh'
                    )

            # Get the max on mesh vertex function values
            theMax = max(theMax, abs(a.compute_vertex_values(mesh)).max())

        elif type(a[0]) is df.Function:
            theMax = vectorize(
                lambda u: max(theMax, abs(u.compute_vertex_values(mesh)).max())
            )(a)

        elif type(a) is np.ndarray:
            theMax = max(theMax, abs(a).max(axis=0))

        else:
            theMax = max(theMax, abs(array(a)).max())

    return theMax


def getFiniteElementDegree(u: df.Function) -> (int):

    if hasattr(u, 'ufl_element'):
        # Get the degree
        degree = u.ufl_element().degree()

    elif len(u.ufl_operands) > 0:
        # Looping in operands
        for operand in u.ufl_operands:
            # Get the operand degree
            degree = getFiniteElementDegree(operand)

            if degree is not None:
                # Stop the loop
                break
    else:
        # Set a default value
        degree = None

    return degree


def showErrorTuple(
    errors: dict[str: tuple],
    precision: int = 3,
    copyTo: str = None
        ) -> None:
    # Set the error format
    format = '{:1.0%(prec)de} ({:1.0%(prec_1)s})'\
        % {'prec': precision, 'prec_1': f'{precision-1}%'}

    # Show the error
    showInfo(
        *[f'{eType} : {format}'.format(*error)
            for eType, error in errors.items()],
        alignment=':', delimiters=False, tab=4,
        breakStart=False, copyTo=copyTo
    )
    return None


def showErrorFloat(
    errors: dict[str: float],
    precision: int = 3,
    copyTo: str = None
        ) -> None:
    # Set the error format
    format = '{:1.0%de}' % precision

    # Show the error
    showInfo(
        *[f'{eType} : {format}'.format(error)
            for eType, error in errors.items()],
        alignment=':', delimiters=False, tab=4,
        breakStart=False, copyTo=copyTo
    )
    return None


def showErrorString(
    errors: dict[str:dict[str: Union[float, tuple]]],
    omg: dict[str:str],
    precision: int = 3,
    copyTo: str = None
        ) -> (None):
    # Looping in errors expression
    for solLbl, errors in errors.items():
        # Show the errors type
        showInfo(
            f"||{solLbl} - {solLbl.lower()}d||_X({omg[solLbl]})", copyTo=copyTo
        )

        # Show the errors values
        showErrorTuple(errors, precision, copyTo=copyTo)\
            if type(errors['L²']) is tuple\
            else showErrorFloat(errors, precision, copyTo=copyTo)

    return None


def showErrorTable(
    errors: dict[str:dict[str: Union[float, tuple]]],
    omg: dict[str:str],
    precision: int = 3,
    copyTo: str = None
        ) -> (None):

    # Check if there is error relative values
    relative = thereIsRelative(errors)

    # Get the error solutions label
    errorsLabels = errors.keys()

    # Set the labels format
    labelsFormat = '||{k:>2} - {k:>2}d||_X({omg})'

    # Get the error solutions label and add the respective domain
    errorsLabels = [labelsFormat.format(k=k, omg=omg[k]) for k in errors.keys()]

    # Set the error format
    errorFormat = '{:1.0%(prec)de} ({:1.0%(prec_1)s})'\
        % {'prec': precision, 'prec_1': f'{precision-1}%'}
    errorFormat = errorFormat.split(' ')

    # Get the error norms values
    errorsByNorm = []
    for k, error in enumerate(errors.values()):

        if k == 0:
            # Set the error norm names
            normNames = getErrorNormNames(list(error.keys()))

        # Turn error values to tuple
        values = tuple(error.values())

        # Set the null tuple complement
        complement = tuple([(0, 0)] * (3 - len(values)))\
            if relative\
            else (0,) * (3 - len(values))

        if not any(complement):
            # Set default value
            complement = tuple()

        # Join the values to complement
        values = np.array(values + complement, dtype=object)

        # Turn to row vector
        values = np.ravel(values)

        # Format values as string
        values[::2] = vectorize(lambda x: errorFormat[0].format(x))(
            values[::2]
            )
        values[1::2] = vectorize(lambda x: errorFormat[1].format(x))(
            values[1::2]
            )

        # Add respective error values
        errorsByNorm.append(values)

    # Turn to array
    errorsByNorm = np.array(errorsByNorm)

    # Turn to dict mapping norm name
    errorsByNorm = dict(
        [(name, errorsByNorm[:, k])
            for k, name in enumerate(normNames)]
    )

    # Turn to table
    errorsTable = table(errorsByNorm, index=errorsLabels)

    # Align the header to center
    setOption('colheader_justify', 'center')

    # Show the table
    print(errorsTable)

    if copyTo is not None:
        print(errorsTable, file=copyTo)

    return None


def showError(
    errors: dict[str:dict[str: Union[float, tuple]]],
    omg: dict[str:str] = 'Ω',
    table: bool = True,
    precision: int = 3,
    copyTo: str = None
        ) -> (None):

    if type(omg) is str:
        # Set default dict
        omg = {k: omg for k in errors.keys()}

    # Show error exhibition title
    showInfo('Approach error', copyTo=copyTo)

    # Show errors in string/dataframe mode
    showErrorString(errors, omg, precision, copyTo=copyTo)\
        if not table\
        else showErrorTable(errors, omg, precision, copyTo=copyTo)

    return None


def thereIsRelative(
    errors: dict[str:dict[str: Union[float, tuple]]]
) -> bool:
    return type(tuple(list(errors.values())[0].values())[0]) is tuple


def getErrorNormNames(
    names: list[str]
) -> list[str]:
    # Turn list to array and repeat twice your elements
    names = np.array(names, dtype='<U10').repeat(2)

    # Replace the odd values to % symbol
    names[1::2] = [names[2*k]+'(%)' for k in range(len(names[1::2]))]

    return names


def Zero(V: Union[df.FunctionSpace, df.Function]) -> df.Function:

    if type(V) is df.Function:
        # Get respective function space
        V = V.function_space()

    try:
        # Set the zero function
        zero = df.Function(V, name='Zero')
    except Exception:
        # Set the zero function
        zero = df.Function(V.collapse(), name='Zero')

    return zero


def getAdjointSystem(
    stateForm: str,
    trialTestFunc: list[Tuple[str, str]],
    gradJ_z: Form,
    Z: df.Function = None,
    globalVars: dict[str: Any] = None
        ) -> (Tuple[Form, Form]):

    # Get the variational form left side
    aAdjoint = replaceProgressive(stateForm, trialTestFunc)

    # Turn left side to form
    aAdjoint = eval(aAdjoint, globalVars)

    # Get the variational form right side
    LAdjoint = df.action(-gradJ_z, Z)\
        if Z is not None\
        else -gradJ_z

    return aAdjoint, LAdjoint


def getOptimalConditionsSystem(
    controls: list[str],
    LState: str,
    stateTestTrialAdjoint: list[Tuple[str, str]],
    gradJ_c: Form,
    liftingStateData: Form,
    globalVars: dict[str: Any] = None,
    timeDependent: bool = False,
    **gradientElements: dict[str: Union[tuple, dict]],
        ) -> (Tuple[Form, Form]):

    # Get gradient elements
    dJ, v, dm, Z_L = getMultipleKeys(
        gradientElements,
        ['dJ', 'v', 'dm', 'Z_L'],
        default=None
    )

    # Get the respectives control terms
    controlTerm = getFormTerm(LState, *controls, timeDependent=timeDependent)

    # Set the variational formulation to optimality conditions
    if Z_L is None:
        # Left side
        aOptimal = gradJ_c
        # Right side
        LOptimal = emptyForm()

    elif len(controls) > 1 and Z_L is None:
        # Left side
        aOptimal = sum(
            [df.dot(dJ[i], v[i])*dm[c]
                for i, c in enumerate(controls)]
        )
        # Right side
        LOptimal = -gradJ_c

    elif len(controls) > 1 and Z_L is not None:
        # Left side
        aOptimal = [
            df.dot(dJ[i], v[i])*dm[c]
                for i, c in enumerate(controls)
        ]
        # Right side
        LOptimal = [-gradJ_ci for gradJ_ci in gradJ_c]

    elif Z_L is None:
        # Left side
        aOptimal = df.dot(dJ[0], v[0])*dm[controls[0]]

        # Right side
        LOptimal = -gradJ_c

    else:
        # Left side
        aOptimal = [df.dot(dJ[0], v[0])*dm[controls[0]]]

        # Right side
        LOptimal = -gradJ_c\
            if type(gradJ_c) is not list\
            else [-gradJ_c[0]]

    if liftingStateData is not None:

        if Z_L is None:
            # Add to optimality conditions variational formulation right side
            LOptimal += -liftingStateData

        else:
            LOptimal[controls.index('ug')] += -df.action(
                liftingStateData, Z_L[0]
                )

    # Get the optimality conditions contributions from state equation
    stateOptimalContributions = [replaceProgressive(
        controlTerm[c],
        [(c, f'v[{i}+2]'), *stateTestTrialAdjoint],
        preserveds=[f'{c}n' for c in controls]
    ) for i, c in enumerate(controls)]

    # Evaluate it
    stateOptimalContributions = vectorize(eval)(
        stateOptimalContributions, globalVars
        )

    # Add to optimality conditions variational formulation right side
    LOptimal += vectorize(df.action)(stateOptimalContributions, Z_L[1])\
        if Z_L is not None\
        else stateOptimalContributions.sum()

    return aOptimal, LOptimal


def evaluateCost(
    J: Function,
    Z: Function,
    ud: Function,
    controls: dict[str: Function],
    a: dict[str: Tuple[df.Constant, df.Constant]],
    dm: dict[str: Measure],
    copyTo: File = None,
    show: bool = False,
    **costData: dict[str: Any]
        ) -> (float):

    # Evaluate the Cost
    cost = J(Z, ud, controls, a, dm, evaluate=True)

    if show or copyTo is not None:
        # Get the control variables labels
        controlsLabel = controls.keys()

        # Get cost data
        iters = costData.get('iters', None)
        rho = costData.get('rho', None)
        previousCost = costData.get('previousCost', None)
        stopCriterion = costData.get('stopCriterion', None)

        if previousCost is not None:
            # Calcule the relative cost
            relativeCost = abs(cost - previousCost)/abs(cost)

        # Set the print write file in kwargs
        kwargs = {}
        if copyTo is not None and not show:
            kwargs['outputName'] = copyTo
        elif copyTo is not None and show:
            kwargs['copyTo'] = copyTo
        elif not show:
            # Set a buffer file
            kwargs['outputName'] = open('./.buffer.txt', 'w')

        # Show the cost data
        showInfo('Cost Data', **kwargs)
        showInfo(
            f'ks = {iters}' if iters is not None else '',
            f'{rho = :1.03e}' if rho is not None else '',
            f'J(u, {", ".join(controlsLabel)}) = {cost:1.03e}'
            + (f' (J_1 = {previousCost:1.03e})'
                if (iters, previousCost) != (None, None) and iters[0] > 0
                else ''),
            f'|Jk - Jk_1|/|Jk| = {relativeCost:1.03e}'
                if (iters, previousCost) != (None, None) and iters[0] > 0
                else '',
            f'Stop criterion = {stopCriterion}'
                if stopCriterion is not None
                else '',
            alignment='=', breakStart=False, ignoreEmptyLines=True,
            delimiters=False, tab=4, **kwargs
        )

    return cost


def getLocal(U: Function) -> (Array):
    return U.vector().get_local()


def setLocal(U: Function, value: Union[Function, Array]) -> (Array):
    if type(value) is Function:
        # Get the function values
        value = getLocal(value)

    # Set the U value
    U.vector().set_local(value)

    return None


def formatTimeData(
    nelt: int,
    Tf: float,
    dt: Union[float, df.Constant],
    teta: Union[float, df.Constant],
    u0Expr: str = None
        ) -> (dict[str:str]):
    d = {
        f'nelt': nelt,
        f'Tf': f'{Tf:1.03e}',
        f'Δt': f'{float(dt):1.03e}',
        f'θ': f'{float(teta)}',
        f'u0': u0Expr
    }
    return d


def showProblemData(
    problemTitle: str,
    programMode: str,
    Th: Element,
    V: df.FunctionSpace,
    bcs: list[df.DirichletBC],
    ds: Measure,
    boundMark: dict[str: int],
    inputData: dict[str: str],
    normals: dict[str: array],
    g: df.Function = None,
    a_s: dict[str: Tuple[df.Constant, df.Constant]] = None,
    dm: dict[str: Measure] = None,
    timeData: dict[str: str] = {},
    copyTo: File = None
        ) -> (None):

    # Get the mesh
    mesh = V.mesh()

    # Extract the exact data
    exactData = {k: v for k, v in inputData.items()
                      if k[-1] == 'd'}

    # Remove the exact data from inputData
    [inputData.pop(k) for k in exactData]

    # Extract the boundary functions
    boundFuncs = {k: v for k, v in inputData.items()
                       if 'g' in k or 'h' in k}

    # Remove the boundary funcs from inputData
    [inputData.pop(k) for k in boundFuncs]

    # Get the finite element data
    femData = getFiniteElementData(Th, mesh)

    # Get the boundary data
    boundData = getBoundaryData(V, bcs, ds.subdomain_data(), boundMark)

    # Get any boundary data
    boundAmount, boundDofs, boundMarks, inletMode = getMultipleKeys(
        boundData,
        ['boundAmount', 'boundDofs', 'boundMarks', 'inletMode'],
        default=None
    )

    # Set the suffix to boundary values
    suffix = {
        'g': f"(Max: {getInletRangeValue(g)[1]})"
    } if g is not None else {}

    # Get the final instant time
    Tf = timeData.get('Tf', None)

    # Show the problem data by category
    showInfo('Problem Title', copyTo=copyTo)
    showInfo(problemTitle, delimiters=False, breakStart=False, tab=4,
             copyTo=copyTo)

    showInfo('Program Mode', copyTo=copyTo)
    showInfo(programMode, delimiters=False, breakStart=False, tab=4,
             copyTo=copyTo)

    showInfo("Boundary Data", copyTo=copyTo)
    showInfo(
        f"Amount = {sum(boundAmount)}" + " (D: {}, N: {})".format(*boundAmount),
        f"Inlet mode = {inletMode}" if inletMode is not None else '',
        *[f"Dof in {bound} = {dof}"
            for bound, dof in sorted(boundDofs.items(), key=lambda x: x[0])],
        f'marks = {boundMarks}',
        alignment='=', delimiters=False, breakStart=False,
        ignoreEmptyLines=True, tab=4, repeatMsg="Dof in ",
        copyTo=copyTo
    )

    if any(boundFuncs):
        showInfo("Boundary Values", copyTo=copyTo)
        showInfo(
            *[f'{f} = {fExpr} {suffix.get(f, "")}'
                for f, fExpr in boundFuncs.items()],
            alignment='=', delimiters=False, breakStart=False,
            ignoreEmptyLines=True, tab=4, copyTo=copyTo
        )

    showInfo("Boundary Area", copyTo=copyTo)
    showInfo(
        *[f'|ds("{bound}")| = {getBoundaryArea(mesh, ds, mark):1.02e}'
            for bound, mark in boundMarks.items()],
        alignment='=', delimiters=False, breakStart=False,
        ignoreEmptyLines=True, tab=4, copyTo=copyTo
    )

    if any(exactData):
        showInfo("Exact Data", copyTo=copyTo)
        showInfo(
            *[f'{udName} = {udExpr}'
                for udName, udExpr in exactData.items()],
            alignment='=', delimiters=False, breakStart=False,
            ignoreEmptyLines=True, tab=4, copyTo=copyTo
        )

    if any(inputData):
        showInfo("Input Data", copyTo=copyTo)
        showInfo(
            *[f'{f} = {fExpr}'
                for f, fExpr in inputData.items()],
            alignment='=', delimiters=False, breakStart=False,
            ignoreEmptyLines=True, tab=4, copyTo=copyTo
        )

    showInfo("Normals", copyTo=copyTo)
    showInfo(
        *[f'{k} = {n} '
            + (f'(|n| = {(n**2).sum()**0.5:1.02})'
                if type(n) in [list, tuple, np.ndarray]
                else '')
            for k, n in normals.items()],
        alignment='=', delimiters=False, breakStart=False,
        ignoreEmptyLines=True, tab=4, copyTo=copyTo
    )

    showInfo("Finite Element Data", copyTo=copyTo)
    showInfo(
        *[f'{key} = {value}'
            for key, value in femData.items()],
        alignment='=', delimiters=False, breakStart=False,
        ignoreEmptyLines=True, tab=4, copyTo=copyTo
    )

    if any(timeData):
        showInfo("Time Partition Data", copyTo=copyTo)
        showInfo(
            *[f'{key} = {value}'
                for key, value in timeData.items()],
            alignment='=', delimiters=False, breakStart=False,
            ignoreEmptyLines=True, tab=4, copyTo=copyTo
        )

    if (a_s, dm) != (None, None):
        showInfo("Cost parameters", copyTo=copyTo)
        showInfo(
            *[f'a_{c} = {tuple(map(float, value.values()))}'
                for c, value in (a_s).items()],
            *[f'dm_{c} = {getMeasureName(value)}'
                for c, value in (dm).items()],
            alignment='=', delimiters=False, breakStart=False,
            ignoreEmptyLines=True, tab=4, copyTo=copyTo
        )

    return None


def getInputExactData(
    globalVars: dict[str: Any]
        ) -> (dict[str: MyExpression]):

    # Get the expressions variables
    exprVars = filter(lambda x: x[-4:] == 'Expr', globalVars)

    # Init a output dict
    inputDataFunctions = {}

    # Looping in expression variables
    for expr in exprVars:
        # Get respective expression
        inputDataFunctions[expr.replace('Expr', '')] = globalVars[expr]

    return inputDataFunctions


def getFunctionExpressions(
    funcs: dict[str: MyExpression]
        ) -> (dict[str: str]):
    # Init a output dict
    funcsExpr = {}

    # Looping in pair (k, v) from funcs
    for name, func in funcs.items():
        # Get the respective func expression
        funcsExpr[name] = func._def

    return funcsExpr


def getInletRangeValue(
    g: Function
        ) -> (float):

    if not any(g.value_shape()):
        # Get the function values
        gValues = getLocal(g)

        # Get the min, max in function values
        gMin = gValues.min()
        gMax = gValues.max()

    else:
        # Split the vector coordinates
        gCoord = array(
            [gi.vector()[:] for gi in g.split()]
        )

        # Calcule the vector norms
        gNorms = (gCoord**2).sum(axis=0)**0.5

        # Get min, max in vector norms
        gMin = gNorms.min()
        gMax = gNorms.max()

    return gMin, gMax


def getFiniteElementData(
    Th: Element,
    mesh: df.Mesh
        ) -> (dict[str:str]):

    # Get elements in string
    Th = turnElementToString(Th)

    d = {
        'Th': Th,
        'nel': convertInt(mesh.num_cells()),
        'h': f'{mesh.hmax():1.03e}',
        'cell': mesh.ufl_cell(),
    }

    return d



def turnElementToString(
    element: Element
        ) -> (str):

    # Get the subelements
    subs = element.sub_elements()\
        if 'Mixed' in type(element).__name__\
        else []

    if any(subs):
        elementString = []
        for sub in subs:
            elementString.append(
                turnElementToString(sub)
            )

    else:
        # Get the element degree
        deg = element.degree()

        # Get the value shape
        shape = element.value_shape()

        # Get the basis family short version
        family = shortFamily(element.family())

        # Build the string
        elementString = f'{family}{deg}'\
            + (f'[{shape[0]}]' if any(shape) else '')

    if type(elementString) is list:
        # Join the elements with x
        elementString = ' x '.join(elementString)

    return elementString


def shortFamily(family: str) -> (str):
    short = {
        'Lagrange': 'P',
        'Hermite': 'HER',
        'Discontinuous Lagrange': 'DG'
    }[family]

    return short


def norm(
    U: df.Function,
    dm: df.Measure = df.dx,
    type: str = 'l2'
        ) -> (float):

    # Get the norm formula
    normForm = getNormFormula(U, type, dm)

    # Evaluate the norm form
    normValue = assemble(normForm)**0.5

    return normValue


def appendErrorsByTime(
    errorsByTime: dict[str: dict[str:list]],
    errors: dict[str: dict[str:float]]
        ) -> (None):

    for k, errors_k in errorsByTime.items():
        for eType, error in errors[k].items():
            errors_k[eType].append(error)

    return None
