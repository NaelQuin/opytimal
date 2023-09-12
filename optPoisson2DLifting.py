'''
Optimal Poisson-based problem on 2D domain
'''

# from modules import *
# from settings import settings

from dolfin import *

from backend.settings import QUADRATURE_DEG
from backend.symbols import x, y, z, t, symbols
from backend.string import showInfo, splitPathFile, replaceProgressive, basename
from backend.meshes import (readXDMFMesh, getInnerNodes, getSubMeshes,
                            getCoordinates, getNormal)
from backend.plots import (plotMesh, plotComparison, adjustFiguresInScreen,
                           show, figure)
from backend.tests import testLoop
from backend.fenics import (setExpression, extractElements, calculeNnz, mySolve,
                            setSolver, getErrorFormula, gradJ, copySolver,
                            getFormTerm, emptyForm, gradientDescent, showError,
                            getDomainLabels, replaceBoundNameByMark,
                            getMeasure, evaluateErrors, Zero, getAdjointSystem,
                            getOptimalConditionsSystem, evaluateCost,
                            getLocal, setLocal, showProblemData,
                            getInputExactData, getFunctionExpressions)
from backend.analytical import (AnalyticalFunction, AnalyticalVectorFunction)
from backend.parallel import parallel
from backend.arrays import (identity, zeros, getComplement, array, concatenate)
from backend.types import Tuple, Union

# ================
# Input Data Begin
# ================
num = 6
meshPath = f'./InputData/Meshes/2D/rectangle{num}'
boundaryDataPath = f'./InputData/Meshes/2D/rectangle{num}_mf'

invertNormalOrientation = True

showMesh = False

exportSolutionsPath = (
    '/output/'.join(splitPathFile(__file__))
    ).replace('.py', '.pvd')

showSolution = True

# Write a copy for all prints in the below file
externalTxt = {
    1: f'./output/{basename(__file__).replace(".py", "")}_data.txt',
    2: None
}[1]

# Finite Element basis and degree
Th = 'P1'

# ================
# Optimal Settings
# ================
# Controls variables
controls = {
    1: ['f', 'ug', 'h'], # Total control
    2: ['ug', 'h'], # Boundary controls
    3: ['f', 'h'], # Mixed Controls (Neumann)
    4: ['f', 'ug'], # Mixed Controls (Dirichlet)
    5: ['ug'],
    6: []
    }[4]

linSolver = 'tfqmr'
preconditioner = ['none', 'jacobi'][1]

# System solve mode
allAtOnce = True

# Cost coefficients
a_s = {
    'z': (1, 1e-1),
    'f': (0, 1e-2),
    'ug': (0, 1e-3),
    'h': (0, 1e-4)
}

# Cost measures
dm = {
    'z': 'dx',
    'f': 'dx',
    'ug': 'ds(inlet)',
    'h': 'ds(outlet)'
}

# Descent step size
rho = 10

# Previous direction weight
gamma = {
    1: '1',
    2: '1/8 * norm(dJk)**2 / norm(dJk_1)**2', # The Best
    3: '1/8 * (dJk * (dJk - dJk_1)) / norm(dJk_1)**2'
    }[2]

# ==========
# Exact Data
# ==========
def ud(*x):
    return x[1]*(2 - x[1]) #+ x[0]**2 + x[1]


def fd(*x):
    return -_ud.divgrad(*x)


def ugd(*x):
    'Dirichlet value on Γ_D'
    return _ud(*x)


def hd(*x):
    'Neumann value on Γ_N'
    return _ud.grad(*x) @ normals['outlet']

# ==========
# Input Data
# ==========
def gW(*x):
    'Dirichlet value on Γ_W'
    return _ud(*x)

if 'f' not in controls:
    def f(*x):
        'Source term'
        return -_ud.divgrad(*x)

if 'ug' not in controls:
    def g(*x):
        'Dirichlet value on Γ_D'
        return _ud(*x)

if 'h' not in controls:
    def h(*x):
        'Neumann value on Γ_N'
        return _ud.grad(*x) @ normals['outlet']

# ===============
# Inital Controls
# ===============
if not allAtOnce:
    # Exact percentage
    percent = 0. # in [0, 1]

    def f0(*x):
        return percent*(-_ud.divgrad(*x))


    def ug0(*x):
        'Dirichlet value on Γ_D'
        return percent*_ud(*x)


    def h0(*x):
        'Neumann value on Γ_N'
        return percent*(_ud.grad(*x) @ normals['outlet'])

# ===============
# Cost Functional
# ===============
def normH1a(u, dx):
    return [1/2*u**2*dx, 1/2*grad(u)**2*dx]


def normH1aDiff(u, dx, v):
    return [dot(u, v)*dx, inner(grad(u), grad(v))*dx]


def J(
    z: Function,
    ud: Function,
    controls: dict[str: Function],
    a: dict[str: Tuple[Constant, Constant]],
    dm: dict[str: Measure],
    evaluate: bool = True
        ) -> (Union[Form, float]):

    if type(z) is tuple:
        # Get the first state solutions
        z = z[0]

    # Get the lift
    ug = controls.get('ug', Zero(z.function_space()))

    # Set the cost function expression
    expression = a['z'].values() @ normH1a(z + ug - ud, dm['z'])
    expression += sum(
        [a[cName].values() @ normH1a(c, dm[cName])
            for cName, c in controls.items()]
    )

    # Set the output
    output = assemble(expression)\
        if evaluate\
        else expression

    return output


def gradJ(*aud, v):
    # Init grad form with a empty form
    grad = emptyForm()

    # Looping in tuple (coeff, func, measure)
    for a, u, dm in aud:
        # Eval the norm differentiating in respective tuple
        grad += a.values() @ normH1aDiff(u, dm, v)

    return grad

# ==============
# Input Data End
# ==============

if externalTxt is not None:
    # Create the txt file
    externalTxt = open(externalTxt, 'w', encoding='utf-8')

# Turn to analytical function object
_ud = AnalyticalFunction(ud(x, y, z), toccode=True)

# Check if is a optimal problem
optimal = any(controls)

# Load the mesh and boundary data
mesh, boundaryData, _ = readXDMFMesh(meshPath, boundaryDataPath)

# Get the geometric dimension
gdim = mesh.geometric_dimension()

# Get the boundary mesh
bmesh = BoundaryMesh(mesh, 'local', True)

# Get the facet normal element
n = FacetNormal(mesh)

# Set the boundary data for the measures
dx = Measure(
    "dx",
    domain=mesh,
    metadata={"quadrature_degree": QUADRATURE_DEG}
)
ds = Measure(
    "ds",
    domain=mesh,
    subdomain_data=boundaryData,
    metadata={"quadrature_degree": QUADRATURE_DEG}
)

# Get the boundary marks
boundaryMark = {
    'inlet': 1,
    'outlet': 2,
    'wall': 3
}

# Set the respective boundary measures
dsInlet = ds(boundaryMark['inlet'])
dsOutlet = ds(boundaryMark['outlet'])

# Looping in controls
for c in a_s.keys():
    # Turn to respective coefficients to constant
    a_s[c] = Constant(a_s[c], name=f"a_{c}")

    # Evaluate the respective cost measure
    dm[c] = eval(replaceBoundNameByMark(f"{dm[c]}", boundaryMark))

# Get the mesh inner nodes
innerNodes = getInnerNodes(mesh, bmesh)

# Get the boundaries submesh
boundarySubMeshes = getSubMeshes(boundaryData, boundaryMark)

# Get the normals
normals = {bound: getNormal(boundMesh)
              for bound, boundMesh in boundarySubMeshes.items()}

if invertNormalOrientation:
    # Invert each normal orientation
    normals = {k: v*-1 for k, v in normals.items()}

if showMesh:
    # Plot the mesh and bmesh nodes by category
    plotMesh(
        innerNodes.T,
        *[subMesh.coordinates().T
            for subMesh in boundarySubMeshes.values()],
        labels=['inner', *boundarySubMeshes.keys()]
    )

# ----------------------------
# Set the Finite Element Space
# ----------------------------
# Extract elements
elements = extractElements(Th, mesh.ufl_cell())

if allAtOnce:
    # Couple the adjoint and optimality conditions problems
    # to the state problem
    elements += optimal*elements + len(controls)*[elements[0]]

elif not optimal:
    raise ValueError("The gradient descent method cannot run without controls")

# Set the Finite Element Space
Th = MixedElement(elements)\
    if allAtOnce and optimal\
    else elements[0]

# Set the function space
W = FunctionSpace(mesh, Th)

if allAtOnce and optimal:
    # Get the state and adjoint subspaces
    V, VAdj = W.split()[:2]

    # Get the controls subspaces
    VOpt = dict(zip(controls, W.split()[2:]))

elif not optimal:
    # Set the state spaces
    V = W

    # Set default values
    VAdj = None
    VOpt = {}

else:
    # Set the control functions space
    WControl = [
        FunctionSpace(mesh, elements[0])
            for _ in range(len(controls))
    ]

    # Set the state and adjoint space
    V = VAdj = W

    # Set the controls subspaces
    VOpt = dict(zip(controls, WControl))

if allAtOnce and optimal:
    # Collapse the respective spaces
    Vc = V.collapse()
    VAdjC = VAdj.collapse()

else:
    # Get state space
    Vc = VAdjC = V

VOptC = dict((c, S.collapse()) for c, S in VOpt.items())\
    if allAtOnce\
    else VOpt

# Get the function space dofs and store it in itself
W.dofs = len(W.dofmap().dofs())

# Get the matrix nonzero amount
thrdNnz, W.nnz = parallel(calculeNnz, W, dx)

# Set the test functions empty lists
v = (1 + optimal + len(controls))*[0]

# Set the Trial and Test Functions
if allAtOnce and optimal:
    # Get the state and adjoint trial functions
    z, l_z = TrialFunctions(W)[:2]

    # Get the optimality conditions trial functions
    c = dict(zip(controls, TrialFunctions(W)[2:]))

    # Get the state and adjoint test functions
    v[0], v[1] = TestFunctions(W)[:2]

    # Get the optimality conditions test functions
    v[2:] = TestFunctions(W)[2:]

elif not optimal:
    # Get the state trial functions
    z = TrialFunction(W)

    # Get the state test functions
    v[0] = TestFunction(W)

else:
    # Get the state and adjoint trial functions
    z, l_z = 2*TrialFunctions(W)

    # Get the control solution vectors
    C = [Function(S, name=c.upper())
            for c, S in zip(controls, WControl)]

    # Get the control solution vectors
    c = dict(zip(controls, C))

    # Get the state and adjoint test functions
    v[0], v[1] = 2*TestFunctions(W)

    # Get the gradient cost trial function
    dJ = [TrialFunction(S) for S in WControl]

    # Get the optimality conditions test functions
    v[2:] = [TestFunction(S) for S in WControl]

# Turn input functions to expression
udExpr = setExpression(_ud, elements[0], name='ud')
gWExpr = setExpression(gW, elements[0], name='gW')
if 'f' not in controls:
    fExpr = setExpression(f, elements[0], name='f')
if 'ug' not in controls:
    gExpr = setExpression(g, elements[0], name='g')
if 'h' not in controls:
    hExpr = setExpression(h, elements[0], name='h')

# Set the current input functions
ud = interpolate(udExpr, Vc); ud.rename('ud', 'ud')
gW = interpolate(gWExpr, Vc); gW.rename('gW', 'gW')
if 'f' not in controls:
    f = interpolate(fExpr, Vc); f.rename('f', 'f')
if 'ug' not in controls:
    g = interpolate(gExpr, Vc); g.rename('g', 'g')
if 'h' not in controls:
    h = interpolate(hExpr, Vc); h.rename('h', 'h')

# Looping in controls
for cLbl in controls:
    # Set the respective control input expressions
    exec(f"{cLbl}dExpr = setExpression({cLbl}d, elements[0], name='{cLbl}d')")

    # Set the respective input data functions
    exec(f"{cLbl}d = interpolate({cLbl}dExpr, VOptC['{cLbl}'])")
    exec(f"{cLbl}d.rename('{cLbl}d', '{cLbl}d')")

    if not allAtOnce:
        # Set the respective initial control expressions
        exec(f"{cLbl}0Expr = setExpression({cLbl}0, elements[0], name='{cLbl}0')")

        # Set the respective intial control functions
        exec(f"{cLbl}0 = interpolate({cLbl}0Expr, VOptC['{cLbl}'])")
        exec(f"{cLbl}0.rename('{cLbl}0', '{cLbl}0')")

        # Group intial controls by category
        initialControls = {c: eval(f'{c}0') for c in controls}

# Group by category the exact data
exactData = {'ud': ud}\
    | {c: eval(f'{c}d') for c in controls}

# Loop in controls name
for cLbl in controls:
    # Set the respective control trial/vector function
    exec(f'{cLbl} = c["{cLbl}"]')

# Get a controls list copy
_controls = controls

# Turn controls list to dictionary
controls = dict((c, eval(c)) for c in controls)

if allAtOnce:
    # Set the problem coupled solution
    ZLZC = Function(W, name='ZLZC')
else:
    # Set the state problem solution
    Z = Function(W, name='Z')

    # Set the adjoint problem solution
    LZ = Function(W, name='LZ')

# Set the state problem solutions
U = Function(Vc, name='U')

# Set the variational system
aState = 'dot(grad(z), grad(v[0]))*dx'
LState = 'dot(f, v[0])*dx + dot(h, v[0])*dsOutlet'
if 'ug' in controls:
    LState += '- dot(grad(ug), grad(v[0]))*dx'

if optimal:
    # Get the gradJ with respect to state var
    gradJ_z = gradJ((a_s['z'], z + ug - ud, dm['z']), v=v[1])\
        if 'ug' in controls\
        else gradJ((a_s['z'], z - ud, dm['z']), v=v[1])\

    # Get adjoint variational system
    aAdjoint, LAdjoint = getAdjointSystem(
        aState,
        [('z', 'v[1]'), ('v[0]', 'l_z')],
        gradJ_z,
        Z if not allAtOnce else None,
        globals()
        )

    # Get the gradJ form for each control
    gradsJ = [
        gradJ((a_s[c], controls[c], dm[c]), v=v[2+i])
            for i, c in enumerate(controls)
        ]

    if 'ug' in controls:
        # Get ug index
        ugIdx = _controls.index('ug')

        if allAtOnce:
            # Sum the forms
            gradsJ = sum(gradsJ)

            # Add the cost lift contribution
            gradsJ += gradJ((a_s['z'], ug, dm['z']), v=v[2 + ugIdx])

        else:
            # Add the cost lift contribution
            gradsJ[ugIdx] += gradJ((a_s['z'], ug, dm['z']), v=v[2 + ugIdx])

        # Set the dirichlet lifting contribution with the state
        # and the input data terms
        liftingStateData = gradJ((a_s['z'], z - ud, dm['z']), v=v[2 + ugIdx])

    else:
        # Sum the forms
        gradsJ = sum(gradsJ)

        # Set default value
        liftingStateData = None

    # Get the optimality conditions variational system
    aOptimal, LOptimal = getOptimalConditionsSystem(
        _controls,
        LState,
        [('v[0]', 'l_z')],
        gradsJ,
        liftingStateData,
        **(dict(dJ=dJ, v=v[2:], dm=dm, Z_L=(Z, LZ))
            if not allAtOnce
            else {}),
        globalVars=globals()
    )

else:
    # Set empty forms
    aAdjoint = LAdjoint = aOptimal = LOptimal = emptyForm()

# Evaluate the string formulations
aState, LState = map(eval, [aState, LState])

# Set the inlet dirichlet value
inletDirichlet = 0\
    if 'ug' in controls\
    else g

# Set the state boundary conditions
bcsState = [
    DirichletBC(V, inletDirichlet, boundaryData, boundaryMark['inlet']),
    DirichletBC(V, gW, boundaryData, boundaryMark['wall'])
]

# Set the adjoint boundary conditions
bcsAdj = [
    DirichletBC(VAdj, 0, boundaryData, boundaryMark['inlet']),
    DirichletBC(VAdj, 0, boundaryData, boundaryMark['wall'])
] if optimal else []

if allAtOnce:
    # Append dirichlet boundary conditions of the adjoint problem
    bcs = bcsState + bcsAdj

else:
    # Split dricihlet boundary conditions by problem
    bcs = {'state': bcsState, 'adjoint': bcsAdj}

# # Set the preconditioner
# preconditioner = ['none', 'jacobi'][1]

# Set the solver
#solver = setSolver('tfqmr', preconditioner)
#solver = setSolver('mumps')
solver = setSolver(linSolver, preconditioner)

# Set the stabilizator scalar
beta = Constant(mesh.hmax()**2, name='β')

if 'h' in controls and allAtOnce:
    # Add the stabilizator term
    aOptimal += beta*dot(grad(h), grad(v[2+_controls.index('h')]))*dx

elif 'h' in controls:
    # Get the h index
    hIdx = _controls.index('h')

    # Add the stabilizator term
    aOptimal[hIdx] += beta*dot(grad(dJ[hIdx]), grad(v[2+hIdx]))*dx

# Get the trial functions of the collapsed subspaces
# (for the approach errors calculus)
u = TrialFunction(V)
c = {c: TrialFunction(VOptC[c]) for c in controls}

# Set the error formulas
errors = {
    'u': getErrorFormula(ud, dm['z'], relative=True),
    **{c: getErrorFormula(eval(f'{c}d'), dm[c], relative=True)
            for c in controls}
}

# Set the domain labels
omg = getDomainLabels(
    {k: getMeasure(v['L²'][0].integrals()[0])
        for k, v in errors.items()},
    boundaryMark
    )

# Stop the nnz calculation thread
thrdNnz.join()

# Get value from list
W.nnz = W.nnz[0]

# Show the program data
showProblemData(
    f'Optimal Stokes-based Problem on {basename(meshPath)}',
    'validation' if _ud is not None else 'simulation',
    Th, W, bcsState if allAtOnce else bcs['state'],
    ds, boundaryMark,
    getFunctionExpressions(getInputExactData(globals())),
    normals,
    g if 'ug' not in controls else None,
    a_s,
    dm,
    copyTo=externalTxt
)

if allAtOnce:
    # Couple the variational formulations
    a, L = system(
        aState + aAdjoint + aOptimal
        - (LState + LAdjoint + LOptimal)
    )

    # Solve the all at once system
    mySolve(a == L, ZLZC, bcs, solver,
            runtime=True, nnz=W.nnz, dofs=W.dofs,
            copyTo=externalTxt)

else:
    # Split the problems
    a = [aState, aAdjoint, aOptimal]
    L = [LState, LAdjoint, LOptimal]
    w = [Z, LZ, C]

    # Get a solver copy
    solverCopy = copySolver(solver)

    # Solve the system by gradient descent method
    errors = gradientDescent(
        _controls, J, a_s, dm, *zip(a, L, w), bcs, (solver, solverCopy),
        exactData, initialControl=initialControls, rho=rho,
        gamma=gamma, errorForm=errors, copyTo=externalTxt
    )[2]

# Split the solutions
if allAtOnce and optimal:
    # Get splitted solutions
    ZLZC = ZLZC.split(deepcopy=True)

    # Get solutions by problem
    Z, LZ = ZLZC[:2]  # State and adjoint
    C = dict(zip(controls, ZLZC[2:])) # Controls

elif not optimal:
    # Get solution
    Z = ZLZC

else:
    # Split the control solutions
    C = dict(zip(controls, C))

if optimal:
    # Rename the adjoint and controls solutions
    LZ.rename('Lz', 'Lz')
    [C[c].rename(c, c) for c in controls]

# Recover the original U solutions
U.assign(Z)

if 'ug' in controls:
    # Add the lifting contribution
    setLocal(U, getLocal(U) + getLocal(C['ug']))

if allAtOnce and optimal:
    # Evaluate and show the cost
    evaluateCost(J, Z, ud, C, a_s, dm, show=True, copyTo=externalTxt)

    # Calcule approach errors
    errors = evaluateErrors(
        (U, errors['u']),
        *[(C[c], errors[c]) for c in controls],
        labels=['u', *controls],
        relative=type(errors['u']['L²']) is tuple
        )

elif not optimal:
    # Calcule approach errors
    errors = evaluateErrors(
        (U, errors['u']),
        labels=['u'],
        relative=type(errors['u']['L²']) is tuple
        )

# Show the approach error
showError(errors, omg, precision=6, copyTo=externalTxt)

import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12
plt.rcParams['legend.fontsize']=15
plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 3


if showSolution:
    # Set the common args
    commonArgs = {
        'splitSols': True,
        'show': False,
        'interactive': False,
        'personalPlot': False
    }

    # Plot the solution comparison
    fig = plotComparison(
        ud, U, **commonArgs
    )

    figs = [fig]

    for c in controls:
        fig = plotComparison(
            exactData[c], C[c], **commonArgs
        )

        figs.append(fig)

    adjustFiguresInScreen(*figs)

    show()

# Create a file to export solutions
file = File(exportSolutionsPath)

# Export the solution
file.write(U)

if externalTxt is not None:
    # Close the external txt file
    externalTxt.close()
