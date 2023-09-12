'''
This script solve the optimal control problem subject
to Heat equation, considering multiple controls
'''

# from modules import *
# from settings import settings

from dolfin import *
import numpy as np
import sympy as sp

from backend.mathFunctions import sin, cos, tan, exp, log, ln
from backend.settings import QUADRATURE_DEG
from backend.symbols import x, y, z, t, symbols
from backend.string import showInfo, splitPathFile, replaceProgressive, basename, showErrors
from backend.tests import testLoop
from backend.analytical import AnalyticalFunction, AnalyticalVectorFunction
from backend.parallel import parallel
from backend.types import Tuple, Union
from backend.profiler import ProgressBar
from backend.numeric import getPow10
from backend.arrays import (identity, zeros, getComplement, array, concatenate,
                            norm as arrNorm)
from backend.plots import (plotMesh, plotComparison, adjustFiguresInScreen,
                           show, figure, dynamicComparison)
from backend.meshes import (readXDMFMesh, getInnerNodes, getSubMeshes,
                            getCoordinates, getNormal)
from backend.fenics import (setExpression, extractElements, calculeNnz, mySolve,
                            setSolver, getErrorFormula, gradJ, copySolver,
                            getFormTerm, emptyForm, gradientDescent, showError,
                            getDomainLabels, replaceBoundNameByMark,
                            getMeasure, evaluateErrors, Zero, getAdjointSystem,
                            getOptimalConditionsSystem, evaluateCost,
                            getLocal, setLocal, showProblemData, 
                            getInputExactData, getFunctionExpressions,
                            appendErrorsByTime)

# Global value
THREADS = []

# ================
# Input Data Begin
# ================
Tf = 1
teta = 0.5
dt = {
    1: "mesh.hmax()",
    2: 1e-3
}[1]

meshPath = './InputData/Meshes/2D/rectangle4'
boundaryDataPath = './InputData/Meshes/2D/rectangle4_mf'

invertNormalOrientation = True

showMesh = False

exportSolutionsPath = {
    'U': '/output/'.join(splitPathFile(__file__)).replace('.py', '_U.pvd'),
    'ud': '/output/'.join(splitPathFile(__file__)).replace('.py', '_ud.pvd'),
    }

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
    5: ['f'],
    6: []
    }[4]

linSolver = 'tfqmr'
preconditioner = ['none', 'jacobi'][1]

# System solve mode
allAtOnce = True

# Cost coefficients
a_s = {
    'z': (10, 1e-1),
    'f': (1e-5, 1e-6),
    'ug': (1e-5, 1e-1),
    'h': (0, 10)
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
def ud(*x, t=0):
    return (x[1]*(2 - x[1]) + x[0]*(4 - x[0])) * exp(-t)

def u0(*x):
    'Inital solution'
    return _ud(*x, t=0)

if allAtOnce:
    def f0(*x):
        return -_ud.divgrad(*x, t=0)

    def ug0(*x):
        'Dirichlet value on Γ_D'
        return _ud(*x, t=0)

    def h0(*x):
        'Neumann value on Γ_N'
        return _ud.grad(*x, t=0) @ normals['outlet']

def fd(*x, t=0):
    return _ud.dt(*x, t=t) - _ud.divgrad(*x, t=t)

def ugd(*x, t=0):
    'Dirichlet value on Γ_D'
    return _ud(*x, t=t)

def hd(*x, t=0):
    'Neumann value on Γ_N'
    return _ud.grad(*x, t=t) @ normals['outlet']

# ==========
# Input Data
# ==========
def gW(*x, t=t):
    'Dirichlet value on Γ_W'
    return _ud(*x, t=t)

if 'f' not in controls:
    def f(*x, t=t):
        'Source term'
        return _ud.dt(*x, t=t) - _ud.divgrad(*x, t=t)

if 'ug' not in controls:
    def g(*x, t=t):
        'Dirichlet value on Γ_D'
        return _ud(*x, t=t)

if 'h' not in controls:
    def h(*x, t=t):
        'Neumann value on Γ_N'
        return _ud.grad(*x, t=t) @ normals['outlet']

# ===============
# Inital Controls
# ===============
if not allAtOnce:
    # Exact percentage
    percent = 0. # in [0, 1]

    if 'f' in controls:
        def f0(*x, t=t):
            return percent*(-_ud.divgrad(*x, t=t))

    if 'ug' in controls:
        def ug0(*x, t=t):
            'Dirichlet value on Γ_D'
            return percent*_ud(*x, t=t)

    if 'h' in controls:
        def h0(*x, t=t):
            'Neumann value on Γ_N'
            return percent*(_ud.grad(*x, t=t) @ normals['outlet'])

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
_ud = AnalyticalFunction(ud(x, y, z, t=t), toccode=True)

input(_ud._grad)

# Check if is a optimal problem
optimal = any(controls)

# Load the mesh and boundary data
mesh, boundaryData, _ = readXDMFMesh(meshPath, boundaryDataPath)

if type(dt) is str:
    # Evaluate the dt choice
    dt = eval(dt)

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
for c in list(a_s.keys()):

    if c not in controls and c != 'z':
        # Remove respective key
        a_s.pop(c)
        dm.pop(c)

        # Go to next
        continue

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
udExpr = setExpression(_ud, elements[0], name='ud', t=True)
gWExpr = setExpression(gW, elements[0], name='gW', t=True)
u0Expr = setExpression(u0, elements[0], name='u0')
f0Expr = setExpression(f0, elements[0], name='f0')
ug0Expr = setExpression(ug0, elements[0], name='ug0')
h0Expr = setExpression(h0, elements[0], name='h0')
if 'f' not in controls:
    fExpr = setExpression(f, elements[0], name='f', t=True)
if 'ug' not in controls:
    gExpr = setExpression(g, elements[0], name='g', t=True)
if 'h' not in controls:
    hExpr = setExpression(h, elements[0], name='h', t=True)

# Set the current input time functions
ud = interpolate(udExpr, Vc); ud.rename('ud', 'ud')
gW = interpolate(gWExpr, Vc); gW.rename('gW', 'gW')
if 'f' not in controls:
    f = interpolate(fExpr, Vc); f.rename('f', 'f')
    f0 = interpolate(f0Expr, Vc); f0.rename('f', 'f0')
if 'ug' not in controls:
    g = interpolate(gExpr, Vc); g.rename('g', 'g')
else:
    ug0 = interpolate(ug0Expr, Vc); ug0.rename('ug0', 'ug0')
if 'h' not in controls:
    h = interpolate(hExpr, Vc); h.rename('h', 'h')
    h0 = interpolate(h0Expr, Vc); h0.rename('h', 'h0')

# Set the respectives previous time functions
zn = Function(Vc, name='zn')
u0 = Function(Vc, name='u0')
fn = Function(Vc, name='fn')
if 'ug' in controls:
    ugn = Function(Vc, name='ugn')
hn = Function(Vc, name='hn')

# Set previous time function as initials respective values
u0.assign(interpolate(u0Expr, Vc))
if 'f' not in controls:
    fn.assign(interpolate(fExpr, Vc))
if 'h' not in controls:
    hn.assign(interpolate(hExpr, Vc))
# if 'ug' in controls:
#     zn.assign(u0 - ug0)
# else:
#     zn.assign(u0)
zn.assign(u0)

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

        # Set the respective previous time functions
        exec(f"{cLbl}n.assign(interpolate({cLbl}0Expr, VOptC['{cLbl}'])")

        # Group intial controls by category
        initialControls = {c: eval(f'{c}0') for c in controls}

# Group by category the exact data
exactData = {'ud': ud}\
    | {c: eval(f'{c}d') for c in controls}

exactExpressions = {'ud': udExpr}\
    | {c: eval(f'{c}dExpr') for c in controls}

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

# Turn input scalars to constant
teta = Constant(teta, name='θ')
_dt = dt # Store dt float
dt = Constant(dt, name='Δt')

# Set the variational problem
aState = 'z*v[0]*dx + dt*teta*dot(grad(z), grad(v[0]))*dx'
LState = 'zn*v[0]*dx\
    + dt*teta*f*v[0]*dx + dt*(1-teta)*fn*v[0]*dx \
    + dt*teta*h*v[0]*dx + dt*(1-teta)*hn*v[0]*dsOutlet\
    - dt*(1-teta)*dot(grad(zn), grad(v[0]))*dx'
if 'ug' in controls:
    LState += '- (ug - ugn)*v[0]*dx \
        - dt*teta*dot(grad(ug), grad(v[0]))*dx\
        - dt*(1-teta)*dot(grad(ugn), grad(v[0]))*dx'

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
        globalVars=globals(),
        timeDependent=True,
        **(dict(dJ=dJ, v=v[2:], dm=dm, Z_L=(Z, LZ))
            if not allAtOnce
            else {})
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

# Set the solver
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
errorsForm = {
    'u': getErrorFormula(ud, dm['z'], relative=True),
    **{c: getErrorFormula(eval(f'{c}d'), dm[c], relative=True)
            for c in controls}
}

# Set the domain labels
omg = getDomainLabels(
    {k: getMeasure(v['L²'][0].integrals()[0])
        for k, v in errorsForm.items()},
    boundaryMark
    )

# Stop the nnz calculation thread
thrdNnz.join()

# Get value from list
W.nnz = W.nnz[0]

# Show the program data
showProblemData(
    f'Optimal Heat-based Problem on {basename(meshPath)}',
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

# Assign the initial solution values
U.assign(u0)
ud.assign(u0)

if not showSolution:
    # Create a file to export solutions
    file = File(exportSolutionsPath['U'])
    fileExact = File(exportSolutionsPath['ud'])

    # Export initial solution value
    file.write(U, 0)
    fileExact.write(ud, 0)

else:
    # Create the solutions store
    store = []
    storeExact = []

    # Storage initial solution value
    store.append(U.copy(deepcopy=True))
    storeExact.append(ud.copy(deepcopy=True))

# Create our progress bar
progressBar = ProgressBar(
    'Solving the evolutive system for all time: ',
    total=Tf,
    label='t',
    formatter=f'1.0{getPow10(_dt)+1}f'
    )

# Start the progress bar
progressBar.start()

# Set the initial time instant
t = 0

# Set the matrix and vector to assembling
A = PETScMatrix()
b = PETScVector()

# Init the errors by time storage
errorsByTime = {
    k: {
        eType: []
            for eType  in errorsForm[k].keys()
    } for k in errorsForm.keys()
}

# Init the exact data storage
exactDataTime = {k: [] for k in exactData}

# Init the numerical data storage
numericalDataTime = {k: [] for k in ['u'] + _controls}

# Looping in time
while t < Tf:
    # Set the time instant that system will be solved
    t += _dt

    # Update the time instant in exact functions
    for k, funcExpr in exactExpressions.items():
        # Update time instant
        funcExpr.t = t
        # Update function values
        exactData[k].assign(interpolate(funcExpr, Vc))
        # Store a function copy
        exactDataTime[k].append(
            exactData[k].copy(deepcopy=True)
            )

    # Set the time instant in time functions
    gWExpr.t = t

    # Update the current time functions
    gW.assign(interpolate(gWExpr, Vc))

    if 'f' not in controls:
        fExpr.t = t
        f.assign(interpolate(fExpr, Vc))
    if 'ug' not in controls:
        gExpr.t = t
        g.assign(interpolate(gExpr, Vc))
    if 'h' not in controls:
        hExpr.t = t
        h.assign(interpolate(hExpr, Vc))

    if allAtOnce:
        # Couple the variational formulations
        a, L = system(
            aState + aAdjoint + aOptimal
            - (LState + LAdjoint + LOptimal)
        )

        # Solve the all at once system
        mySolve(a == L, ZLZC, bcs, solver,
                runtime=t==_dt, nnz=W.nnz, dofs=W.dofs,
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
            gamma=gamma, errorForm=errorsForm, copyTo=externalTxt
        )[2]

    # Split the solutions
    if allAtOnce and optimal:
        # Get splitted solutions
        _ZLZC = ZLZC.split(deepcopy=True)

        # Get solutions by problem
        Z, LZ = _ZLZC[:2]  # State and adjoint
        C = dict(zip(controls, _ZLZC[2:])) # Controls

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
            (U, errorsForm['u']),
            *[(C[c], errorsForm[c]) for c in controls],
            labels=['u', *controls],
            relative=type(errorsForm['u']['L²']) is tuple
            )

    elif not optimal:
        # Calcule approach errors
        errors = evaluateErrors(
            (U, errorsForm['u']),
            labels=['u'],
            relative=type(errorsForm['u']['L²']) is tuple
            )

    # # Show the approach error
    # showError(errors, omg, precision=6, copyTo=externalTxt)

    # Store the respective time instant error
    appendErrorsByTime(errorsByTime, errors)

    # Update the previous time solutions
    zn.assign(Z)
    if 'f' in controls:
        fn.assign(C['f'])
    else:
        fn.assign(f)
    if 'ug' in controls:
        ugn.assign(C['ug'])
    if 'h' in controls:
        hn.assign(C['h'])
    else:
        hn.assign(h)

    if not showSolution:
        # Export the current solution
        file.write(U, t)
        fileExact.write(ud, t)

    # else:
    #     # Store the current solution
    #     store.append(U.copy(deepcopy=True))
    #     storeExact.append(ud.copy(deepcopy=True))

    # Store the current numerical solutions
    numericalDataTime['u'].append(U.copy(deepcopy=True))
    [numericalDataTime[c].append(C[c].copy(deepcopy=True))
        for c in controls]

    # Update the progrss list in parallel
    thrd, _ = parallel(
        progressBar.update,
        t,
        suffix=showErrors(
            {k: (np.array(v)**2).sum()**0.5 for k, v in errors['u'].items()},
            mode='horizontal',
            preffix=4*' ' + 'L²(0,T): '
            )
            + showErrors(
            {k: max(v) for k, v in errors['u'].items()},
            mode='horizontal',
            preffix=4*' ' + 'L∞(0,T): '
            )
        )

    # Add to global threads list
    THREADS.append(thrd)

# Erase the progress bar
progressBar.erase()

# # Calcule the exact time norms
# exactTimeNorms = {
#     k: arrNorm(func.compute_vertex_values(), 'l2')
#         for k, func in exactDataTime.items()
# }

# Apply the error time norm
errorsTime = {k: {'L²': {}, 'L∞': {}} for k in errorsByTime}
for k, errors_k in errorsByTime.items():
    for eType, error in errors_k.items():
        # Absolute errors
        errorsTime[k]['L²'][eType] = (
            arrNorm(np.array(error)[:, 0], 'l2')
            #(np.array(error)[:, 0]**2).sum()**0.5,
        )
        errorsTime[k]['L∞'][eType] = (
            arrNorm(np.array(error)[:, 0], 'loo')
            #np.array(error)[:, 0].max(),
        )

        # # Relative errors
        # for normType in ['L²', 'L∞']:
        #     errorsTime[k][normType][eType] += (
        #         errorsTime[k][normType][eType]/exactTimeNorms[k]['L²'],
        #     )

# Show the approach errors
for k, errors_k in errorsTime.items():
    showInfo(
        f"||{k.upper()} - {k}d||_Y(0,T; X(Ω))"
        )
    showInfo(
        *np.ravel(
            [[f'{eTypeT}, {eType} : {e:1.03e}'
                for eType, e in errors.items()]
            for eTypeT, errors in errors_k.items()]
        ), alignment=':', delimiters=False, tab=4,
        breakStart=False
    )

import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12
plt.rcParams['legend.fontsize']=15
plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 3

if showSolution:
    # Show the dynamic plot
    fig = dynamicComparison(
        (numericalDataTime['u'], exactDataTime['ud']),
        iterator=np.arange(0, Tf+_dt, _dt),
        labels=['U', 'ud'],
        linestyles=['-', ''],
        markers=['o', '*'],
        multipleViews=False,
        show=False
        )

    # Init a figures list
    figs = [fig]

    for c in controls:
        # Show the dynamic plot
        fig = dynamicComparison(
            (numericalDataTime[c], exactDataTime[c]),
            iterator=np.arange(0, Tf+_dt, _dt),
            labels=[c.upper(), f'{c}d'],
            linestyles=['-', ''],
            markers=['o', '*'],
            multipleViews=True,
            show=False
            )

        # Append to the figures list
        figs.append(fig)

    adjustFiguresInScreen(*figs)
    show()

# Stop all parallel process
[thrd.join() for thrd in THREADS]

# For nonlinear problems, to use (a - L) == 0