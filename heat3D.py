'''
This script solve the optimal control problem subject
to Heat equation, considering multiple controls
'''

# from modules import *
# from settings import settings

from dolfin import *
import numpy as np
import sympy as sp

from backend.string import showInfo, showErrors
from backend.meshes import readXdmfMesh, getInnerNodes, getSubMeshes
from backend.plots import plotMesh, dynamicComparison
from backend.tests import testLoop
from backend.fenics import (setExpression, extractElements, calculeNnz,
                            setSolver, mySolve)
from backend.analytical import AnalyticalFunction
from backend.parallel import parallel
from backend.profiler import ProgressBar
from backend.numeric import getPow10

# Global value
QUADRATURE_DEG = 5
THREADS = []

# ====================
# Analytical solutions
# ====================
_ud = AnalyticalFunction(
    '{fx}*{ft}'.format(
        fx='1 - ({r})/{R}**2 + x**2'.format(
            r='(y - 1)**2 + (z - 1)**2',
            R=1
            ),
        ft=' exp(-t)'
        ),
    toccode=True
    )

# ==========
# Parameters
# ==========
Tf = 1
dt = 2**-1
teta = 0.5

meshPath = './InputData/Meshes/3D/cylinder16'
boundaryDataPath = './InputData/Meshes/3D/cylinder16_mf'

showMesh = False

exportSolutionsPath = {
    'U': __file__.replace('.py', '_U.pvd'),
    'ud': __file__.replace('.py', '_ud.pvd'),
    }

showSolution = True

Th = 'P1'

def f(*x, t=0):
    return _ud.dt(*x, t=t) - _ud.divgrad(*x, t=t)

def g(*x, t=0):
    'Dirichlet value on Γ_D'
    return _ud(*x, t=t) * int(inInlet(x))

def gW(*x, t=0):
    'Dirichlet value on Γ_W'
    return _ud(*x, t=t) * int(inWall(x))

def h(*x, t=0):
    'Neumann value on Γ_N'
    return (_ud.grad(*x, t=t) @ outletNormal) * int(inOutlet(x))

def u0(*x):
    return _ud(*x, t=0)

def inInlet(x):
    return x in boundarySubMeshes['inlet'].coordinates()

def inOutlet(x):
    return x in boundarySubMeshes['outlet'].coordinates()

def inWall(x):
    return x in boundarySubMeshes['wall'].coordinates()

# Load the mesh and boundary data
mesh, boundaryData, _ = readXdmfMesh(meshPath, boundaryDataPath)

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

# Get the mesh inner nodes
innerNodes = getInnerNodes(mesh, bmesh)

# Get the boundaries submesh
boundarySubMeshes = getSubMeshes(boundaryData, boundaryMark)

# Get the inlet and outlet normals
inletNormal = next(
    cells(boundarySubMeshes['inlet'])
    ).cell_normal().array()
outletNormal = next(
    cells(boundarySubMeshes['outlet'])
    ).cell_normal().array()

if showMesh:
    # Plot the mesh and bmesh nodes by category
    plotMesh(
        innerNodes.T,
        *[subMesh.coordinates().T
            for subMesh in boundarySubMeshes.values()],
        labels=['inner', *boundarySubMeshes.keys()],
        projection3D=True
        )

# ----------------------------
# Set the Finite Element Space
# ----------------------------
# Extract elements
elements = extractElements(Th, mesh.ufl_cell())

# Set the Finite Element Space
Th = MixedElement(elements)\
    if len(elements) > 1\
    else elements[0]

# Set the function space
V = FunctionSpace(mesh, Th)

# Get the function space dofs and store it in itself
V.dofs = len(V.dofmap().dofs())

# Get the matrix nonzero amount
thrdNnz, V.nnz = parallel(calculeNnz, V, dx)

# Set the Trial and Test Functions
u = TrialFunction(V)
v = TestFunction(V)

# Turn input functions to expression
udExpr = setExpression(_ud, elements[0], name='ud', t=True)
fExpr = setExpression(f, elements[0], name='f', t=True)
gExpr = setExpression(g, elements[0], name='g', t=True)
gWExpr = setExpression(gW, elements[0], name='gW', t=True)
hExpr = setExpression(h, elements[0], name='h', t=True)
u0Expr = setExpression(u0, elements[0], name='u0')

# Set a current input time functions
ud = Function(V, name='ud')
f = Function(V, name='f')
g = Function(V, name='g')
gW = Function(V, name='gW')
h = Function(V, name='h')

# Set the respectives previous time functions
un = Function(V, name='un')
fn = Function(V, name='fn')
gn = Function(V, name='gn')
gWn = Function(V, name='gWn')
hn = Function(V, name='hn')

# Set previous time function as initials respective values
un.assign(interpolate(u0Expr, V))
fn.assign(interpolate(fExpr, V))
gn.assign(interpolate(gExpr, V))
gWn.assign(interpolate(gWExpr, V))
hn.assign(interpolate(hExpr, V))

# Turn input scalars to constant
teta = Constant(teta, name='θ')
_dt = dt # Store dt float
dt = Constant(dt, name='Δt')

# Set the variational problem
a = u*v*dx + dt*teta*dot(grad(u), grad(v))*dx
L = un*v*dx\
    + dt*(teta*f + (1-teta)*fn)*v*dx \
    + dt*(teta*h + (1-teta)*hn)*v*ds(boundaryMark['outlet'])\
    - dt*(1-teta)*dot(grad(un), grad(v))*dx

# Set the boundary conditions
bcs = [
    DirichletBC(V, g, boundaryData, boundaryMark['inlet']),
    DirichletBC(V, gW, boundaryData, boundaryMark['wall'])
    ]

# Set the system solution
U = Function(V, name='U')

# Assign the inital solution values
U.assign(un)
ud.assign(un)

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

# Set the error formulas
error = {'L²': (u - ud)**2*dx}
error['H¹'] = error['L²'] + dot(grad(u - ud), grad(u - ud))*dx
if elements[0].degree() > 1:
    error['H²'] = error['H¹'] + div(grad(u - ud))**2*dx

# Init a errors list for all time instant
errors = {eType: [] for eType in error}

# Set my solver
#solver = setSolver('tfqmr', 'ml_amg')
solver = setSolver('mumps')

# Set the matrix and vector to assembling
A = PETScMatrix()
b = PETScVector()

# Looping in time
while t < Tf:
    # Set the time instant that system will be solved
    t += _dt

    # Set the time instant in time functions
    udExpr.t = t
    fExpr.t = t
    gExpr.t = t
    gWExpr.t = t
    hExpr.t = t

    # Update the current time functions
    ud.assign(interpolate(udExpr, V))
    f.assign(interpolate(fExpr, V))
    g.assign(interpolate(gExpr, V))
    gW.assign(interpolate(gWExpr, V))
    h.assign(interpolate(hExpr, V))

    # Solve the system
    A, b = mySolve(a == L, U, bcs, solver, A=A, b=b)

    # Calcule the approach error
    [errors[eType].append(
        assemble(action(e, U))**0.5
        ) for eType, e in error.items()]

    # Update the previous time solutions
    un.assign(U)
    fn.assign(f)
    gn.assign(g)
    gWn.assign(gW)
    hn.assign(h)

    if not showSolution:
        # Export the current solution
        file.write(U, t)
        fileExact.write(ud, t)

    else:
        # Store the current solution
        store.append(U.copy(deepcopy=True))
        storeExact.append(ud.copy(deepcopy=True))

    # Update the progrss list in parallel
    thrd, _ = parallel(
        progressBar.update,
        t,
        suffix=showErrors(
            {k: (np.array(v)**2).sum()**0.5 for k, v in errors.items()},
            mode='horizontal',
            preffix=4*' ' + 'L²(0,T): '
            )
            + showErrors(
            {k: max(v) for k, v in errors.items()},
            mode='horizontal',
            preffix=4*' ' + 'L∞(0,T): '
            )
        )

    # Add to global threads list
    THREADS.append(thrd)

# Erase the progress bar
progressBar.erase()

# Stop the nnz calculation thread
thrdNnz.join()

# Get value from list
V.nnz = V.nnz[0]

# Apply the error time norm
errorsTime = {
    'L²': {
        eType: (np.array(error)**2).sum()**0.5
            for eType, error in errors.items()
        },
    'L∞': {
        eType: max(error)
            for eType, error in errors.items()
        }
    }

# Show the approach errors
showInfo(
    "||U - ud||_Y(0,T; X(Ω))"
    )
showInfo(
    *np.ravel([[f'{eTypeT}, {eType} : {e:1.03e}'
            for eType, e in errors.items()]
        for eTypeT, errors in errorsTime.items()]),
    alignment=':', delimiters=False, tab=4,
    breakStart=False
    )

if showSolution:
    # Show the dynamic plot
    dynamicComparison(
        (store, storeExact),
        iterator=np.arange(0, Tf+_dt, _dt),
        labels=['U', 'ud'],
        linestyles=['-', ''],
        markers=['*', '*'],
        multipleViews=True,
        splitSols=True
        )

# Stop all parallel process
[thrd.join() for thrd in THREADS]

# For nonlinear problems, to use (a - L) == 0