'''
This script solve the optimal control problem subject
to Heat equation, considering multiple controls
'''

# from modules import *
# from settings import settings

from dolfin import *
import numpy as np
import sympy as sp

from backend.string import showInfo, showErrors, splitPathFile
from backend.meshes import (readXDMFMesh, getInnerNodes, getSubMeshes,
                            getSubMeshesComplement)
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

meshPath = './InputData/Meshes/3D/cylinder1'
boundaryDataPath = './InputData/Meshes/3D/cylinder1_mf'

showMesh = False

exportSolutionsPath = {
    'U': '/output/'.join(splitPathFile(__file__)).replace('.py', '_U.pvd'),
    'ud': '/output/'.join(splitPathFile(__file__)).replace('.py', '_ud.pvd'),
    }

showSolution = True

Th = 'P1'

def f(*x, t=0):
    return _ud.dt(*x, t=t) - _ud.divgrad(*x, t=t)

def ug(*x, t=0):
    'Dirichlet value on Γ_D'
    return _ud(*x, t=t) * int(inInlet(x) and not inWall(x))

def ugW(*x, t=0):
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

# Get the mesh inner nodes
innerNodes = getInnerNodes(mesh, bmesh)

# Get the boundaries submesh
boundarySubMeshes = getSubMeshes(boundaryData, boundaryMark)

# Add the submesh inlet minus wall
boundarySubMeshes['inlet\wall'] = getSubMeshesComplement(boundaryData, 1, 3)

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
Th = elements[0]

# Set the function space
V = FunctionSpace(mesh, Th)

# Get the function space dofs and store it in itself
V.dofs = len(V.dofmap().dofs())

# Get the matrix nonzero amount
thrdNnz, V.nnz = parallel(calculeNnz, V, dx)

# Set the Trial and Test Functions
u = z = TrialFunction(V)
v = TestFunction(V)

# Turn input functions to expression
udExpr = setExpression(_ud, elements[0], name='ud', t=True)
fExpr = setExpression(f, elements[0], name='f', t=True)
ugExpr = setExpression(ug, elements[0], name='ug', t=True)
ugWExpr = setExpression(ugW, elements[0], name='ugW', t=True)
hExpr = setExpression(h, elements[0], name='h', t=True)
u0Expr = setExpression(u0, elements[0], name='u0')

# Set a current input time functions
ud = Function(V, name='ud')
f = Function(V, name='f')
ug = Function(V, name='ug')
ugW = Function(V, name='ugW')
h = Function(V, name='h')

# Set the respectives previous time functions
zn = Function(V, name='zn')
fn = Function(V, name='fn')
ugn = Function(V, name='ugn')
ugWn = Function(V, name='ugWn')
hn = Function(V, name='hn')
u0 = Function(V, name='u0')

# Set previous time function as initials respective values
u0.assign(interpolate(u0Expr, V))
fn.assign(interpolate(fExpr, V))
ugn.assign(interpolate(ugExpr, V))
ugWn.assign(interpolate(ugWExpr, V))
hn.assign(interpolate(hExpr, V))
zn.assign(u0 - ugn - ugWn)

# Turn input scalars to constant
teta = Constant(teta, name='θ')
_dt = dt # Store dt float
dt = Constant(dt, name='Δt')

# Set the variational problem
a = z*v*dx + dt*teta*dot(grad(z), grad(v))*dx
L = - (ug - ugn)*v*dx - (ugW - ugWn)*v*dx + zn*v*dx\
    + dt*(teta*f + (1-teta)*fn)*v*dx \
    + dt*(teta*h + (1-teta)*hn)*v*ds(boundaryMark['outlet'])\
    - dt*teta*dot(grad(ug + ugW), grad(v))*dx\
    - dt*(1-teta)*dot(grad(zn + ugn + ugWn), grad(v))*dx

# Set the boundary conditions
bcs = [
    DirichletBC(V, 0, boundaryData, boundaryMark['inlet']),
    DirichletBC(V, 0, boundaryData, boundaryMark['wall'])
    ]

# Set the system solution
Z = Function(V, name='Z')
U = Function(V, name='U')

# Assign the inital solution values
Z.assign(zn)
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
    ugExpr.t = t
    ugWExpr.t = t
    hExpr.t = t

    # Update the current time functions
    ud.assign(interpolate(udExpr, V))
    f.assign(interpolate(fExpr, V))
    ug.assign(interpolate(ugExpr, V))
    ugW.assign(interpolate(ugWExpr, V))
    h.assign(interpolate(hExpr, V))

    # Solve the system
    A, b = mySolve(a == L, Z, bcs, solver, A=A, b=b)

    # Get the U value
    U.assign(Z + ug + ugW)

    # Calcule the approach error
    [errors[eType].append(
        assemble(action(e, U))**0.5
        ) for eType, e in error.items()]

    # Update the previous time solutions
    zn.assign(Z)
    fn.assign(f)
    ugn.assign(ug)
    ugWn.assign(ugW)
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

input(errorsTime)

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