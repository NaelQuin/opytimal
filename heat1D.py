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
from backend.meshes import getInnerNodes, getSubMeshes
from backend.plots import plotMesh, dynamicComparison
from backend.tests import testLoop
from backend.fenics import (setExpression, extractElements, calculeNnz,
                            setSolver, mySolve)
from backend.analytical import AnalyticalFunction
from backend.parallel import parallel
from backend.profiler import ProgressBar
from backend.numeric import getPow10
from backend.arrays import array

# Global value
QUADRATURE_DEG = 5
THREADS = []

# ====================
# Analytical solutions
# ====================
_ud = AnalyticalFunction(
    '{fx}*{ft}'.format(
        fx='x * (4 - x)',
        ft='exp(-t)'
        ),
    toccode=True
    )

# ==========
# Parameters
# ==========
Tf = 1
dt = {
    1: "mesh.hmax()",
    2: 1e-3
}[1]
teta = 0.5

meshLimits = (0, 4)
meshNel = 64

invertNormalOrientation = False

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
    return _ud(*x, t=t)

def h(*x, t=0):
    'Neumann value on Γ_N'
    return _ud.grad(*x, t=t) @ normals['outlet']

def u0(*x):
    return _ud(*x, t=0)

# Generate the mesh
mesh = IntervalMesh(meshNel, *meshLimits)

# Set the dt value as h
if type(dt) is str:
    # Evaluate the dt choice
    dt = eval(dt)

# Set the boudnary data
boundaryData = MeshFunction(
    'size_t', mesh, mesh.geometric_dimension()-1, value=0
    )

# Create boundary indetifyers
inlet = CompiledSubDomain(
    f'near(x[0], {meshLimits[0]}, DOLFIN_EPS) and on_boundary'
    )
outlet = CompiledSubDomain(
    f'near(x[0], {meshLimits[1]}, DOLFIN_EPS) and on_boundary'
    )

# Mark the boundary data
inlet.mark(boundaryData, 1)
outlet.mark(boundaryData, 2)

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
    'outlet': 2
    }

# Get the mesh inner nodes
innerNodes = getInnerNodes(mesh, bmesh)

# Get the inlet and outlet boundary node finders
inletNodeFinder = [
    inlet.inside(x, x in bmesh.coordinates())
        for x in mesh.coordinates()
    ]
outletNodeFinder = [
    outlet.inside(x, x in bmesh.coordinates())
        for x in mesh.coordinates()
    ]

# Get the resepctive node
inletNode = (np.ravel(mesh.coordinates())*inletNodeFinder).sum()
outletNode = (np.ravel(mesh.coordinates())*outletNodeFinder).sum()

# Adjust the vectors to plot
innerNodes = np.hstack((innerNodes, 0*innerNodes))
inletNode = [inletNode, 0]
outletNode = [outletNode, 0]

# Get the normals
normals = {
    'inlet': array([assemble(n[0]*ds(boundaryMark['inlet'])), 0, 0]),
    'outlet': array([assemble(n[0]*ds(boundaryMark['outlet'])), 0, 0])
}

if invertNormalOrientation:
    # Invert each normal orientation
    normals = {k: v*-1 for k, v in normals.items()}

if showMesh:
    # Plot the mesh and bmesh nodes by category
    plotMesh(
        innerNodes.T, inletNode, outletNode,
        labels=['inner', 'inlet', 'outlet']
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
hExpr = setExpression(h, elements[0], name='h', t=True)
u0Expr = setExpression(u0, elements[0], name='u0')

# Set a current input time functions
ud = Function(V, name='ud')
f = Function(V, name='f')
g = Function(V, name='g')
h = Function(V, name='h')

# Set the respectives previous time functions
un = Function(V, name='un')
fn = Function(V, name='fn')
gn = Function(V, name='gn')
hn = Function(V, name='hn')

# Set previous time function as initials respective values
un.assign(interpolate(u0Expr, V))
fn.assign(interpolate(fExpr, V))
gn.assign(interpolate(gExpr, V))
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
    DirichletBC(V, g, boundaryData, boundaryMark['inlet'])
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
    hExpr.t = t

    # Update the current time functions
    ud.assign(interpolate(udExpr, V))
    f.assign(interpolate(fExpr, V))
    g.assign(interpolate(gExpr, V))
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

    # # Update the progress bar
    # progressBar.update(
    #     t,
    #     suffix=showErrors(
    #         {k: (np.array(v)**2).sum()**0.5 for k, v in errors.items()},
    #         mode='horizontal',
    #         preffix=4*' ' + 'L²(0,T): '
    #         )
    #         + showErrors(
    #         {k: max(v) for k, v in errors.items()},
    #         mode='horizontal',
    #         preffix=4*' ' + 'L∞(0,T): '
    #         )
    #     )

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
        linestyles=['--', '-'],
        markers=['o', '*'],
        markersizes=[5, 10],
        markevery=range(0, V.dofs, max(V.dofs//8, 8))
        )

# Stop all parallel process
[thrd.join() for thrd in THREADS]

# For nonlinear problems, to use (a - L) == 0