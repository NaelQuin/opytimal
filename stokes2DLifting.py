'''
Stokes problem on 2D domain
'''

# from modules import *
# from settings import settings

from dolfin import *

from backend.string import showInfo, splitPathFile
from backend.meshes import (readXDMFMesh, getInnerNodes, getSubMeshes,
                            getCoordinates)
from backend.plots import (plotMesh, plotComparison, adjustFiguresInScreen,
                           show, simulation)
from backend.tests import testLoop
from backend.fenics import (setExpression, extractElements, calculeNnz, mySolve,
                            setSolver, getErrorFormula, evaluateErrors,
                            showError)
from backend.analytical import (AnalyticalFunction, AnalyticalVectorFunction)
from backend.parallel import parallel
from backend.arrays import array, identity, zeros, getComplement

# Global value
QUADRATURE_DEG = 5

# ====================
# Analytical solutions
# ====================
#_ud, _pd = womersly(1, 20, 3*pi)
_ud = AnalyticalVectorFunction(
    ['y*(2 - y)', '0'],  # Poiselle
    #[_ud(*sp.symbols('x, y'), t=1), '0'],
    toccode=True
    )

_pd = AnalyticalFunction(
    '4 - x', # Poiselle
    #_pd(*sp.symbols('x, y'), t=1),
    toccode=True
    )

_ud = _pd = None

# ==========
# Parameters
# ==========
meshPath = './InputData/Meshes/2D/rectangle4'
boundaryDataPath = './InputData/Meshes/2D/rectangle4_mf'

showMesh = False

exportSolutionsPath = {
    'U': ('/output/'.join(splitPathFile(__file__))).replace('.py', '_U.pvd'),
    'P': ('/output/'.join(splitPathFile(__file__))).replace('.py', '_P.pvd')
    }

showSolution = True

Th = 'P2[2], P1'

def D(u, choice=1):
    if choice == 1:  # [Poiselle]
        output = grad(u)
    elif choice == 2:  # [Turek]
        output = 2*sym(grad(u))
    return output

def f(*x):
    #return -_ud.divgrad(*x) + _pd.grad(*x, dim=2)
    return zeros(gdim)

def ug(*x):
    'Dirichlet value on Γ_D'
    #return _ud(*x)
    #return array([x[1]*(2 - x[1]), 0]) #* inInlet(x)
    return 50 * inletNormal[:gdim] * inInletOnly(x)

def ugW(*x):
    'Dirichlet value on Γ_W'
    #return _ud(*x) * inWall(x)
    return zeros(gdim) * inWall(x)

def h(*x):
    'Neumann value on Γ_N'
    #return (_ud.grad(*x) - _pd(*x)*I) @ outletNormal
    return 0 * outletNormal[:gdim]

def inWall(x):
    return x in boundaryCoordinates['wall']

def inInlet(x):
    return x in boundaryCoordinates['inlet']

def inInletOnly(x):
    return x in boundaryCoordinates['inlet']\
        and x not in boundaryCoordinates['wall']

# Load the mesh and boundary data
mesh, boundaryData, _ = readXDMFMesh(meshPath, boundaryDataPath)

# Get the geometric dimension
gdim = mesh.geometric_dimension()

# Set the identity matrix
I = identity(3)[:gdim]

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

# Get the boundaries coordiantes
boundaryCoordinates = getCoordinates(**boundarySubMeshes)

# Get the boundary coordinates complements
boundaryCoordinates['inlet\wall'] = getComplement(
    boundaryCoordinates['inlet'],
    boundaryCoordinates['wall']
    )
boundaryCoordinates['outlet\wall'] = getComplement(
    boundaryCoordinates['outlet'],
    boundaryCoordinates['wall']
    )

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
        labels=['inner', *boundarySubMeshes.keys()]
        )

# ----------------------------
# Set the Finite Element Space
# ----------------------------
# Extract elements
elements = extractElements(Th, mesh.ufl_cell())

# Set the Finite Element Space
Th = MixedElement(elements)

# Set the function space
W = FunctionSpace(mesh, Th)

# Get the subspaces
V, Q = W.split()

# Get the sollapsed subspaces
Vc = V.collapse()
Qc = Q.collapse()

# Get the function space dofs and store it in itself
W.dofs = len(W.dofmap().dofs())
V.dofs = len(Vc.dofmap().dofs())
Q.dofs = len(Qc.dofmap().dofs())

# Get the matrix nonzero amount
thrdNnz_1, W.nnz = parallel(calculeNnz, W, dx)
thrdNnz_2, V.nnz = parallel(calculeNnz, Vc, dx)
thrdNnz_3, Q.nnz = parallel(calculeNnz, Qc, dx)

# Set the Trial and Test Functions
z, p = TrialFunctions(W)
v, q = TestFunctions(W)

if _ud is not None:
    # Turn input functions to expression
    udExpr = setExpression(_ud, elements[0], name='ud')
    pdExpr = setExpression(_pd, elements[1], name='pd')

    # Set the respective input functions
    ud = interpolate(udExpr, Vc); ud.rename('ud', 'ud')
    pd = interpolate(pdExpr, Qc); pd.rename('pd', 'pd')

# Turn input functions to expression
fExpr = setExpression(f, elements[0], name='f')
ugExpr = setExpression(ug, elements[0], name='ug')
ugWExpr = setExpression(ugW, elements[0], name='ugW')
hExpr = setExpression(h, elements[0], name='h')

# Set the respective input functions
f = interpolate(fExpr, Vc); f.rename('f', 'f')
ug = interpolate(ugExpr, Vc); ug.rename('ug', 'ug')
ugW = interpolate(ugWExpr, Vc); ugW.rename('ugW', 'ugW')
h = interpolate(hExpr, Vc); h.rename('h', 'h')

# Set the variational problem
a = inner(D(z), D(v))*dx\
    - p*div(v)*dx\
    + div(z)*q*dx

L = dot(f, v)*dx\
    + dot(h, v)*ds(boundaryMark['outlet'])\
    - inner(D(ug + ugW), D(v))*dx\
    - div(ug + ugW)*q*dx

# Set the boundary conditions
bcs = [
    DirichletBC(V, zeros(gdim), boundaryData, boundaryMark['inlet']),
    DirichletBC(V, zeros(gdim), boundaryData, boundaryMark['wall'])
    ]

# Set the system solution
ZP = Function(W, name='ZP')
U = Function(Vc, name='U')
P = Function(Qc, name='P')

# Set my solver
solver = setSolver('gmres', 'ml_amg')

if type(solver) is PETScKrylovSolver:
    # # Set the stabilizator scalar
    # beta = 0

    # # Add the stabilizator term
    # a += Constant(beta)*dot(grad(p), grad(q))*dx

    # Set a preconditioner to system
    AC = assemble(
        inner(grad(z), grad(v))*dx + p*q*dx
        )

    # Apply the boundary conditions to preconditioner
    [bc.apply(AC) for bc in bcs]

else:
    raise ValueError('The system solver must be a Krylov Interactive Method')

# Solve the system
A, b = mySolve(a == L, ZP, bcs, solver, P=AC)

# Stop the nnz calculation thread
[thrd.join() for thrd in [thrdNnz_1, thrdNnz_2, thrdNnz_3]]

# Get value from list
W.nnz = W.nnz[0]
V.nnz = V.nnz[0]
Q.nnz = Q.nnz[0]

if _ud is not None:
    # Set the error formulas
    error_u = getErrorFormula(ud, dx, relative=True)
    error_p = getErrorFormula(pd, dx, relative=True)

# Split the solutions
Z, _P = ZP.split(deepcopy=True)

# Fill the pressure vector
P.assign(_P)

# Recover the original U solution
U.vector().set_local(
    Z.vector().get_local()
    + ug.vector().get_local()
    + ugW.vector().get_local()
    )

# # Apply the liftings trace to ZP solution
# bcsRecover = [
#     DirichletBC(Vc, ug, boundaryData, boundaryMark['inlet']),
#     DirichletBC(Vc, ugW, boundaryData, boundaryMark['wall'])
#     ]
# [bc.apply(U.vector()) for bc in bcsRecover]

if _ud is not None:
    # Calcule the approach errors
    errors = evaluateErrors(
        (U, error_u),
        (P, error_p),
        labels=['u', 'p'],
        relative=type(error_u['L²']) is tuple
        )

    # Show the approach error
    showError(errors)

if showSolution and _ud is not None:
    # Set the common args
    commonArgs = {
        'splitSols': True,
        'show': False,
        'interactive': False
        }

    # Plot the velocity comparison
    fig_U = plotComparison(
        ud, U, **commonArgs
        )

    # Plot the pressure comparison
    fig_P = plotComparison(
        pd, P, **commonArgs
        )

elif showSolution:
    # Plot the velocity
    fig_U = simulation(
        U, show=False
        )

    # Plot the pressure
    fig_P = simulation(
        P, show=False
        )

if showSolution:
    # Adjust figures in screen
    adjustFiguresInScreen(fig_U, fig_P)

    # Show the plots
    show()

# Create a file to export solutions
file_u = File(exportSolutionsPath['U'])
file_p = File(exportSolutionsPath['P'])

# Export the solution
file_u.write(U)
file_p.write(P)
