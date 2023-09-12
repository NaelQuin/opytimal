'''
Stokes problem on 2D domain
'''

# from modules import *
# from settings import settings

from dolfin import *

from backend.settings import QUADRATURE_DEG
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

# ====================
# Analytical solutions
# ====================
_ud = AnalyticalVectorFunction(
    ['y*(2 - y)', '0'],  # Poiselle
    toccode=True
    )

_pd = AnalyticalFunction(
    '4 - x',
    toccode=True
    )

_ud = _pd = None

# ==========
# Parameters
# ==========
meshPath = './InputData/Meshes/2D/rectangle7'
boundaryDataPath = './InputData/Meshes/2D/rectangle7_mf'

invertNormalOrientation = True

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

def g(*x):
    'Dirichlet value on Γ_D'
    #return _ud(*x)
    return x[1]*(2 - x[1]) * -inletNormal[:gdim]

def gW(*x):
    'Dirichlet value on Γ_W'
    #return _ud(*x)
    return zeros(gdim)

def h(*x):
    'Neumann value on Γ_N'
    #return (_ud.grad(*x) - _pd(*x)*I) @ outletNormal
    return (0 * outletNormal)[:gdim]

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

# Get the inlet and outlet normals
inletNormal = next(
    cells(boundarySubMeshes['inlet'])
    ).cell_normal().array()
outletNormal = next(
    cells(boundarySubMeshes['outlet'])
    ).cell_normal().array()

if invertNormalOrientation:
    inletNormal *= -1
    outletNormal *= -1

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
thrdNnz, W.nnz = parallel(calculeNnz, W, dx)

# Set the Trial and Test Functions
u, p = TrialFunctions(W)
v, q = TestFunctions(W)

# Turn input functions to expression
if _ud is not None:
    udExpr = setExpression(_ud, elements[0], name='ud')
    pdExpr = setExpression(_pd, elements[1], name='pd')
fExpr = setExpression(f, elements[0], name='f')
gExpr = setExpression(g, elements[0], name='g')
gWExpr = setExpression(gW, elements[0], name='gW')
hExpr = setExpression(h, elements[0], name='h')

# Set the respective functions
if _ud is not None:
    ud = interpolate(udExpr, Vc); ud.rename('ud', 'ud')
    pd = interpolate(pdExpr, Qc); pd.rename('pd', 'pd')
f = interpolate(fExpr, Vc); f.rename('f', 'f')
g = interpolate(gExpr, Vc); g.rename('g', 'g')
gW = interpolate(gWExpr, Vc); gW.rename('gW', 'gW')
h = interpolate(hExpr, Vc); h.rename('h', 'h')

# Set the variational problem
a = (inner(D(u), D(v)) - p*div(v) + div(u)*q)*dx
L = dot(f, v)*dx\
    + dot(h, v)*ds(boundaryMark['outlet'])

# Set the boundary conditions
bcs = [
    DirichletBC(V, g, boundaryData, boundaryMark['inlet']),
    DirichletBC(V, gW, boundaryData, boundaryMark['wall']),
    #DirichletBC(Q, 0, boundaryData, boundaryMark['outlet'])
    ]

# Set the system solution
UP = Function(W, name='UP')

# Set my solver
solver = setSolver('tfqmr', 'hypre_amg')

if type(solver) is PETScKrylovSolver:
    # Set the stabilizator scalar
    beta = Constant(mesh.hmax()**2, name='β')

    # Add the stabilizator term
    a += beta*dot(grad(p), grad(q))*dx

else:
    raise ValueError('The system solver must be a Krylov Interactive Method')

# Stop the nnz calculation thread
thrdNnz.join()

# Get value from list
W.nnz = W.nnz[0]

# Solve the system
A, b = mySolve(a == L, UP, bcs, solver, runtime=True, nnz=W.nnz, dofs=W.dofs)

if _ud is not None:
    # Set the error formulas
    error_u = {'L²': (u - ud)**2*dx}
    error_u['H¹'] = error_u['L²'] + inner(grad(u - ud), grad(u - ud))*dx
    if elements[0].degree() > 1:
        error_u['H²'] = error_u['H¹'] + div(grad(u - ud))**2*dx

    error_p = {'L²': (p - pd)**2*dx}
    error_p['H¹'] = error_p['L²'] + inner(grad(p - pd), grad(p - pd))*dx
    if elements[1].degree() > 1:
        error_p['H²'] = error_p['H¹'] + div(grad(p - pd))**2*dx

    # Calcule the approach error
    errors_up = {
        name: {eType: assemble(action(e, UP))**0.5
                for eType, e in error.items()}
            for name, error in zip(['U', 'P'], [error_u, error_p])
        }

    # Looping in errors expression
    for eName, errors in errors_up.items():
        # Show the approach error
        showInfo(
            f"||{eName} - {eName.lower()}d||_X(Ω)"
            )
        showInfo(
            *[f'{eType} : {e:1.03e}'
                for eType, e in errors.items()],
            alignment=':', delimiters=False, tab=4,
            breakStart=False
            )

# Split the solutions
U, P = UP.split()

# Rename the solution vectors
U.rename('U', 'U')
P.rename('P', 'P')

if showSolution and _ud is not None:
    # Set the common args
    commonArgs = {
        'splitSols': True,
        'show': False,
        'interactive': False,
        }

    # Plot the velocity comparison
    fig_U = plotComparison(
        ud, U, **commonArgs
        )

    # Plot the pressure comparison
    fig_P = plotComparison(
        pd, P, **commonArgs, personalPlot=False,
        projection3D=False
        )

    # Adjust figures in screen
    adjustFiguresInScreen(fig_U, fig_P)

    # Show the plots
    show()

elif showSolution:
    # Set the common args
    commonArgs = {
        'show': False
        }

    # Plot the velocity comparison
    fig_U = simulation(
        U, **commonArgs
        )

    # Plot the pressure comparison
    fig_P = simulation(
        P, **commonArgs, personalPlot=False,
        projection3D=False
        )

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
