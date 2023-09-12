'''
Poisson problem on 2D domain
'''

# from modules import *
# from settings import settings

from dolfin import *

from backend.string import showInfo, replaceAll
from backend.meshes import (readXdmfMesh, getInnerNodes, getSubMeshes,
                            getCoordinates)
from backend.plots import plotMesh, plotComparison
from backend.tests import testLoop
from backend.fenics import (setExpression, extractElements, calculeNnz,
                            setSolver, mySolve)
from backend.analytical import AnalyticalFunction
from backend.parallel import parallel
from backend.arrays import getComplement

# Global value
QUADRATURE_DEG = 5

# ====================
# Analytical solutions
# ====================
_ud = AnalyticalFunction(
    'y*(2 - y) - x*(4 - x)',
    toccode=True
    )

# ==========
# Parameters
# ==========
meshPath = './InputData/Meshes/2D/rectangle32'
boundaryDataPath = './InputData/Meshes/2D/rectangle32_mf'

showMesh = False

exportSolutionsPath = (
    '/output/'.join(splitPathFile(__file__))
    ).replace('.py', '.pvd')

showSolution = True

Th = 'P1'

def f(*x):
    return -_ud.divgrad(*x)

def ug(*x):
    'Dirichlet value on Γ_D'
    return _ud(*x) * inInletOnly(x)

def ugW(*x):
    'Dirichlet value on Γ_W'
    return _ud(*x) * inWall(x)

def h(*x):
    'Neumann value on Γ_N'
    return _ud.grad(*x) @ outletNormal

def inInletOnly(x):
    return x in boundaryCoordinates['inlet\wall']

def inWall(x):
    return x in boundaryCoordinates['wall']

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

# Get the boundaries coordiantes
boundaryCoordinates = getCoordinates(**boundarySubMeshes)

# Get the boundary coordinates complements
boundaryCoordinates['inlet\wall'] = getComplement(
    boundaryCoordinates['inlet'],
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
udExpr = setExpression(_ud, elements[0], name='ud')
fExpr = setExpression(f, elements[0], name='f')
ugExpr = setExpression(ug, elements[0], name='ug')
ugWExpr = setExpression(ugW, elements[0], name='ugW')
hExpr = setExpression(h, elements[0], name='h')

# Set a current time functions
ud = interpolate(udExpr, V); ud.rename('ud', 'ud')
f = interpolate(fExpr, V); f.rename('f', 'f')
ug = interpolate(ugExpr, V); f.rename('ug', 'ug')
ugW = interpolate(ugWExpr, V); f.rename('ugW', 'ugW')
h = interpolate(hExpr, V); f.rename('h', 'h')

# Set the variational problem
a = dot(grad(z), grad(v))*dx
L = f*v*dx\
    + h*v*ds(boundaryMark['outlet'])\
    - dot(grad(ug + ugW), grad(v))*dx

# Set the boundary conditions
bcs = [
    DirichletBC(V, 0, boundaryData, boundaryMark['inlet']),
    DirichletBC(V, 0, boundaryData, boundaryMark['wall'])
    ]

# Set the system solution
Z = Function(V, name='Z')
U = Function(V, name='U')

# Set my solver
#solver = setSolver('tfqmr', 'ml_amg')
solver = setSolver('mumps')

# Solve the system
A, b = mySolve(a == L, Z, bcs, solver)

# Recover the U value
U.assign(Z + ug + ugW)

# Stop the nnz calculation thread
thrdNnz.join()

# Get value from list
V.nnz = V.nnz[0]

# Set the error formulas
error = {'L²': (u - ud)**2*dx}
error['H¹'] = error['L²'] + dot(grad(u - ud), grad(u - ud))*dx
if elements[0].degree() > 1:
    error['H²'] = error['H¹'] + div(grad(u - ud))**2*dx

# Calcule the approach error
errors = {
    eType: assemble(action(e, U))**0.5
        for eType, e in error.items()
    }

# Show the approach error
showInfo(
    "||U - ud||_X(Ω)"
    )
showInfo(
    *[f'{eType} : {e:1.03e}'
        for eType, e in errors.items()],
    alignment=':', delimiters=False, tab=4,
    breakStart=False
    )

if showSolution:
    # Plot the solutions comparison
    plotComparison(
        ud, U, splitSols=False, scatters=True, projection3D=True,
        markerSizes=[20, 1], colors=[None, 'k']
        )

# Create a file to export solutions
file = File(exportSolutionsPath)

# Export the solution
file.write(U)
