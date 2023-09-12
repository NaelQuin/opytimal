'''
Poisson problem on 1D domain
'''

# from modules import *
# from settings import settings

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

from backend.string import showInfo
from backend.meshes import readXdmfMesh, getInnerNodes, getSubMeshes
from backend.plots import plotMesh, plotComparison
from backend.tests import testLoop
from backend.fenics import (setExpression, extractElements, calculeNnz,
                            setSolver, mySolve)
from backend.analytical import AnalyticalFunction
from backend.parallel import parallel

# Global value
QUADRATURE_DEG = 5

# ====================
# Analytical solutions
# ====================
_ud = AnalyticalFunction(
    'x*(4 - x)/4 + 1/(x+1)',
    toccode=True
    )

# ==========
# Parameters
# ==========
meshLimits = (0, 4)
meshNel = 64

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
    return _ud(*x) * inInlet(x)

def h(*x):
    'Neumann value on Γ_N'
    return (_ud.grad(*x) @ outletNormal) * inOutlet(x)

def inInlet(x):
    return inlet.inside(x, x in bmesh.coordinates())

def inOutlet(x):
    return outlet.inside(x, x in bmesh.coordinates())

# Generate the mesh
mesh = IntervalMesh(meshNel, *meshLimits)

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

# Get the inlet and outlet normals
inletNormal = [assemble(n[0]*ds(boundaryMark['inlet'])), 0, 0]
outletNormal = [assemble(n[0]*ds(boundaryMark['outlet'])), 0, 0]

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
hExpr = setExpression(h, elements[0], name='h')

# Set the current input functions
ud = interpolate(udExpr, V); ud.rename('ud', 'ud')
f = interpolate(fExpr, V); f.rename('f', 'f')
ug = interpolate(ugExpr, V); f.rename('ug', 'ug')
h = interpolate(hExpr, V); f.rename('h', 'h')

# Set the variational problem
a = dot(grad(z), grad(v))*dx 
L = f*v*dx \
    + h*v*ds(boundaryMark['outlet']) \
    - dot(grad(ug), grad(v))*dx

# Set the boundary conditions
bcs = [
    DirichletBC(V, 0, boundaryData, boundaryMark['inlet'])
    ]

# Set the system solution
Z = Function(V, name='Z')
U = Function(V, name='U')

# Set my solver
#solver = setSolver('tfqmr', 'ml_amg')
solver = setSolver('mumps')

# Solve the system
A, b = mySolve(a == L, Z, bcs, solver)

# Get the U value
U.assign(Z + ug)

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
    plotComparison(ud, U, markevery=range(0, V.dofs, max(V.dofs//8, 8)))

# Create a file to export solutions
file = File(exportSolutionsPath)

# Export the solution
file.write(U)
