'''
Stokes problem on 2D domain
'''

# from modules import *
# from settings import settings

from dolfin import *

from backend.string import showInfo, splitPathFile
from backend.meshes import (readXdmfMesh, getInnerNodes, getSubMeshes,
                            getCoordinates)
from backend.plots import (plotMesh, plotComparison, adjustFiguresInScreen,
                           show)
from backend.tests import testLoop
from backend.fenics import (setExpression, extractElements, calculeNnz, mySolve,
                            setSolver, getErrorFormula)
from backend.analytical import (AnalyticalFunction, AnalyticalVectorFunction)
from backend.parallel import parallel
from backend.arrays import identity, zeros
from backend.types import Argument

# Global value
QUADRATURE_DEG = 5

# ====================
# Analytical solutions
# ====================
_ud = AnalyticalVectorFunction(
    ['{C}*(1 - ({r})/{R}**2)'.format(
        C=1/2,
        r='y**2 + z**2',
        R=1
        ), '0', '0'],
    toccode=True
    )

_pd = AnalyticalFunction(
    '(4 - x)',
    toccode=True
    )

# ==========
# Parameters
# ==========
meshPath = './InputData/Meshes/3D/cylinder4'
boundaryDataPath = './InputData/Meshes/3D/cylinder4_mf'

showMesh = False

exportSolutionsPath = {
    'U': ('/output/'.join(splitPathFile(__file__))).replace('.py', '_U.pvd'),
    'P': ('/output/'.join(splitPathFile(__file__))).replace('.py', '_P.pvd')
    }

showSolution = True

Th = 'P2[:], P1'

def D(u: Argument, choice: int = 1) -> Form:
    output = {
        1: grad(u), # [Poiselle]
        2: 2*sym(grad(u)) # [Turek]
        }[choice]
    return output

def f(*x: float) -> list[float]:
    return -_ud.divgrad(*x) + _pd.grad(*x)

def ug(*x: float) -> list[float]:
    'Dirichlet value on Γ_D'
    return _ud(*x)

def ugW(*x: float) -> list[float]:
    'Dirichlet value on Γ_W'
    return _ud(*x) * inWall(x)

def h(*x: float) -> list[float]:
    'Neumann value on Γ_N'
    return (_ud.grad(*x) - _pd(*x)*I) @ outletNormal

def inWall(x: tuple[float]) -> bool:
    return x in boundaryCoordinates['wall']

# Load the mesh and boundary data
mesh, boundaryData, _ = readXdmfMesh(meshPath, boundaryDataPath)

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

# Turn input functions to expression
udExpr = setExpression(_ud, elements[0], name='ud')
pdExpr = setExpression(_pd, elements[1], name='pd')
fExpr = setExpression(f, elements[0], name='f')
ugExpr = setExpression(ug, elements[0], name='ug')
ugWExpr = setExpression(ugW, elements[0], name='ugW')
hExpr = setExpression(h, elements[0], name='h')

# Set the current input functions
ud = interpolate(udExpr, Vc); ud.rename('ud', 'ud')
pd = interpolate(pdExpr, Qc); pd.rename('pd', 'pd')
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
solver = setSolver('tfqmr', 'ml_amg')

if type(solver) is PETScKrylovSolver:
    # Set the stabilizator scalar
    beta = 0

    # Add the stabilizator term
    a += Constant(beta)*dot(grad(p), grad(q))*dx

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

# Get the splitted trial functions for the approach 
# errors calculus
u = TrialFunction(Vc)
p = TrialFunction(Qc)

# Set the error formulas
error_u = getErrorFormula(u, ud)
error_p = getErrorFormula(p, pd)

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

# Calcule the approach error
errors_up = {
    S.name(): {eType: assemble(action(e, S))**0.5
            for eType, e in error.items()}
        for S, error in zip([U, P], [error_u, error_p])
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

if showSolution:
    # Set the common args
    commonArgs = {
        'splitSols': True,
        'projection3D': True,
        'show': False,
        'interactive': False,
        'personalPlot': True
        }

    # Plot the velocity comparison
    fig_U = plotComparison(ud, U, **commonArgs, strides=5)

    # Plot the pressure comparison
    fig_P = plotComparison(pd, P, **commonArgs)

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
