'''
Optimal Control Stokes-based Problem on 3D domain
'''

import os
from dolfin import *
from opytimal import *
from opytimal.settings import QUADRATURE_DEG

# ================
# Input Data Begin
# ================
choice = 1
meshName ={
    1: "cylinder3",
    2: "cylinder4",
    3: "cylinder5",
}[choice]
meshPath = f'./meshes/3D/{meshName}'
boundaryDataPath = f'./meshes/3D/{meshName}_mf'

boundaryMark = {
    'inlet': 1,
    'outlet': 2,
    'wall': 3
}

normals = {
    'inlet': [-1, 0, 0],
    'outlet': [1, 0, 0]
}
# To get normals from mesh, uncomment the code below
# normals = None

invertNormalOrientation = False # Multiply all normals to -1

showMesh = False # Turn to True to check the normal vectors
showSolution = False
showSolutionType = {
    1: 'tex', # generate a tikz plot in a file.tex (to 1D plots)
    2: 'png,300', # savefig as png with dpi scaling
    3: 'none' # interactive show
}[3]
plotStyle = {
    1: "seaborn-v0_8-talk", # To talks
    2: "ggplot", # To soft background and colors
    3: "default", # Python standard plot
    4: 'fivethirtyeight',
    5: 'grayscale',
    6: "seaborn-v0_8-pastel",
}[2]

# Set the pyplot graphs style
setPltStyle(plotStyle)

outputPath = '/output/stokes3D/'
outputFileNameRoot = splitPathFile(__file__) # The script file name

functionsToExport = [
    'U', 'P', 'ud', 'pd',
    'f', 'U_g', 'H',
    'fd', 'ugd', 'hd',
    'f0', 'ug0', 'h0'
    ]
outputSolutionsPaths = {
    s: (outputPath.join(outputFileNameRoot)).replace('.py', f'_{s}.pvd')
        for s in functionsToExport
}

# Write a copy for all prints in the below file
outputData = {
    1: (outputPath.join(outputFileNameRoot)).replace(".py", "_data.txt"),
    2: None
}[1]

# Finite Element basis and degree
Th = 'P1[:], P1'

# Set the stabilizator scalar
beta = {
    1: "CellDiameter(mesh)**2",
    2: "Constant(mesh.hmax()**2, name='β')"
}[2]

def stabTerm(*trialTest, dm=dx):
    # Get the global beta value
    global beta

    # Init a empty form
    integrand = 0

    # Looping in pairs (trial, test)
    for trial, test in trialTest:
        # Add the velocity stabilization parcel
        integrand += inner(grad(trial), grad(test))\
            if any(z.ufl_shape)\
            else dot(grad(trial), grad(test))

    # Multiply by beta scalar and integrate
    term = beta*integrand*dm

    return term

# ================
# Optimal Settings
# ================
# Controls variables
controls = {
    1: ['f', 'ug', 'h'], # Total control
    2: ['ug', 'h'], # Boundary controls
    3: ['f', 'h'], # Mixed Controls (Neumann)
    4: ['f', 'ug'], # Mixed Controls (Dirichlet)
    5: ['ug'], # One Control
    6: [] # Without controls solve only the state equation
    }[2]

linSolver = {
    # Default
    0: 'tfqmr',
    # Iteractive solvers
    1.1: 'tfqmr',
    1.2: 'bicgstab',
    # Direct solvers
    2.0: 'mumps'
}[0]
preconditioner = {
    # Default
    0: 'jacobi',
    # Options
    1.0: 'none',
    1.1: 'jacobi',
    1.2: 'hypre_amg',
    1.3: 'ml_amg'
}[0]

# System solve mode
allAtOnce = True

# Cost coefficients
a_s = {
    'z': (10, 1e-1),
    #'f': (1e-2, 1e-2),
    'ug': (1e-2, 1e-2),
    'h': (0, 1e-1)
}

# Cost measures
dm = {
    'z': 'dx',
    #'f': 'dx',
    'ug': 'ds["inlet"]',
    'h': 'ds["outlet"]'
}

# Descent step size
rho = 10

# Previous direction weight
gamma = {
    1: '1',
    2: '1/8 * df.norm(dJk)**2 / df.norm(dJk_1)**2', # The Best
    3: '1/8 * (dJkFunc * (dJkFunc - dJk_1Func)) / df.norm(dJk_1)**2'
    }[2]

# =========================
# Exact and Input Functions
# =========================
# Active the validation mode
validation = True

def ud(*x):
    'Analytical Velocity'
    R = 1
    r = x[1]**2 + x[2]**2
    return (1 - r/R**2, 0, 0)


if validation:
    def pd(*x):
        'Analytical Pressure'
        return 2*(4 - x[0])


    def fd(*x):
        'Source term on Omega'
        return -_ud.divgrad(*x) + _pd.grad(*x)


    def ugd(*x):
        'Dirichlet value on Γ_D'
        return _ud(*x)


    def hd(*x):
        'Neumann value on Γ_N'
        return (_ud.grad(*x) - _pd(*x)*I) @ normals['outlet']

    def gW(*x):
        'Dirichlet value on Γ_W'
        return _ud(*x)

    if 'f' not in controls:
        def f(*x):
            'Source term'
            return -_ud.divgrad(*x) + _pd.grad(*x)

    if 'ug' not in controls:
        def g(*x):
            'Dirichlet value on Γ_D'
            return _ud(*x)

    if 'h' not in controls:
        def h(*x):
            'Neumann value on Γ_N'
            return (_ud.grad(*x) - _pd(*x)*I) @ normals['outlet']

else:

    def gW(*x):
        'Dirichlet value on Γ_W'
        return [0, 0]

    if 'f' not in controls:
        def f(*x):
            'Source term'
            return [0, 0]

    if 'ug' not in controls:
        def g(*x):
            'Dirichlet value on Γ_D'
            return x[1]*(2 - x[1]) * -normals['inlet']

    if 'h' not in controls:
        def h(*x):
            'Neumann value on Γ_N'
            return 10 @ normals['outlet']

# ===============
# Inital Controls
# ===============
if not allAtOnce:
    # Exact percentage
    percent = 0. # in [0, 1]

    def f0(*x):
        return percent*(-_ud.divgrad(*x) + _pd.grad(*x))


    def ug0(*x):
        'Dirichlet value on Γ_D'
        return percent*_ud(*x)


    def h0(*x):
        'Neumann value on Γ_N'
        return percent*((_ud.grad(*x) - _pd(*x)*I) @ normals['outlet'])

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
    expression = a['z'] @ normH1a(z + ug - ud, dm['z'])
    expression += sum(
        [a[cName] @ normH1a(c, dm[cName])
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
        grad += a @ normH1aDiff(u, dm, v)

    return grad

def D(u, choice=1):
    if choice == 1:  # [Poiselle]
        output = grad(u)
    elif choice == 2:  # [Turek]
        output = 2*sym(grad(u))
    return output
# ==============
# Input Data End
# ==============

if outputData is not None:
    # Create the folder if it doesn't exist
    createFolder(outputData)

    # Create the txt file
    outputData = open(outputData, 'w', encoding='utf-8')

# Turn to analytical function object
_ud = AnalyticalVectorFunction(ud(x, y, z), toccode=True)
if validation:
    _pd = AnalyticalFunction(pd(x, y, z), toccode=True)

# Check if is a optimal problem
optimal = any(controls)

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
_ds = Measure(
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
ds = {bound: _ds(boundaryMark[bound]) for bound in boundaryMark}

# Copy the measures labels
dms = dm.copy()

# Looping in controls
for c in a_s.keys():
    # Turn to respective coefficients to constant
    a_s[c] = array(a_s[c])

    # Evaluate the respective cost measure
    dm[c] = eval(replaceBoundNameByMark(f"{dm[c]}", boundaryMark))

# Get the mesh inner nodes
innerNodes = getInnerNodes(mesh, bmesh)

# Get the boundaries submesh
boundarySubMeshes = getSubMeshes(boundaryData, boundaryMark)

# Init the solutions meshes map
solMeshes = {}

# Looping in solutions and measures labels
for solLbl, _dm in dms.items():
    # Set the respective mesh
    _mesh = mesh\
        if _dm == 'dx'\
        else boundarySubMeshes[_dm.strip("ds[]]\"\'")]

    # Store the respective solution's mesh
    solMeshes[solLbl] = _mesh

if normals is None:
    # Get the normals
    normals = {bound: getNormal(boundMesh)
                for bound, boundMesh in boundarySubMeshes.items()}

    if invertNormalOrientation:
        # Invert each normal orientation
        normals = {k: v*-1 for k, v in normals.items()}
else:
    # Turn normals to array
    normals = {k: array(v) for k, v in normals.items()}

if showMesh:
    # Plot the mesh and bmesh nodes by category
    plotMesh(
        innerNodes.T,
        *[subMesh.coordinates().T
            for subMesh in boundarySubMeshes.values()],
        labels=['inner', *boundarySubMeshes.keys()],
        projection3D=True, normals=normals
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
Th = MixedElement(elements)

# Set the function space
W = FunctionSpace(mesh, Th)

if allAtOnce and optimal:
    # Get the state and adjoint subspaces
    V, Q, VAdj, QAdj = W.split()[:4]

    # Get the controls subspaces
    VOpt = dict(zip(controls, W.split()[4:]))

elif not optimal:
    # Set the state spaces
    V, Q = W.split()

    # Set default values
    VAdj = QAdj = None
    VOpt = {}

else:
    # Set the control functions space
    WControl = [
        FunctionSpace(mesh, elements[0])
            for _ in range(len(controls))
    ]

    # Get the state subspaces
    V, Q = W.split()

    # Get the adjoint subspaces
    VAdj = V
    QAdj = Q

    # Get the controls subspaces
    VOpt = dict(zip(controls, WControl))

if allAtOnce and optimal:
    # Collapse the respective spaces
    Vc = V.collapse()
    Qc = Q.collapse()
    VAdjC = VAdj.collapse()
    QAdjC = QAdj.collapse()

else:
    # Get state spaces
    Vc = VAdjC = V.collapse()
    Qc = QAdjC = Q.collapse()

VOptC = dict((c, S.collapse()) for c, S in VOpt.items())\
    if allAtOnce\
    else VOpt

# Get the function space dofs and store it in itself
W.dofs = len(W.dofmap().dofs())

# Get the matrix nonzero amount
thrdNnz, W.nnz = parallel(calculeNnz, W, dx)

# Set the test functions empty lists
v = (1 + optimal + len(controls))*[0]
q = [0] + optimal*[0]

# Set the Trial and Test Functions
if allAtOnce and optimal:
    # Get the state and adjoint trial functions
    z, p, l_z, l_p = TrialFunctions(W)[:4]

    # Get the optimality conditions trial functions
    c = dict(zip(controls, TrialFunctions(W)[4:]))

    # Get the state and adjoint test functions
    v[0], q[0], v[1], q[1] = TestFunctions(W)[:4]

    # Get the optimality conditions test functions
    v[2:] = TestFunctions(W)[4:]

elif not optimal:
    # Get the state trial functions
    z, p = TrialFunctions(W)

    # Get the state test functions
    v[0], q[0] = TestFunctions(W)

else:
    # Get the state and adjoint trial functions
    z, p, l_z, l_p = 2*TrialFunctions(W)

    # Get the control solution vectors
    C = [Function(S, name=c.upper())
            for c, S in zip(controls, WControl)]

    # Get the control solution vectors
    c = dict(zip(controls, C))

    # Get the state and adjoint test functions
    v[0], q[0], v[1], q[1] = 2*TestFunctions(W)

    # Get the gradient cost trial function
    dJ = [TrialFunction(S) for S in WControl]

    # Get the optimality conditions test functions
    v[2:] = [TestFunction(S) for S in WControl]

# Turn input functions to expression
udExpr = setExpression(_ud, elements[0], name='ud')
if validation:
    pdExpr = setExpression(_pd, elements[1], name='pd')
gWExpr = setExpression(gW, elements[0], name='gW')
if 'f' not in controls:
    fExpr = setExpression(f, elements[0], name='f')
if 'ug' not in controls:
    gExpr = setExpression(g, elements[0], name='g')
if 'h' not in controls:
    hExpr = setExpression(h, elements[0], name='h')

# Set the current input functions
ud = interpolate(udExpr, Vc); ud.rename('ud', 'ud')
if validation:
    pd = interpolate(pdExpr, Qc); pd.rename('pd', 'pd')
gW = interpolate(gWExpr, Vc); gW.rename('gW', 'gW')
if 'f' not in controls:
    f = interpolate(fExpr, Vc); f.rename('f', 'f')
if 'ug' not in controls:
    g = interpolate(gExpr, Vc); g.rename('g', 'g')
if 'h' not in controls:
    h = interpolate(hExpr, Vc); h.rename('h', 'h')

# Looping in controls
for cLbl in controls:

    if validation:
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
initialControls = {c: eval(f'{c}0') for c in controls}\
    if not allAtOnce\
    else {}

if validation:
    # Group by category the exact data
    exactData = {'ud': ud, 'pd': pd}\
        | {c: eval(f'{c}d') for c in controls}

else:
    # Store the observation data
    exactData = {'ud': ud}

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
    ZPLZLPC = Function(W, name='ZPLZLPC')
else:
    # Set the state problem solution
    ZP = Function(W, name='ZP')

    # Set the adjoint problem solution
    LZLP = Function(W, name='LZLP')

# Set the state problem solutions
U = Function(Vc, name='U')
P = Function(Qc, name='P')

# Set the variational system
aState = 'inner(D(z), D(v[0]))*dx\
    + div(z)*q[0]*dx\
    - p*div(v[0])*dx'
LState = 'dot(f, v[0])*dx \
    + dot(h, v[0])*ds["outlet"]'
if 'ug' in controls:
    LState += '- inner(D(ug), D(v[0]))*dx - div(ug)*q[0]*dx'

if optimal:
    # Get the gradJ with respect to state var
    gradJ_z = gradJ((a_s['z'], z + ug - ud, dm['z']), v=v[1])\
        if 'ug' in controls\
        else gradJ((a_s['z'], z - ud, dm['z']), v=v[1])\

    # Get adjoint variational system
    aAdjoint, LAdjoint = getAdjointSystem(
        aState,
        [('z', 'v[1]'), ('p', 'q[1]'), ('v[0]', 'l_z'), ('q[0]', 'l_p')],
        gradJ_z,
        ZP if not allAtOnce else None,
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
        [('v[0]', 'l_z'), ('q[0]', 'l_p')],
        gradsJ,
        liftingStateData,
        **(dict(dJ=dJ, v=v[2:], dm=dm, Z_L=(ZP, LZLP))
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
inletDirichlet = zeros(gdim)\
    if 'ug' in controls\
    else g

# Set the state boundary conditions
bcsState = [
    DirichletBC(V, inletDirichlet, boundaryData, boundaryMark['inlet']),
    DirichletBC(V, gW, boundaryData, boundaryMark['wall'])
]

# Set the adjoint boundary conditions
bcsAdj = [
    DirichletBC(VAdj, zeros(gdim), boundaryData, boundaryMark['inlet']),
    DirichletBC(VAdj, zeros(gdim), boundaryData, boundaryMark['wall'])
] if optimal else []

if allAtOnce:
    # Append dirichlet boundary conditions of the adjoint problem
    bcs = bcsState + bcsAdj

else:
    # Split dricihlet boundary conditions by problem
    bcs = {'state': bcsState, 'adjoint': bcsAdj}

# Set the solver
solver = setSolver(linSolver, preconditioner)

# Evaluate the beta expression chosen
beta = eval(beta)

# Add the stabilizator term
aState += stabTerm((p, q[0]), dm=dx)

if optimal:
    # Add the stabilizator to adjoint equation
    aAdjoint += stabTerm((l_p, q[1]), dm=dx)

    # Get the h control variable index
    hIdx = _controls.index('h') if 'h' in controls else None
    fIdx = _controls.index('f') if 'f' in controls else None
    ugIdx = _controls.index('ug') if 'ug' in controls else None

    if fIdx is not None and allAtOnce:
        # Set the respective trial function
        trial = f\
            if allAtOnce\
            else dJ[fIdx]

        # Add the stabilizator to "f" optimality condition
        aOptimal += stabTerm((trial, v[2+fIdx]), dm=dx)

    if globals().get('ugIdx', None) is not None:
        # Set the respective trial function
        trial = ug\
            if allAtOnce\
            else dJ[ugIdx]

        # Add the stabilizator to "ug" optimality condition
        aOptimal += stabTerm((trial, v[2+ugIdx]), dm=ds['inlet'])

    if hIdx is not None:
        # Set the respective trial function
        trial = h\
            if allAtOnce\
            else dJ[hIdx]

        # Add the stabilizator to "h" optimality condition
        aOptimal += stabTerm((trial, v[2+hIdx]), dm=dx)

# Get the trial functions of the collapsed subspaces
# (for the approximated errors calculus)
u = TrialFunction(Vc)
p = TrialFunction(Qc)
c = {c: TrialFunction(VOptC[c]) for c in controls}

# Set the error formula to observation data
errors = {
    'u': getErrorFormula(ud, dm['z'], relative=True)
}

if validation:
    # Set the error formulas
    errors.update({
        'p': getErrorFormula(pd, dm['z'], relative=True),
        **{c: getErrorFormula(eval(f'{c}d'), dm[c], relative=True)
                for c in controls}
    })

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
    _ds, boundaryMark,
    getFunctionExpressions(getInputExactData(globals())),
    normals,
    g if 'ug' not in controls else None,
    a_s, dm, copyTo=outputData
)

if allAtOnce:
    # Set the full system
    fullSystem = aState + aAdjoint + aOptimal \
        - (LState + LAdjoint + LOptimal)

    # Couple the variational formulations
    a, L = system(fullSystem)

    # Solve the all at once system
    mySolve(a == L, ZPLZLPC, bcs, solver,
            runtime=True, nnz=W.nnz, dofs=W.dofs,
            copyTo=outputData)

else:
    # Split the problems
    a = [aState, aAdjoint, aOptimal]
    L = [LState, LAdjoint, LOptimal]
    w = [ZP, LZLP, C]

    # Get a solver copy
    solverCopy = copySolver(solver)

    # Solve the system by gradient descent method
    errors = gradientDescent(
        _controls, J, a_s, dm, *zip(a, L, w), bcs, (solver, solverCopy),
        exactData, initialControl=initialControls, rho=rho,
        gamma=gamma, errorForm=errors if validation else None,
        copyTo=outputData
    )[2]

# Split the solutions
if allAtOnce and optimal:
    # Get splitted solutions
    ZPLZLPC = ZPLZLPC.split(deepcopy=True)

    # Get solutions by problem
    Z, _P = ZPLZLPC[:2]  # State
    LZ, LP = ZPLZLPC[2:4]  # Adjoint
    C = dict(zip(controls, ZPLZLPC[4:])) # Controls

elif not optimal:
    # Get solutions
    Z, _P = ZPLZLPC.split(deepcopy=True)

else:
    # Get solutions by problem
    Z, _P = ZP.split(deepcopy=True)  # State
    LZ, LP = LZLP.split(deepcopy=True)  # Adjoint
    C = dict(zip(controls, C))

if optimal:
    # Rename the adjoint and controls solutions
    LZ.rename('LZ', 'LZ')
    LP.rename('LP', 'LP')
    [C[c].rename(c, c) for c in controls]

# Fill the pressure vector
P.assign(_P)

# Recover the original U solutions
U.assign(Z)

if 'ug' in controls:
    # Add the lifting contribution
    setLocal(U, getLocal(U) + getLocal(C['ug']))

# Set the evaluate errors function arguments
errorsArgs = [(U, errors['u'])]
errorsKwargs = {
    'labels':['u'],
    'relative': type(errors['u']['L²']) is tuple
}

if allAtOnce and optimal:
    # Evaluate and show the cost
    evaluateCost(J, Z, ud, C, a_s, dm, show=True, copyTo=outputData)

    if validation:
        # Add the pressure and controls error formulas and labels
        errorsArgs += [
            (P, errors['p']),
            *[(C[c], errors[c]) for c in controls]
            ]
        errorsKwargs['labels'] += ['p', *controls]

elif not optimal and validation:
    # Add the pressure error formula and label
    errorsArgs += [(P, errors['p'])]
    errorsKwargs['labels'] += ['p']

# Calcule approach errors
errors = evaluateErrors(*errorsArgs, **errorsKwargs)

# Show the approach error
showError(errors, omg, precision=3, copyTo=outputData)

if showSolution:
    # Set the common args
    commonArgs = {
        'splitSols': True,
        'projection3D': True,
        'show': False,
        'interactive': False,
        'personalPlot': False
        }

    # Plot the velocity comparison
    fig_u = plotComparison(U, ud, mesh=solMeshes['z'], **commonArgs)

    # Plot the pressure comparison
    fig_p = plotComparison(P, pd, mesh=solMeshes['z'], **commonArgs)

    # Init a figures list
    figs = [fig_u, fig_p]

    # Looping in controls labels
    for c in controls:
        # Plot the respective control comparison plot
        fig = plotComparison(
            exactData[c], mesh=solMeshes[c], **commonArgs
        )

        # Plot the respective control comparison plot
        fig_d = plotComparison(
            C[c], mesh=solMeshes[c], **commonArgs
        )

        # Append to figures list
        figs.extend([fig, fig_d])

    # Distribute figures in screen
    adjustFiguresInScreen(*figs)

    if showSolutionType == 'none':
        args = []

    else:
        # Set the output path and file name
        args = [outputPath.join(outputFileNameRoot).replace('.py', '')]

        if showSolutionType == 'tex':
            # Add the file extension
            args += ['tex']

        elif ',' in showSolutionType:
            # Get the file extension and dpi
            extension, dpi = showSolutionType.split(',')

            # Add it
            args += [extension, int(dpi)]

        elif showSolutionType in imageTypes():
            # Add the file extension
            args += [showSolutionType]

        else:
            # Wrong choice
            args = []

    # Show the plots
    show(
        *args,
        grid=plotStyle not in ['ggplot', 'bmh', 'fivethirtyeight',
                               'grayscale']
        )

if optimal:
    # Set the solutions to export pvd file
    allFunctions = [
        U, P, *C.values(),
        *exactData.values(),
        *initialControls.values()
    ]
else:
    # Set the solutions to export pvd file
    allFunctions = [
        U, P, *exactData.values()
    ]

showInfo("Exporting solutions to PVD")
# Looping in solutions to export
for sol in allFunctions:

    print(f'  {sol.name():>2}: {outputSolutionsPaths[sol.name()]}')
    if sol.name() in outputSolutionsPaths:
        # Create respective file
        file = File(outputSolutionsPaths[sol.name()])

    # Export the respective solution
    file.write(sol)

try:
    os.system(
        f"paraview {os.path.dirname(outputSolutionsPaths[sol.name()])}/*.pvd"
    )
except Exception:
    pass

if outputData is not None:
    # Close the external txt file
    outputData.close()

