'''
Optimal Control Heat-based Evolutive Problem on 1D domain
'''

from dolfin import *
from opytimal import *
from opytimal.settings import QUADRATURE_DEG, THREADS

# ================
# Input Data Begin
# ================
Tf = 1
teta = 0.5
dt = {
    1: "mesh.hmax()",
    2: 1e-3
}[1]

meshLimits = (0, 4)
meshNel = 128

boundaryMark = {
    'inlet': 1,
    'outlet': 2,
}

normals = {
    'inlet': [-1, 0, 0],
    'outlet': [1, 0, 0]
}

invertNormalOrientation = False # Multiply all normals to -1

showMesh = False # Turn to True to check the normal vectors
showSolution = True
showSolutionMode = {
    1: 'solutions',
    2: 'numerical_error',
    3: 'error', # Absolute difference node by node divided by |exact|_oo
    4: 'numerical',
    5: 'exact'
}[3]
showSolutionType = {
    1: 'tex', # generate a tikz plot in a file.tex
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

# Set the global graph properties
graphProperty = {
    'ticksFont': 12,
    'labelsFont': 12,
    'noTitles': True,
    'noColorbarLabel': True
}

# Set the pyplot graphs style
setPltStyle(plotStyle)

outputPath = '/output/heat1D/'
outputFileNameRoot = splitPathFile(__file__) # The script file name

functionsToExport = [
    'U', 'ud',
    'f','ug','h',
    'fd','ugd','hd',
    'f0','ug0','h0'
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
Th = 'P1'

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
        integrand += dot(grad(trial), grad(test))

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
    5: ['h'],
    6: []
    }[1]

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
    'f': (1e-2, 1e-2),
    'ug': (1e-2, 1e-2),
    'h': (0, 1e-2)
}

# Cost measures
dm = {
    'z': 'dx',
    'f': 'dx',
    'ug': 'ds["inlet"]',
    'h': 'ds["outlet"]'
}

# Descent step size
rho = 10

# Previous direction weight
gamma = {
    1: '1',
    2: '1/8 * norm(dJk)**2 / norm(dJk_1)**2', # The Best
    3: '1/8 * (dJk * (dJk - dJk_1)) / norm(dJk_1)**2'
    }[2]

# =========================
# Exact and Input Functions
# =========================
# Active the validation mode
validation = True

def ud(*x, t=0):
    return x[0]*(4 - x[0]) * exp(-t)

def u0(*x):
    'Inital solution'
    return _ud(*x, t=0)

if validation:

    def fd(*x, t=0):
        return _ud.dt(*x, t=t) - _ud.divgrad(*x, t=t)

    def ugd(*x, t=0):
        'Dirichlet value on Γ_D'
        return _ud(*x, t=t)

    def hd(*x, t=0):
        'Neumann value on Γ_N'
        return _ud.grad(*x, t=t) @ normals['outlet']

    if allAtOnce:
        def f0(*x):
            return _ud.dt(*x, t=0) - _ud.divgrad(*x, t=0)

        def ug0(*x):
            'Dirichlet value on Γ_D'
            return _ud(*x, t=0)

        def h0(*x):
            'Neumann value on Γ_N'
            return _ud.grad(*x, t=0) @ normals['outlet']

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

else:

    if allAtOnce:
        def f0(*x):
            return 0

        def ug0(*x):
            'Dirichlet value on Γ_D'
            return 0

        def h0(*x):
            'Neumann value on Γ_N'
            return 10

    if 'f' not in controls:
        def f(*x, t=t):
            'Source term'
            return 0

    if 'ug' not in controls:
        def g(*x, t=t):
            'Dirichlet value on Γ_D'
            return x[0]*(4 - x[0]) * exp(-t)

    if 'h' not in controls:
        def h(*x, t=t):
            'Neumann value on Γ_N'
            return 10

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

if outputData is not None:
    # Create the folder if it doesn't exist
    createFolder(outputData)

    # Create the txt file
    outputData = open(outputData, 'w', encoding='utf-8')

# Turn to analytical function object
_ud = AnalyticalFunction(ud(x, y, z, t=t), toccode=True)

# Check if is a optimal problem
optimal = any(controls)

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
_ds = Measure(
    "ds",
    domain=mesh,
    subdomain_data=boundaryData,
    metadata={"quadrature_degree": QUADRATURE_DEG}
)

# Set the respective boundary measures
ds = {bound: _ds(boundaryMark[bound]) for bound in boundaryMark}

# Looping in controls
for c in a_s.keys():
    # Turn to respective coefficients to constant
    a_s[c] = Constant(a_s[c], name=f"a_{c}")

    # Evaluate the respective cost measure
    dm[c] = eval(replaceBoundNameByMark(f"{dm[c]}", boundaryMark))

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
inletNode = (ravel(mesh.coordinates())*inletNodeFinder).sum()
outletNode = (ravel(mesh.coordinates())*outletNodeFinder).sum()

# Adjust the vectors to plot
innerNodes = hstack((innerNodes, 0*innerNodes))
inletNode = [inletNode, 0]
outletNode = [outletNode, 0]

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
        innerNodes.T, inletNode, outletNode,
        labels=['inner', 'inlet', 'outlet']
        )

# # Looping in controls
# for c in list(a_s.keys()):

#     if c not in controls and c != 'z':
#         # Remove respective key
#         a_s.pop(c)
#         dm.pop(c)

#         # Go to next
#         continue

#     # Turn to respective coefficients to constant
#     a_s[c] = Constant(a_s[c], name=f"a_{c}")

#     # Evaluate the respective cost measure
#     dm[c] = eval(replaceBoundNameByMark(f"{dm[c]}", boundaryMark))

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
u0Expr = setExpression(u0, elements[0], name='u0')
if 'f' not in controls:
    fExpr = setExpression(f, elements[0], name='f', t=True)
if 'ug' not in controls:
    gExpr = setExpression(g, elements[0], name='g', t=True)
if 'h' not in controls:
    hExpr = setExpression(h, elements[0], name='h', t=True)

# Set the current input time functions
ud = interpolate(udExpr, Vc); ud.rename('ud', 'ud')
if 'f' not in controls:
    f = interpolate(fExpr, Vc); f.rename('f', 'f')
if 'ug' not in controls:
    g = interpolate(gExpr, Vc); g.rename('g', 'g')
if 'h' not in controls:
    h = interpolate(hExpr, Vc); h.rename('h', 'h')

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

# Looping in controls
for cLbl in controls:

    if validation:
        # Set the respective control input expressions
        exec(f"{cLbl}dExpr = setExpression({cLbl}d, elements[0], name='{cLbl}d')")

        # Set the respective input data functions
        exec(f"{cLbl}d = interpolate({cLbl}dExpr, VOptC['{cLbl}'])")
        exec(f"{cLbl}d.rename('{cLbl}d', '{cLbl}d')")

    # Set the respective initial control expressions
    exec(f"{cLbl}0Expr = setExpression({cLbl}0, elements[0], name='{cLbl}0')")

    # Set the respective intial control functions
    exec(f"{cLbl}0 = interpolate({cLbl}0Expr, VOptC['{cLbl}'])")
    exec(f"{cLbl}0.rename('{cLbl}0', '{cLbl}0')")

    # Set the respective previous time functions
    exec(f"{cLbl}n.assign(interpolate({cLbl}0Expr, VOptC['{cLbl}']))")

if 'ug' in controls:
    setLocal(zn, getLocal(u0) - getLocal(ug0))
else:
    zn.assign(u0)

# Group intial controls by category
initialControls = {c: eval(f'{c}0') for c in controls}

if validation:
    # Group by category the exact datas
    exactData = {'u': ud}\
        | {c: eval(f'{c}d') for c in controls}

    # Group  by category the exact expressions
    exactExpressions = {'u': udExpr}\
        | {c: eval(f'{c}dExpr') for c in controls}

else:
    # Store the observation data and expression
    exactData = {'u': ud}
    exactExpressions = {'u': udExpr}

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
    + dt*teta*h*v[0]*dx + dt*(1-teta)*hn*v[0]*ds["outlet"]\
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
]

# Set the adjoint boundary conditions
bcsAdj = [
    DirichletBC(VAdj, 0, boundaryData, boundaryMark['inlet']),
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

if optimal:
    # Get the h control variable index
    hIdx = _controls.index('h') if 'h' in controls else None

    if hIdx is not None:
        # Set the respective trial function
        trial = h\
            if allAtOnce\
            else dJ[hIdx]

        # Add the stabilizator to "h" optimality condition
        aOptimal += stabTerm((trial, v[2+hIdx]), dm=dx)

# Get the trial functions of the collapsed subspaces
# (for the approach errors calculus)
u = TrialFunction(V)
c = {c: TrialFunction(VOptC[c]) for c in controls}

# Set the error formula to observation data
errorsForm = {
    'u': getErrorFormula(ud, dm['z'], relative=True)
}

if validation:
    # Set the error formulas
    errorsForm.update({
        **{c: getErrorFormula(eval(f'{c}d'), dm[c], relative=True)
                for c in controls}
    })

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
    f'Optimal Heat-based Problem on segment{meshNel}',
    'validation' if validation else 'simulation',
    Th, W, bcsState if allAtOnce else bcs['state'],
    _ds, boundaryMark,
    getFunctionExpressions(getInputExactData(globals())),
    normals,
    g if 'ug' not in controls else None,
    a_s, dm, copyTo=outputData
)

# Assign the initial solution values
U.assign(u0)
ud.assign(u0)

# Set the solutions to export pvd file
allFunctions = [
    U, *exactData.values()
]

# Create the exportation files store
exportFiles = {
    c: File(outputSolutionsPaths[c])
        for c in controls
}

# Looping in solutions to export
for sol in allFunctions + [*initialControls.values()]:

    if sol.name() in outputSolutionsPaths:
        # Create respective file
        exportFiles[sol.name()] = File(outputSolutionsPaths[sol.name()])

    # Export the respective solution
    exportFiles[sol.name()].write(sol, 0)

# Create our progress bar
progressBar = ProgressBar(
    'Solving the evolutive system for all time: ',
    total=Tf,
    label='t',
    formatter=f'1.0{getPow10(_dt)+1}f',
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

if showSolution:
    # Init the exact data storage
    exactDataTime = {k: [v.copy(deepcopy=True)] for k, v in exactData.items()}

    # Init the numerical data storage
    numericalDataTime = {k: [] for k in ['u'] + _controls}

# Init the cost values list
costTime = []

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
                copyTo=outputData)

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
            gamma=gamma, errorForm=errorsForm if validation else None,
            copyTo=outputData
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
        Jk = evaluateCost(J, Z, ud, C, a_s, dm, show=False, copyTo=outputData)

        # Store the current time cost value
        costTime.append(Jk)

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

    # Export respective solution
    [exportFiles[sol.name()].write(sol, t)\
        for sol in allFunctions + [*C.values()]]

    if showSolution:
        # Store the current numerical solutions
        numericalDataTime['u'].append(U.copy(deepcopy=True))
        [numericalDataTime[c].append(C[c].copy(deepcopy=True))
            for c in controls]

    # Update the progrss list in parallel
    thrd, _ = parallel(
        progressBar.update,
        t,
        suffix=showErrors(
            {k: (array(v)**2).sum()**0.5 for k, v in errors['u'].items()},
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
            arrNorm(array(error)[:, 0], 'l2')
            #(array(error)[:, 0]**2).sum()**0.5,
        )
        errorsTime[k]['L∞'][eType] = (
            arrNorm(array(error)[:, 0], 'loo')
            #array(error)[:, 0].max(),
        )

        # # Relative errors
        # for normType in ['L²', 'L∞']:
        #     errorsTime[k][normType][eType] += (
        #         errorsTime[k][normType][eType]/exactTimeNorms[k]['L²'],
        #     )

# Finalize the cost time integral
costTime = sum(costTime)*_dt

# Show the cost value
showInfo('Cost Value', copyTo=outputData)
showInfo(
    f'J(u, {", ".join(_controls)}) = {costTime:1.03e}',
    breakStart=False, delimiters=False, tab=4, copyTo=outputData
    )

# Show the approach errors
for k, errors_k in errorsTime.items():
    showInfo(
        f"||{k.upper()} - {k}d||_Y(0,T; X(Ω))",
        copyTo=outputData
        )
    showInfo(
        *ravel(
            [[f'{eTypeT}, {eType} : {e:1.03e}'
                for eType, e in errors.items()]
            for eTypeT, errors in errors_k.items()]
        ), alignment=':', delimiters=False, tab=4,
           breakStart=False, copyTo=outputData
    )

if showSolution:
    # Show the dynamic plot
    fig = dynamicComparison(
        (numericalDataTime['u'], exactDataTime['ud']),
        iterator=arange(0, Tf+_dt, _dt),
        labels=['U', 'ud'],
        linestyles=['-', ''],
        markers=['o', '*'],
        multipleViews=False,
        show=False
        )

    # Init a figures list
    figs = [fig]

    # for c in controls:
    #     # Show the dynamic plot
    #     fig = dynamicComparison(
    #         (numericalDataTime[c], exactDataTime[c]),
    #         iterator=arange(0, Tf+_dt, _dt),
    #         labels=[c.upper(), f'{c}d'],
    #         linestyles=['-', ''],
    #         markers=['o', '*'],
    #         multipleViews=False,
    #         show=False
    #         )

    #     # Append to the figures list
    #     figs.append(fig)

    # Distribute figures in screen
    adjustFiguresInScreen(*figs)

    # Show the plots
    show()

if outputData is not None:
    # Close the external txt file
    outputData.close()

# Stop all parallel process
[thrd.join() for thrd in THREADS]
