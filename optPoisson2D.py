'''
Optimal Poisson-based problem on 2D domain
'''

from dolfin import *
from opytimal import *
from opytimal.settings import QUADRATURE_DEG

# ================
# Input Data Begin
# ================
num = 4
meshPath = f'./InputData/Meshes/2D/rectangle{num}'
boundaryDataPath = f'./InputData/Meshes/2D/rectangle{num}_mf'

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
showSolution = True
showSolutionMode = {
    1: 'solutions',
    2: 'numerical_error',
    3: 'error', # Absolute difference node by node divided by |exact|_oo
    4: 'numerical',
    5: 'exact'
}[4]
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

outputPath = '/output/poisson2D/'
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
    2: "Constant(0)"#"Constant(mesh.hmax()**2, name='β')"
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
    5: ['ug'],
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
    'f': (0, 1e-2),
    'ug': (0, 1e-3),
    'h': (0, 1e-4)
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

def ud(*x):
    return x[1]*(2 - x[1]) + 1e-1*x[0]


if validation:
    def fd(*x):
        return -_ud.divgrad(*x)

    def ugd(*x):
        'Dirichlet value on Γ_D'
        return _ud(*x)

    def hd(*x):
        'Neumann value on Γ_N'
        return _ud.grad(*x) @ normals['outlet']

    def gW(*x):
        'Dirichlet value on Γ_W'
        return _ud(*x)

    if 'f' not in controls:
        def f(*x):
            'Source term'
            return -_ud.divgrad(*x)

    if 'ug' not in controls:
        def g(*x):
            'Dirichlet value on Γ_D'
            return _ud(*x)

    if 'h' not in controls:
        def h(*x):
            'Neumann value on Γ_N'
            return _ud.grad(*x) @ normals['outlet']

else:

    def gW(*x):
        'Dirichlet value on Γ_W'
        return 0

    if 'f' not in controls:
        def f(*x):
            'Source term'
            return 0

    if 'ug' not in controls:
        def g(*x):
            'Dirichlet value on Γ_D'
            return x[1]*(2 - x[1])

    if 'h' not in controls:
        def h(*x):
            'Neumann value on Γ_N'
            return 10

# ===============
# Inital Controls
# ===============
if not allAtOnce:
    # Exact percentage
    percent = 0. # in [0, 1]

    def f0(*x):
        return percent*(-_ud.divgrad(*x))


    def ug0(*x):
        'Dirichlet value on Γ_D'
        return percent*_ud(*x)


    def h0(*x):
        'Neumann value on Γ_N'
        return percent*(_ud.grad(*x) @ normals['outlet'])

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
    evaluate: bool = True,
    splitParcels: bool = False
        ) -> (Union[Form, float]):

    if type(z) is tuple:
        # Get the first state solutions
        z = z[0]

    # Get the lift
    ug = controls.get('ug', Zero(z.function_space()))

    if not splitParcels:
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

    else:
        # Multiply the parameters and set the state parcels
        stateParcels = [*(a['z'].values() * normH1a(z + ug - ud, dm['z']))]

        # Multiply the parameters and set the controls parcels
        controlsParcels = [
            (a[cName].values() * normH1a(c, dm[cName]))
                for cName, c in controls.items()
        ]

        # Join the parcels
        parcels = stateParcels + flatten(controlsParcels).tolist()

        # Set the output
        output = sum([assemble(parcel) for parcel in parcels])\
            if evaluate\
            else sum(parcels)

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
# Set the pyplot graphs style
setPltStyle(plotStyle)

if outputData is not None:
    # Create the folder if it doesn't exist
    createFolder(outputData)

    # Create the txt file
    outputData = open(outputData, 'w', encoding='utf-8')

# Turn to analytical function object
_ud = AnalyticalFunction(ud(x, y, z), toccode=True)

# Check if is a optimal problem
optimal = any(controls)

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
_ds = Measure(
    "ds",
    domain=mesh,
    subdomain_data=boundaryData,
    metadata={"quadrature_degree": QUADRATURE_DEG}
)

# Set the respective boundary measures
ds = {bound: _ds(boundaryMark[bound]) for bound in boundaryMark}

# Copy the measures labels
dms = dm.copy()

# Looping in controls
for c in a_s.keys():
    # Turn to respective coefficients to constant
    a_s[c] = Constant(a_s[c], name=f"a_{c}")

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
        normals=normals
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
udExpr = setExpression(_ud, elements[0], name='ud')
gWExpr = setExpression(gW, elements[0], name='gW')
if 'f' not in controls:
    fExpr = setExpression(f, elements[0], name='f')
if 'ug' not in controls:
    gExpr = setExpression(g, elements[0], name='g')
if 'h' not in controls:
    hExpr = setExpression(h, elements[0], name='h')

# Set the current input functions
ud = interpolate(udExpr, Vc); ud.rename('ud', 'ud')
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
    exactData = {'u': ud}\
        | {c: eval(f'{c}d') for c in controls}

else:
    # Store the observation data
    exactData = {'u': ud}

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

# Set the variational system
aState = 'dot(grad(z), grad(v[0]))*dx'
LState = 'dot(f, v[0])*dx + dot(h, v[0])*ds["outlet"]'
if 'ug' in controls:
    LState += '- dot(grad(ug), grad(v[0]))*dx'

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
        **(dict(dJ=dJ, v=v[2:], dm=dm, Z_L=(Z, LZ))
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
inletDirichlet = 0\
    if 'ug' in controls\
    else g

# Set the state boundary conditions
bcsState = [
    DirichletBC(V, inletDirichlet, boundaryData, boundaryMark['inlet']),
    DirichletBC(V, gW, boundaryData, boundaryMark['wall'])
]

# Set the adjoint boundary conditions
bcsAdj = [
    DirichletBC(VAdj, 0, boundaryData, boundaryMark['inlet']),
    DirichletBC(VAdj, 0, boundaryData, boundaryMark['wall'])
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

# Set the error formulas
errors = {
    'u': getErrorFormula(ud, dm['z'], relative=True)
}
if validation:
    # Set the error formulas
    errors.update({
        **{c: getErrorFormula(eval(f'{c}d'), dm[c], relative=True)
                for c in controls}
    })

# Set the domain labels
omg = getDomainLabels(
    {lbl: getMeasure(err['L²'][0].integrals()[0])
        for lbl, err in errors.items()},
    boundaryMark
    )

# Stop the nnz calculation thread
thrdNnz.join()

# Get value from list
W.nnz = W.nnz[0]

# Show the program data
showProblemData(
    f'Optimal Poisson-based Problem on {basename(meshPath)}',
    'validation' if validation else 'simulation',
    Th, W, bcsState if allAtOnce else bcs['state'],
    _ds, boundaryMark,
    getFunctionExpressions(getInputExactData(globals())),
    normals,
    g if 'ug' not in controls else None,
    a_s, dm, copyTo=outputData
)

if allAtOnce:
    # Couple the variational formulations
    a, L = system(
        aState + aAdjoint + aOptimal
        - (LState + LAdjoint + LOptimal)
    )

    # Solve the all at once system
    mySolve(a == L, ZLZC, bcs, solver,
            runtime=True, nnz=W.nnz, dofs=W.dofs,
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
        gamma=gamma, errorForm=errors if validation else None,
        copyTo=outputData
    )[2]

# Split the solutions
if allAtOnce and optimal:
    # Get splitted solutions
    ZLZC = ZLZC.split(deepcopy=True)

    # Get solutions by problem
    Z, LZ = ZLZC[:2]  # State and adjoint
    C = dict(zip(controls, ZLZC[2:])) # Controls

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
            *[(C[c], errors[c]) for c in controls]
            ]
        errorsKwargs['labels'] += [*controls]

# Calcule approach errors
errors = evaluateErrors(*errorsArgs, **errorsKwargs)

# Show the approach error
showError(errors, omg, precision=3, copyTo=outputData)

if showSolution:
    # Set the common args
    commonArgs = {
        'splitSols': True,
        'show': False,
        'interactive': False,
        'personalPlot': False,
        'position': 'top'
    }

    if commonArgs.get('personalPlot', False):
        commonArgs['projection3D'] = False
        commonArgs['splitSols'] = False

    if showSolutionMode == 'solutions':
        # Show the solutions
        sols2Show = {
            'u': (exactData['u'], U),
        } | {
            c: (exactData[c], C[c])
                for c in controls
        }

    elif showSolutionMode == 'numerical':
        # Show the solutions
        sols2Show = {
            'u': (U,),
        } | {
            c: (C[c],)
                for c in controls
        }

    elif showSolutionMode == 'exact':
        # Show the solutions
        sols2Show = {
            'u': (exactData['u'], ),
        } | {
            c: (exactData[c], )
                for c in controls
        }

    elif 'error' in showSolutionMode:
        # Set the errors functions
        E = {
            'u': U.copy(deepcopy=True)
        } | {
            c: C[c].copy(deepcopy=True)
                for c in controls
        }

        # Put name in errors functions
        {v.rename(*(2*(f'E_{r"{"+k.title()+r"}"}',)))
            for k, v in E.items()}

        # Calcule the absolute difference between numerical and
        # exact functions
        [setLocal(
            E[s], abs(
                (getLocal(E[s]) - getLocal(exactData[s]))
                    /abs(getLocal(exactData[s])).max()
            ))
            for s in E.keys()]


        if "numerical" in showSolutionMode:
            # Get the solutions map
            sols = {'u': U, **C}

            # Join the solutions to plot
            sols2Show = {k: (sols[k], v) for k, v in E.items()}

        else:
            # Set the errors to plot
            sols2Show = {k: (v,) for k, v in E.items()}

    # Plot the solution comparison
    fig = plotComparison(
        *sols2Show['u'], mesh=solMeshes['z'], **commonArgs
    )

    # Init a figures list
    figs = [fig]

    # Looping in controls labels
    for c in controls:
        # Plot the respective control comparison plot
        fig = plotComparison(
            *sols2Show[c], mesh=solMeshes[c], **commonArgs
        )

        # Append to figures list
        figs.append(fig)

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
                               'grayscale'],
        label=showSolutionMode,
        props=graphProperty
        )

# Set the solutions to export pvd file
allFunctions = [
    U, *C.values(),
    *exactData.values(),
    *initialControls.values()
]

# Looping in solutions to export
for sol in allFunctions:

    if sol.name() in outputSolutionsPaths:
        # Create respective file
        file = File(outputSolutionsPaths[sol.name()])

    # Export the respective solution
    file.write(sol)

if outputData is not None:
    # Close the external txt file
    outputData.close()
