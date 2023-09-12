'''
Optimal Stokes-based problem on 3D domain
'''

# from modules import *
# from settings import settings

from dolfin import *

from backend.settings import QUADRATURE_DEG
from backend.symbols import x, y, z, t, symbols
from backend.string import showInfo, splitPathFile, replaceProgressive, basename
from backend.meshes import (readXDMFMesh, getInnerNodes, getSubMeshes,
                            getCoordinates, getNormal)
from backend.plots import (plotMesh, plotComparison, adjustFiguresInScreen,
                           show, figure)
from backend.tests import testLoop
from backend.fenics import (setExpression, extractElements, calculeNnz, mySolve,
                            setSolver, getErrorFormula, gradJ, copySolver,
                            getFormTerm, emptyForm, gradientDescent, showError,
                            getDomainLabels, replaceBoundNameByMark,
                            getMeasure, evaluateErrors, Zero, getAdjointSystem,
                            getOptimalConditionsSystem, evaluateCost,
                            getLocal, setLocal, showProblemData,
                            getInputExactData, getFunctionExpressions)
from backend.analytical import (AnalyticalFunction, AnalyticalVectorFunction)
from backend.parallel import parallel
from backend.arrays import (identity, zeros, getComplement, array, concatenate)
from backend.types import Tuple, Union

import numpy as np
values = np.meshgrid(
    [1e-5, 1e-2, 0, 1, 1e1],
    [1e-5, 1e-2, 0, 1, 1e1]
    )

for vals in zip(np.ravel(values[0]), np.ravel(values[1])):
    for _c in ['f', 'ug', 'h']:
        externalTxt = open(
            f'./{_c}_{vals[0]}_{vals[1]}.txt',
            'w',
            encoding='utf-8'
        )

        from backend.symbols import x, y, z, t, symbols

        # ================
        # Input Data Begin
        # ================
        meshPath = './InputData/Meshes/3D/cylinder2'
        boundaryDataPath = './InputData/Meshes/3D/cylinder2_mf'

        invertNormalOrientation = True
        showMesh = False

        exportSolutionsPath = {
            'U': ('/output/'.join(splitPathFile(__file__))).replace('.py', '_U.pvd'),
            'P': ('/output/'.join(splitPathFile(__file__))).replace('.py', '_P.pvd')
        }

        showSolution = False

        # Finite Element basis and degree
        Th = 'P2[:], P1'

        # Write a copy for all prints in the below file
        externalTxt = {
            1: open(f'./output/{basename(__file__)}_data.txt'),
            2: None
        }[1]

        # ================
        # Optimal Settings
        # ================
        # Controls variables
        controls = {
            1: ['f', 'ug', 'h'], # Total control
            2: ['ug', 'h'], # Boundary controls
            3: ['f', 'h'], # Mixed Controls (Neumann)
            4: ['f', 'ug'], # Mixed Controls (Dirichlet)
            5: ['h'], # One Control
            6: [] # Without controls
            }[1]

        linSolver = 'tfqmr'
        preconditioner = ['none', 'jacobi'][1]

        # System solve mode
        allAtOnce = True

        # Cost coefficients
        a_s = {
            'z': (1, 1e-1),
            'f': (0, 1e-2),
            'ug': (0, 1e-3),
            'h': (0, 1e-4)
        }

        a_s[_c] = vals
        # a_s = { 
        #     'z': (1, 1e-1),
        #     'f': (1e-5, 1e-2),
        #     'ug': (0, 1e-5),
        #     'h': (0, 1e-4)
        # }

        # Cost measures
        dm = {
            'z': 'dx',
            'f': 'dx',
            'ug': 'ds(inlet)',
            'h': 'ds(outlet)'
        }

        # Descent step size
        rho = 10

        # Previous direction weight
        gamma = {
            1: '1',
            2: '1/8 * norm(dJk)**2 / norm(dJk_1)**2', # The Best
            3: '1/8 * (dJkFunc * (dJkFunc - dJk_1Func)) / norm(dJk_1)**2'
            }[2]

        # ==========
        # Exact Data
        # ==========
        def ud(*x):
            R = 1
            r = x[1]**2 + x[2]**2
            return (1 - r/R**2, 0, 0)


        def pd(*x):
            return 4 - x[0]


        def fd(*x):
            return -_ud.divgrad(*x) + _pd.grad(*x)


        def ugd(*x):
            'Dirichlet value on Γ_D'
            return _ud(*x)


        def hd(*x):
            'Neumann value on Γ_N'
            return (_ud.grad(*x) - _pd(*x)*I) @ normals['outlet']

        # ==========
        # Input Data
        # ==========
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


        def D(u, choice=1):
            if choice == 1:  # [Poiselle]
                output = grad(u)
            elif choice == 2:  # [Turek]
                output = 2*sym(grad(u))
            return output


        # Turn to analytical function object
        _ud = AnalyticalVectorFunction(ud(x, y, z), toccode=True)
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

        # Set the respective boundary measures
        dsInlet = ds(boundaryMark['inlet'])
        dsOutlet = ds(boundaryMark['outlet'])

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

        # Get the normals
        normals = {bound: getNormal(boundMesh)
                    for bound, boundMesh in boundarySubMeshes.items()}

        if invertNormalOrientation:
            # Invert each normal orientation
            normals = {k: v*-1 for k, v in normals.items()}

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
                initialControls = {c: eval(f'{c}0') for c in controls}

        # Group by category the exact data
        exactData = {'ud': ud, 'pd': pd}\
            | {c: eval(f'{c}d') for c in controls}

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
        aState = 'inner(D(z), D(v[0]))*dx + div(z)*q[0]*dx\
            - p*div(v[0])*dx'
        LState = 'dot(f, v[0])*dx + dot(h, v[0])*dsOutlet'
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

        # # Set the preconditioner
        # preconditioner = ['none', 'jacobi'][1]

        # Set the solver
        #solver = setSolver('tfqmr', preconditioner)
        #solver = setSolver('mumps')
        solver = setSolver(linSolver, preconditioner)

        # Set the stabilizator scalar
        beta = Constant(mesh.hmax()**2, name='β')

        # Add the stabilizator term
        aState += beta*dot(grad(p), grad(q[0]))*dx

        if optimal:
            # Add the stabilizator to adjoint equation
            aAdjoint += beta*dot(grad(l_p), grad(q[1]))*dx

            if 'h' in controls and allAtOnce:
                # Add the stabilizator to optimalality condition with respect
                # to 'h'
                aOptimal += beta*inner(grad(h), grad(v[2+_controls.index('h')]))*dx

            elif 'h' in controls:
                # Get the h index
                hIdx = _controls.index('h')

                # Add the stabilizator to optimalality condition with respect
                # to 'h'
                aOptimal[hIdx] += beta*inner(grad(dJ[hIdx]), grad(v[2+hIdx]))*dx

        # Get the trial functions of the collapsed subspaces
        # (for the approach errors calculus)
        u = TrialFunction(Vc)
        p = TrialFunction(Qc)
        c = {c: TrialFunction(VOptC[c]) for c in controls}

        # Set the error formulas
        errors = {
            'u': getErrorFormula(ud, dm['z'], relative=True),
            'p': getErrorFormula(pd, dm['z'], relative=True),
            **{c: getErrorFormula(eval(f'{c}d'), dm[c], relative=True)
                    for c in controls}
        }

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
            ds, boundaryMark,
            getFunctionExpressions(getInputExactData(globals())),
            normals,
            g if 'ug' not in controls else None,
            a_s,
            dm,
        )

        if allAtOnce:
            # Couple the variational formulations
            a, L = system(
                aState + aAdjoint + aOptimal
                - (LState + LAdjoint + LOptimal)
            )

            # Solve the all at once system
            mySolve(a == L, ZPLZLPC, bcs, solver,
                    runtime=True, nnz=W.nnz, dofs=W.dofs,
                    copyTo=externalTxt)

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
                gamma=gamma, errorForm=errors, copyTo=externalTxt
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

        if allAtOnce and optimal:
            # Evaluate and show the cost
            evaluateCost(J, Z, ud, C, a_s, dm, show=True)

            # Calcule approach errors
            errors = evaluateErrors(
                (U, errors['u']),
                (P, errors['p']),
                *[(C[c], errors[c]) for c in controls],
                labels=['u', 'p', *controls],
                relative=type(errors['u']['L²']) is tuple
                )

        elif not optimal:
            # Calcule approach errors
            errors = evaluateErrors(
                (U, errors['u']),
                (P, errors['p']),
                labels=['u', 'p'],
                relative=type(errors['u']['L²']) is tuple
                )

        # Show the approach error
        showError(errors, omg, precision=6, copyTo=externalTxt)

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
    externalTxt.close()
