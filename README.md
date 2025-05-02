# Opytimal - Python/FEniCS framework to solve optimal control PDE-based problems

[![PyPI version](https://badge.fury.io/py/opytimal.svg)](https://badge.fury.io/py/opytimal)

## 📖 Overview

Opytimal is a Python/FEniCS framework with the main goal of solving Optimal Control problems considering multiple and mixed controls based to linear and nonlinear PDEs, in addition to solving PDEs with mixed boundary conditions simply and clearly.

## 🔬 Publications

- Quintino, N., & Tiago, J. (2025). Opytimal - A Python/FEniCS framework to solve PDE-based optimal control problems considering multiple controls in 2D and 3D domains. Research Square. [https://doi.org/10.21203/rs.3.rs-6546784/v1](https://doi.org/10.21203/rs.3.rs-6546784/v1)

## 🚀 Installation

```bash
pip install opytimal
```

## 🔧 Quick Example

```python
from opytimal import getAdjointSystem, getOptimalConditionsSystem

# Read mesh data
meshPath = "cylinder.xdmf"  # The Mesh file in XDMF format
boundaryDataPath = "cylinder_mf.xdmf"  # The Boundary Data in XDMF format
mesh, boundaryData, _ = readXDMFMesh(meshPath, boundaryDataPath)
boundaryMark = {  # Labeling the boundary marks
    'inlet': 1,
    'outlet': 2,
    'wall': 3
}

# Set the cost function on Opytimal
a_s = {  # Cost parcel constants
    'z': (10, 1e-1),
    'f': (0, 1e-2),
    'ug': (0, 1e-3),
    'h': (0, 1e-4)
}
dm = {  # Cost parcel measures
    'z': 'dx',
    'f': 'dx',
    'ug': 'ds["inlet"]',
    'h': 'ds["outlet"]'
}

def J(a, u, dm):
    return 0.5*(a[0]*u**2 + a[1]*grad(u)**2)*dm

def gradJ(a, u, dm, v):
    return (a[0]*u*v + a[1]*inner(grad(u), grad(v)))*dm

# Define the trial and test functions on standard Fenics
z, l_z, f, ug, h = TrialFunctions(W)
v = TestFunctions(W)

# Set the variational formulations on Opytimal
aState = '''
    dot(grad(z), grad(v[0]))*dx
'''
LState = '''
    dot(f, v[0])*dx + dot(h, v[0])*ds["outlet"] 
        - dot(grad(ug), grad(v[0]))*dx
'''

# Calculate cost gradients
J_z = gradJ(
    a_s["z"], z+ug-ud, dm["z"], v[1]
)
J_f = gradJ(
    a_s["f"], f, dm["f"], v[2]
)
J_ug = gradJ(
    a_s["ug"], ug, dm["ug"], v[3]
)
J_ug += gradJ(
    a_s["z"], ug, dm["z"], v[3]
)
J_ug_zud = gradJ(
    a_s["z"], z-ud, dm["z"], v[3]
)
J_h = gradJ(
    a_s["h"], h, dm["h"], v[4]
)

# Calculate the adjoint equation
aAdjoint, LAdjoint = getAdjointSystem(
    aState,                           # State equation left side
    [('z', 'v[1]'), ('v[0]', 'l_z')], # Test function replacements
    J_z,                              # Adjoint equation right side
    globalVars=globals()              # Global variables dictionary
)

# Get the optimality conditions variational system
aOptimal, LOptimal = getOptimalConditionsSystem(
    ["f", "ug", "h"],     # Controls variables
    LState,               # State equation right side
    [('v[0]', 'l_z')],    # Test function substitutions
    [J_f, J_ug, J_h],     # Cost gradient in control variables
    J_ug_zud,             # What's left of the lifting gradient
    globalVars=globals()  # Global variables 
)

# # Or setting manually the adjoint equation and optimality conditions
# aAdjoint = fenicsVarForm
# LAdjoint = fenicsVarForm
# aOptimal = fenicsVarForm
# LOptimal = fenicsVarForm

aState = eval(aState)  # Evaluating the strings
LState = eval(LState) 
# aAdjoint = eval(aAdjoint) 
# LAdjoint = eval(LAdjoint) 
# aOptimal = eval(aOptimal) 
# LOptimal = eval(LOptimal) 

a = aState + aAdjoint + aOptimal  # Variational system left side
L = LState + LAdjoint + LOptimal  # right side

# Define the boundary conditions
bcsState = [
    DirichletBC(Vh, 0, boundaryData, boundaryMark['inlet']),
    DirichletBC(Vh, gW, boundaryData, boundaryMark['wall'])
]  # Set the state boundary conditions

bcsAdj = [
    DirichletBC(Vadjh, 0, boundaryData, boundaryMark['inlet']),
    DirichletBC(Vadjh, 0, boundaryData, boundaryMark['wall'])
]  # Set the adjoint boundary conditions

bcs = bcsState + bcsAdj  # Join the dirichlet boundary conditions

# Define the solution function space and the solution vectors
ZLC = Function(W)  # Initialising the system solution
solve(a == L, ZLC, bcs, "tfqmr", "jacobi")  # Solve system
Z, L, F, Ug, H = ZLC.split(deepcopy=True)  # Split solution
U = Function(Z.ufl_function_space(), name="U")  # Initialising the state solution
U.vector().set_local(
    Z.vector().get_local() + Ug.vector().get_local()  # Fill state solution with Z + Ug
)
```

## 🧪 Demos

In the folder `opytimal/demos`, we find examples of the Opytimal application to optimal problems based on Poisson, Stokes, and Heat equations.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the project's style guidelines and passes all tests.

## 👥 Authors

- **Natanael Quintino** - *Initial work* - [NaelQuin](https://github.com/NaelQuin)

## 🙏 Acknowledgments

- Based on open-source platform [FEniCS](https://fenicsproject.org/)
