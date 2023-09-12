'''
Module with all setting parameters
'''

__all__ = ['QUADRATURE_DEG', 'luSolverSettings', 'krylovSolverSettings',
           'descentSettings', 'tolerances']

import dolfin as df

df.set_log_active(False)

# Quadrature degree
QUADRATURE_DEG = 5

tolerances = {
    'linearSystem': {
        'absolute': 1e-3,
        'relative': 1e-5
        },

    'nonlinearSystem': {
        'absolute': 1e-4,
        'relative': 1e-8
        }
    }

krylovSolverSettings = {
    'absolute_tolerance': tolerances['linearSystem']['absolute'],
    'relative_tolerance': tolerances['linearSystem']['relative'],
    'maximum_iterations': 10000,
    'monitor_convergence': False,
    'report': False,
    "error_on_nonconvergence": False,
    }

luSolverSettings = {
    'report': False,
    'symmetric': False
    }

descentSettings = {
    'iteractivePlots': False,
    'maxIter': 2000,
    'costTol': 1e-6,
    'costTolRel': 1e-9,
    'rhoTol': 1e-6,
    'stateTolPercent': 1, # % Percentage
    'controlTolPercent': 3, # % Percentage
    'solutionsTolRelation': 'and'
    }

# standardSolver = {
#     'krylov_solver': krylovSolver,
#     'lu_solver': luSolver,
#     'linear_solver': linearSolver,
#     'preconditioner': preconditioner
#     }

# nonlinearSolver = {
#     'nonlinear_solver': 'newton',
#     'newton_solver': {
#         'absolute_tolerance': systemTol["nonlinear"],
#         'relative_tolerance': systemRelTol["nonlinear"],
#         "maximum_iterations": 100,
#         "error_on_nonconvergence": False,
#         'convergence_criterion': 'residual',
#         'report': True
#         } | standardSolver,
#     'snes_solver': {
#         "absolute_tolerance": systemTol["nonlinear"],
#         "relative_tolerance": systemRelTol["nonlinear"],
#         'solution_tolerance': systemTol["nonlinear"],
#         "maximum_iterations": 100,
#         'maximum_residual_evaluations': 2000,
#         "error_on_nonconvergence": False,
#         'report': True,
#         } | standardSolver,
#     'print_matrix': False,
#     'print_rhs': False,
#     }
