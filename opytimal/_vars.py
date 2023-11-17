'''
Module of the functions proccessment methods
'''

__all__ = ['variationalTerms']

variationalTerms = {
    'div(grad(#))': 'inner(dot(grad(#), n), v)*ds - inner(grad(#), grad(v))*dx',
    'grad(#)': 'inner(#, div(v))*dx',
    'div(grad(div(grad(#)))': 'inner(dot(grad(div(grad(#))), n), v)*ds \
        - inner(div(grad(#)), dot(grad(v), n))*ds\
        + inner(div(grad(#)), div(grad(v)))*dx'.replace('\n', '')
                                               .replace(' ', ''),
}

systemTol = {
    'linear': 1e-3,
    'nonlinear': 1e-4
    }

systemRelTol = {
    'linear': 1e-5,
    'nonlinear': 1e-8
    }
