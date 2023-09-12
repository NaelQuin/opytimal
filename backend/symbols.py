'''
Module of the sympy symbols
'''

from sympy import symbols
from sympy.abc import *

from .string import replaceProgressive
from .types import Union, SpExpression

def toccode(expression: Union[str, SpExpression]) -> SpExpression:
    expression = replaceProgressive(
        str(expression),
        [('z', 'x[2]'), ('y', 'x[1]'), ('x[', '#'),
         ('x', 'x[0]'), ('#', 'x[')],
        preserveds = ['exp']
        )
    return expression
