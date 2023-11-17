'''
Module of the functions proccessment methods
'''

__all__ = ['getVars']

from .types import Function

def getVars(func: Function) -> (list[str]):
    'Get the function variables'
    return func.__code__.co_varnames[:getArgsCount(func)]


def getArgsCount(func: Function) -> int:
    return func.__code__.co_argcount
