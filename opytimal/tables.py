'''
Module of the tables proccessment methods
'''

__all__ = ['table']

import functools

import pandas as pd

from .types import Any


@functools.wraps(pd.DataFrame)
def table(*args, **kwargs):
    return pd.DataFrame(*args, **kwargs)


@functools.wraps(pd.set_option)
def setOption(opt: str, value: Any):
    return pd.set_option(opt, value)
