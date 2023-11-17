'''
Module of the dictionaries proccessment methods
'''

__all__ = ['revertKeysValues', 'getMultipleKeys']

from .arrays import array, vstack, vectorize
from .types import Any, Union, Array

def revertKeysValues(
    d: dict[Any: Any]
        ) -> dict[Any: Any]:

    # Get the keys and values
    keys = list(d.keys())
    values = list(d.values())

    # Replace the keys with values
    dArr = vstack([values, keys], dtype=object).T

    # Get the new dictionary with keys and values changed itself
    dNew = dict(dArr.tolist())

    return dNew


def getMultipleKeys(
    d: dict[Any: Any],
    keys: Union[Array, list, tuple],
    default: Any = None
        ) -> dict:

    try:
        output = vectorize(d.get)(keys, len(keys)*[default])
    except TypeError:
        output = [d.get(k, default) for k in keys]

    return output

