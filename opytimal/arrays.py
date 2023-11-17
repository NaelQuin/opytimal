'''
Module of the arrays proccessment methods
'''

__all__ = ['identity', 'split', 'zeros', 'zero', 'complement', 'prod',
           'splitSlice', 'removeEmpty', 'vectorize', 'concatenate',
           'contains', 'vstack', 'flatten']

from copy import copy

import numpy as np
from numpy import hstack, vstack, ravel, arange

from .types import (Array, Union, Tuple, Map, Any, Function)
from .tests import testLoop


def identity(
    dim: int
        ) -> (Array):
    return np.eye(dim)


def split(
    arr: Union[Array, list, tuple],
    stepby: int = 1
        ) -> (Tuple[list[Union[Array, list, tuple]], Union[list, None]]):

    # Init the splitted arr
    splittedArray = []

    # Looping in step times
    for i in range(len(arr)//stepby):
        # Split the array
        split = arr[i*stepby:(i+1)*stepby]

        # Add to splitted list
        splittedArray.append(split)

    # Get the leftover
    leftover = None\
        if len(splittedArray)*stepby == len(arr)\
        else arr[(i+1)*stepby:]

    return splittedArray, leftover


def splitSlice(
    arr: Union[Array, list, tuple, str],
    step: int = 1
        ) -> (list[Union[Array, list, tuple, str]]):

    if type(arr) is not list:
        # Turn to list
        arr = tolist(arr)

    # Init the output list
    splitted = []

    # Set a counter
    k = 0

    while np.size(splitted) < len(arr):
        # Add the respective slice to the output list
        splitted.append(arr[k::step])

        # Update the counter
        k += 1

    return splitted


def array(
    *arr: Array,
    dtype: type = None
        ) -> (Union[Array, Map[Array]]):
    # Turn to numpy array
    output = map(lambda a: np.array(a, dtype=dtype), arr)\
        if len(arr) > 1\
        else np.array(arr[0], dtype=dtype)

    return output


def zero() -> (float):
    return 0


def zeros(
    dim: int
        ) -> (Array):
    return np.zeros(dim)


def getComplement(
    *arrays: Union[Array, list, tuple]
        ) -> (Union[Array, list]):

    if len(arrays) == 1:
        return arrays[0]

    # Get the first array and turn one-by-one to tuple
    arr1 = map(tuple, arrays[0])

    # Turn array to set
    arr1 = set(arr1)

    # Looping in other arrays
    for arr in arrays[1:]:
        # Turn one-by-one to tuple
        arr = map(tuple, arr)

        # Turn respective array to set
        arr = set(arr)

        # Get the complement
        arr1 -= arr

    return arr1


def prod(
    arr: Union[Array, list, tuple]
        ) -> (Any):
    return np.prod(arr)


def tolist(
    arr: Union[Array, list, tuple, str]
        ) -> (list[Any]):
    output = arr.tolist()\
        if type(arr) is Array\
        else list(arr)
    return output


def count(
    arr: Union[Array, list, tuple, str],
    element: Any,
        ) -> (int):

    if element not in arr:
        return 0

    if type(arr) is not np.ndarray:
        # Turn to numpy array
        arr = np.array(
            arr if type(arr) is not str
                else tuple(arr),
            dtype=object
            )

    try:
        # Count the elements
        output = (arr == element).sum()
    except ValueError:
        # Set a default value
        output = None

    return output


def manualCount(
    arr: Union[Array, list, tuple, str],
    element: Any,
        ) -> (int):

    if element not in arr:
        return 0

    if type(arr) is not np.ndarray:
        # Get the element amount
        amount = arr.count(element)

    else:
        # Init a counter
        amount = 0
        # Looping in array values
        for v in arr:
            # Verify the equality
            if v == element:
                # Update the counter
                amount += 1

    return amount


def manualRemove(
    arr: Union[Array, list, tuple, str],
    element: Any,
    occur: int = None,
    inplace: bool = False
        ) -> (Union[Array, list, tuple, str]):

    if not inplace and type(arr) not in (tuple, str):
        # Get an array copy
        arr = copy(arr)

    if occur is None:
        # Get the values amount
        occur = manualCount(arr, element)

    # Store original array
    _arr = arr

    # Turn to list
    arr = list(arr)\
        if type(arr) is not np.ndarray\
        else arr.tolist()

    # Init counters
    idx = count = 0

    # Looping for removal
    while element in arr and count < occur:

        if arr[idx] == element:
            # Remove the current value
            arr = arr[:idx] + arr[idx+1:]

            # Update the counter
            count += 1

        # Update the index
        idx += 1

    if inplace and type(arr) not in (tuple, str):
        # Put values inplace
        _arr[:] = arr

    return arr


def remove(
    arr: Union[Array, list, tuple, str],
    element: Any,
    occur: int = None,
    inplace: bool = False
        ) -> (Union[Array, list, tuple, str]):

    if not inplace and type(arr) not in (tuple, str):
        # Get an array copy
        arr = copy(arr)

    # Get the element amount in arr
    elementAmount = count(arr, element)

    if elementAmount == 0:
        return arr

    elif elementAmount is None:
        return manualRemove(arr, element, occur, inplace)

    elif occur is None:
        # Get the values amount
        occur = elementAmount

    # Get arr type
    arrType = type(arr)

    # Store the original arr
    _arr = arr

    if arrType is not np.ndarray:
        # Turn to array
        arr = np.array(arr)

    if occur is None or occur == elementAmount:
        # Get only the different elements
        arr = arr[arr != element]

    else:
        # Get the occur index
        occurIdx = (arr != element).argsort()[occur-1]

        # Split in two parts
        arrOccur = arr[:occurIdx]
        arr = arr[occurIdx+1:]

        # Join that parts
        arr = np.hstack([arrOccur[arrOccur!=element], arr])

    if arrType is tuple:
        arr = arr(arr)

    elif arrType in [list, str]:
        arr = arr.tolist()

        if arrType is str:
            arr = ''.join(arr)

    if inplace and arrType not in (tuple, str):
        # Put values inplace
        _arr[:] = arr

    return arr


def removeEmpty(
    arr: Union[Array, list, tuple, str],
    type: type = list,
    inplace: bool = False
        ) -> (Union[Array, list, tuple, str]):

    if not inplace:
        # Get a array copy
        arr = arr.copy()

    # Remove the respective empty element
    remove(arr, type(), inplace=inplace)

    return None


def vectorize(
    f: Function
        ) -> (Function):
    return np.vectorize(f)


def concatenate(
    *arrs: Union[Array, list, tuple, str],
    axis: int = 0
        ) -> (Union[Array, list, tuple, str]):

    if len(arrs) == 0:
        raise ValueError('The variable "arrs" not be empty')
    elif len(arrs) == 1 and not hasattr(arrs[0], '__len__'):
        return arrs[0]
    elif len(arrs) == 1 and hasattr(arrs[0], '__len__'):
        # Get the inner list
        arrs = arrs[0]

    # Init the concatenated array
    concatenated = arrs[0]

    # Looping in arrs
    for arr in arrs[1:]:

        if type(arr) is np.ndarray:
            # Apply the stack concatenation
            concatenated = np.stack((concatenated, arr), axis)
        else:
            # Apply the common concatenation
            concatenated += arr

    return concatenated


def contains(
    arr1: Array,
    arr2: Array
        ) -> (Array):
    # Turn to array
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)\
        if hasattr(arr2, '__len__') and type(arr2) is not str\
        else np.array([arr2])

    # To n-dimensional arrays
    if len(arr2.shape) > 1:
        # Init a multidimensional null array
        answer = np.zeros(arr2.shape, dtype=bool)

        # Looping in arr2 rows
        for i, row in enumerate(arr2):
            # Verify if resepctive row values in arr1
            answer[i] = contains(row, arr1)

    # To 1d array with strings
    elif 'str' in arr2.dtype.name:
        # Call the contains function to strings
        answer = containsString(arr1, arr2)

    # To 1d arrays
    else:
        # Verify if arr2 values in arr1
        answer = np.in1d(arr2, arr1)

    return answer


def contain(
    c: str,
    s: str
        ) -> (bool):
    return c in s


def containsString(
    arr1: Array,
    arr2: Array
        ) -> (Array):
    answer = []
    for a2 in arr2:
        answer.append(list(map(contain, len(arr1)*[a2], arr1)))
    if len(answer) == 1:
        answer = answer[0]
    return np.array(answer)


def norm(
    arr: Array,
    type: str = 'l2',
    dx: float = 1e-6
        ) -> (float):

    if type != 'loo':
        normValue = (arr**2).sum()**0.5

        if type == 'h1':
            normValue += (
                (arr[2:] - arr[:-2]) / (2*dx)
            ).sum()**0.5


        elif type == 'h2':
            normValue += (
                (arr[2:]**2 - 2*arr[1:-1] + arr[:-2]**2) / dx**2
            ).sum()**0.5

    else:
        normValue = arr.max()

    return normValue


def getMidPoint(
    arr: Array
        ) -> (float):

    # Calcule the midpoint
    midpoint = (arr.max() + arr.min())/2

    return midpoint


def flatten(
    arr: Union[Array, list, tuple]
        ) -> (Array):
    return np.array(arr).flatten()
