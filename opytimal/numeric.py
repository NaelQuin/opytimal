'''
Module for number proccessment methods
'''

__all__ = ['getPow10']

from math import ceil

import numpy as np

def getPow10(
    num: float,
    absolute: bool = True
        ) -> int:
    '''Extract the power of 10 from number'''
    power = int(round(np.log10(num), 0))
    if absolute:
        power = abs(power)
    return power


