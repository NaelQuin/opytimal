'''Opytimal examples'''

from . import *
import json

_stationaryExamples = '''\
    optPoisson1D
    optPoisson2D
    optPoisson3D
    optStokes2D
    optStokes3D'''

_evolutiveExamples = '''\
    optHeat1D
    optHeat2D'''

allExamples = '''\
Available examples:
    Stationary: 
        {}

    Evolutive:
        {}'''.format(
    _stationaryExamples.replace('\n', '\n'+8*' '),
    _evolutiveExamples.replace('\n', '\n'+8*' ')
)


def examples():
    return print(allExamples)


def stationary_examples():
    stationaryExamples = 'Stationary examples:\n' + _stationaryExamples
    return print(stationaryExamples)


def evolutive_examples():
    evolutiveExamples = 'Evolutive examples:\n' + _evolutiveExamples
    return print(evolutiveExamples)
