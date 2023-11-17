'''
Module of the parallel proccessment functions
'''

__all__ = ['parallel', 'activeCount']

from threading import Thread, active_count
from multiprocessing import Process


def getFuncResult(func, *args, output, **kwargs):
    'Get the result of the respective function executed in a thread'
    output.append(func(*args, **kwargs))
    return None


def parallel(func, *args, **kwargs):
    'Execute in a parallel proccess the respective function'
    output = []
    thrd = Thread(
        target=getFuncResult,
        args=(func, *args),
        kwargs={'output': output} | kwargs
        )
    thrd.start()
    return thrd, output


def parallelMP(func, *args, **kwargs):
    'Execute in a parallel proccess the respective function'
    thrd = Process(
        target=func,
        args=args,
        kwargs=kwargs
        )
    thrd.start()
    return thrd


def activeCount():
    return active_count()
