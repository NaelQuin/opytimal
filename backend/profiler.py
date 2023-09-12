'''
Module of the profiler proccessment methods
'''

__all__ = ['setColorbar', 'sca', 'subplot', 'plotComparison',
           'plotMesh', 'addCheckButtons', 'ProgressBar']

import time
import os
import psutil

from .string import (adjustId, formatBytes, convertTime, showInfo,
                     showProgress, eraseProgress)

from .types import Any

class ProgressBar:

    def __init__(
        self: object,
        msg: str,
        label: str = None,
        total: float = None,
        formatter: str = None,
        suffix: str = ''
            ) -> None:

        self.msg = msg
        self.label = label
        self.total = total
        self.formatter = formatter
        self.value = 0
        self.suffix = suffix

        return None

    def __str__(self):

        # Print the current progress
        self.__progressMsg = showProgress(
            self.msg, self.value, total=self.total, label=self.label,
            formatter=self.formatter, suffix=self.suffix
            )
        return self.__progressMsg

    def start(self):
        if self.value > 0:
            return None
        print(self, end='\r')
        return None

    def update(self, value: float, suffix: str = None):
        self.value = value
        if suffix is not None:
            self.suffix = suffix
        print(self, end='\r')
        return None

    def erase(self):
        eraseProgress(self.__progressMsg)
        return None


def tic(
    globals=globals(),
    id: str = None
        ) -> None:
    'Start a timer'

    # Adjust id
    id = adjustId(id)

    # Update the respective runtimer
    globals.update({f'RUNTIME{id}': time.time()})

    return None


def toc(
    globalVars: dict[str: Any] = globals(),
    printer: bool = True,
    id: str = None,
    convert: bool = False
        ) -> (int):
    'Stop the timer'

    # Adjust id
    id = adjustId(id)

    # Get the global var
    RUNTIME = globalVars[f'RUNTIME{id}']

    # Calcule the runtime
    RUNTIME = time.time() - RUNTIME

    if printer:
        # Mount the message to show
        message = f'Runtime: {convertTime(RUNTIME)}'

        # Show the message
        showInfo(message, breakStart=True)

    # Delete the respective global var
    globalVars.pop(f'RUNTIME{id}')

    # Set the output
    output = RUNTIME\
        if not convert\
        else convertTime(RUNTIME)

    return output


def getMemory(pid: int) -> (int):
    'Get memory in bytes from pid'

    # Open the process
    process = psutil.Process(
        os.getpid()
        )

    # Get the current memory usage
    RAM = process.memory_info().rss

    return RAM


def ticRAM(
    globalVars: dict[str: Any] = globals(),
    id: str = None
        ) -> (None):
    # Adjust id
    id = adjustId(id)

    # Get the current memory RAM usage
    currentMemory = getMemory(os.getpid())

    # Update the respective runtimer
    globalVars.update({
        f'MEMORY{id}': currentMemory
        })

    return None


def tocRAM(
    globalVars: dict[str: Any] = globals(),
    printer: bool = True,
    id: str = None,
    convert: bool = False
        ) -> (int):
    'Stop the timer'

    # Adjust id
    id = adjustId(id)

    # Get the global var
    MEMORY = globalVars[f'MEMORY{id}']

    # Calcule the runtime
    MEMORY = getMemory(os.getpid()) - MEMORY

    if printer:
        # Mount the message to show
        message = f'RAM Usage: {formatBytes(MEMORY)}'

        # Show the message
        showInfo(message, breakStart=True)

    # Delete the respective global var
    globalVars.pop(f'MEMORY{id}')

    # Set the output
    output = MEMORY\
        if not convert\
        else formatBytes(MEMORY)

    return output


