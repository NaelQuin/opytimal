'''
 Decorators script
'''

import functools
import json
import os
import pickle
import time
from typing import Callable

try:
    import psutil

except ModuleNotFoundError:
    pass


# Define python function type
PyFunction = type(lambda:None)

# The profiler call: @alias
def constant(func):
    'Call functions without arguments'
    return func()


# The profiler call: @alias(functionAlias)
def alias(name, globalVars=globals()):
    'Put an alias in the functions'

    def funcAliased(func):
        # Create the alias with the same function definition
        globalVars[name] = func

        return func

    return funcAliased


# The profiler call: @progress(startMessage, endMessage)
def progress(
    start: str,
    end: str = ' Finish',
    breakLine: bool = False,
    breakStart: bool = False,
    argName: str = None
        ) -> (PyFunction):
    'Put an alias in the functions'

    def funcDecorator(func):

        @functools.wraps(func)
        def funcDecorated(*args, **kwargs):
            # Show the start message
            print(
                100*' ' + '\r',
                breakStart*'\n',
                start,
                end=(breakLine*'\n')
                )

            # Call the aim function
            output = func(*args, **kwargs)

            try:
                # Set the progress suffix from "argName" argument
                suffix = f'({kwargs[argName].title()})' \
                    if argName is not None and type(kwargs[argName]) is str \
                    else ''
            except KeyError:
                # Set a default value
                suffix = ''

            # Show the end message
            print(end, suffix, end='\r')

            return output

        return funcDecorated

    return funcDecorator


# The profiler call: @profiler(True, 'both', 'profilerStorePath')
def profiler(*args):
    "Decorator to show profiler to functions"

    global profStoreFile

    # Adjust the input data
    if ('function' in str(type(args[0]))):
        func_global = args[0]
        actv = False
        mode = 'both'
        profStoreFilePath = None

    # Set the attributes
    elif (len(args) == 1):
        actv = args
        mode = 'both'
        profStoreFilePath = None

    elif (len(args) == 2):
        actv, mode = args
        profStoreFilePath = None

    elif (len(args) == 3):
        actv, mode, profStoreFilePath = args

    # Get the profile store dict
    if profStoreFilePath is not None:

        # if 'profStoreFile' not in globals:
        try:
            # Open the file with the profiler dict
            profStoreFile = open(profStoreFilePath, 'br+')

        except FileNotFoundError:
            # Create the file with the profiler dict
            profStoreFile = open(profStoreFilePath, 'bw+')

            # Add an empty dict in file
            pickle.dump(dict(), profStoreFile)

            # Seeking the cursor
            profStoreFile.seek(0)

        # Get the dict from binary file
        profStore = pickle.load(profStoreFile)

    def profilerFunction(*func):
        "Applying the profiler decorator"

        if actv:

            if (mode == 'both'):

                # Retrieving function data
                @functools.wraps(func[0])
                def applyProfiler(*args, **kwargs):
                    "Calculate the memory usage and runtime this function"

                    # Get acess to RAM memory
                    acess = psutil.Process(os.getpid())

                    # Count memory usage before executing the function
                    memoryBefore = acess.memory_info().rss

                    # Start the timer
                    timer = time.time()

                    # Execute the function and show it
                    return_func = func[0](*args, **kwargs)

                    # Stop the timer
                    timer = time.time() - timer

                    # Count memory usage after executing the function
                    memoryAfter = acess.memory_info().rss

                    # Count added memory
                    memoryAdded = memoryAfter - memoryBefore

                    # Get the class name, if it's a method
                    className = getClassName(func[0])

                    # Verify if position is found
                    if className is not None:  # Found
                        # Add the class and method names
                        funcName = f'{className}.{func[0].__name__}'

                    else:
                        # Not found, it's only a function
                        funcName = func[0].__name__

                    # Update/Create the function 'caller counter',
                    # 'runtime' and 'memoryUsage'
                    if (funcName in profStore):
                        profStore[funcName][0] += 1
                        profStore[funcName][1] += timer
                        profStore[funcName][2] += memoryAdded

                    else:
                        profStore[funcName] = [1, timer, memoryAdded]

                    if profStoreFilePath is not None:

                        try:
                            profStoreFile = open(profStoreFilePath, 'br+')
                        except:
                            pass

                        # # Get the dict from binary file
                        profStoreFile.seek(0)
                        oldContent = pickle.load(profStoreFile)

                        # Make the update
                        oldContent.update(profStore)

                        # Overwrite the file
                        profStoreFile.seek(0)  # Reset the cursor
                        profStoreFile.truncate(0)  # Clean the file
                        pickle.dump(
                            oldContent,
                            profStoreFile
                            )  # Add new values

                    else:
                        # Show the profiler
                        print(json.loads(profStore))

                    return return_func

            elif (mode == 'memoryUsage'):

                # Retrieving function data
                @functools.wraps(func[0])
                def applyProfiler(*args, **kwargs):
                    "Calculate the memory usage this function"

                    # Get acess to RAM memory
                    acess = psutil.Process(os.getpid())

                    # Count memory usage before executing the function
                    memoryBefore = acess.memory_info().rss

                    # Execute the function and show it
                    return_func = func[0](*args, **kwargs)

                    # Count memory usage after executing the function
                    memoryAfter = acess.memory_info().rss

                    # Count added memory
                    memoryAdded = memoryAfter - memoryBefore

                    # Get the class name, if it's a method
                    className = getClassName(func[0])

                    # Verify if position is found
                    if className is not None:  # Found
                        # Add the class and method names
                        funcName = f'{className}.{func[0].__name__}'

                    else:
                        # Not found, it's only a function
                        funcName = func[0].__name__

                    # Update/Create the function 'caller counter'
                    # and 'memoryUsage'
                    if (funcName in profStore):
                        profStore[funcName][0] += 1
                        profStore[funcName][1] += memoryAdded

                    else:
                        profStore[funcName] = [1, memoryAdded]

                    if profStoreFilePath is not None:

                        try:
                            profStoreFile = open(profStoreFilePath, 'br+')
                        except:
                            pass

                        # Get the dict from binary file
                        profStoreFile.seek(0)
                        oldContent = pickle.load(profStoreFile)

                        # Make the update
                        oldContent.update(profStore)

                        # Overwrite the file
                        profStoreFile.seek(0)  # Reset the cursor
                        profStoreFile.truncate(0)  # Clean the file
                        pickle.dump(
                            oldContent,
                            profStoreFile
                            )  # Add new values

                    else:
                        # Show the profiler
                        print(json.loads(profStore))

                    return return_func

            elif (mode == 'runtime'):

                # Retrieving function data
                @functools.wraps(func[0])
                def applyProfiler(*args, **kwargs):
                    "Calculate the runtime this function"

                    # Start the timer
                    timer = time.time()

                    # Execute the function and show it
                    return_func = func[0](*args, **kwargs)

                    # Stop the timer
                    timer = time.time() - timer

                    # Get the class name, if it's a method
                    className = getClassName(func[0])

                    # Verify if position is found
                    if className is not None:  # Found
                        # Add the class and method names
                        funcName = f'{className}.{func[0].__name__}'

                    else:
                        # Not found, it's only a function
                        funcName = func[0].__name__

                    # Update/Create the function 'caller counter'
                    # and 'runtime'
                    if (funcName in profStore):
                        profStore[funcName][0] += 1
                        profStore[funcName][1] += timer

                    else:
                        profStore[funcName] = [1, timer]

                    if profStoreFilePath is not None:

                        try:
                            profStoreFile = open(profStoreFilePath, 'br+')
                        except:
                            pass

                        # Get the dict from binary file
                        profStoreFile.seek(0)
                        oldContent = pickle.load(profStoreFile)

                        # Make the update
                        oldContent.update(profStore)

                        # Overwrite the file
                        profStoreFile.seek(0)  # Reset the cursor
                        profStoreFile.truncate(0)  # Clean the file
                        pickle.dump(
                            oldContent,
                            profStoreFile
                            )  # Add new values

                    else:
                        # Show the profiler
                        print(json.loads(profStore))

                    return return_func

        # Don't work, return only function's value
        elif ('function' not in str(type(args[0]))):
            return func[0]

        # Don't work, return only function's value
        else:
            return func_global(*func)

        return applyProfiler

    return profilerFunction


def getClassName(
    method: Callable
        ) -> (str):
    'Get the class name from method caller'

    # Get the method information
    methodInfo = str(method)

    # Verify if method given is associated to any class
    if 'bound method' in methodInfo:
        typeName = 'method'
    elif 'function' in methodInfo:
        typeName = 'function'

    if '.' in methodInfo:
        # Get the position of the type information
        typePos = methodInfo.index(typeName)

        # Get the whitespace after 'method' substring
        whitespacePos = methodInfo.index(' ', typePos)

        # Get the position of the class name suffix
        suffixPos = methodInfo.index('.')

        # Get the class name
        className = methodInfo[whitespacePos+1:suffixPos]

    else:
        className = None

    return className
