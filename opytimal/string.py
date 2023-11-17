'''
Module of the strings proccessment methods
'''

__all__ = ['replaceFormat', 'replaceProgressive', 'replaceAll',
           'functionToStr', 'setFunction', 'splitParcels', 'string',
           'showInfo', 'showProgress', 'eraseProgress', 'convertTime',
           'formatBytes', 'generateId', 'showErrors', 'splitPathFile',
           'convertInt', 'basename']

import os
import re
import sys
import textwrap
import copy

import sympy as sp
import numpy as np
from termcolor import colored
from ufl.algorithms.replace import replace as uflReplace

from .tests import testLoop
from .functions import getVars
from .types import Any, Union, Tuple, Function, File

def extractScalars(
    expr: str,
    vars: list[str] = None,
    simplify: bool = False
        ) -> (Tuple[list[str], str]):
    'Extract scalars from expression'

    if vars is None:
        # Set default value
        vars = []

    # Init the scalars and varsCatched lists
    scalars = []
    varsCatched = []

    if '*' in expr:
        # Split scalar and term
        eSplit = split(expr, '*', 1)

        # Get the scalar and raw term
        scalar = eSplit[0]
        rawTerm = eSplit[1]

        if scalar not in vars:
            # Add respective scalar to list
            scalars.append(scalar)
        else:
            varsCatched.append(scalar)

        if '*' in rawTerm:
            # Aplly recursivity
            scalar, rawTerm = extractScalars(rawTerm, vars)

            # Add to scalars list
            scalars.extend(scalar)

    else:
        # The expression is the term
        rawTerm = expr

    # Looping in var catched as scalars
    for var in varsCatched:
        # Multiply the var to rawTerm end
        rawTerm = f'{rawTerm}*{var}'

    if simplify:
        # Simplify the rawTerm expression
        rawTerm = sp.simplify(rawTerm)

    return scalars, rawTerm


def adjustParenthesisOccurrences(
    parcels: list[str],
    inplace: bool = True
        ) -> (list[str]):

    if not inplace:
        # Get a parcels copy
        pacerls = parcels.copy()

    # Init the index counter
    idx = 0

    # Looping to make the parenthesis adjustment
    while True:

        while parcels[idx].count('(') != parcels[idx].count(')'):
            # Join current parcel to previous parcel
            parcels[idx] = f'{parcels[idx]} + {parcels[idx+1]}'

            # Remove current parcel from parcels
            del(parcels[idx+1])

            # Security condition
            if idx+1 == len(parcels):
                # Stop the loop
                break

        # Update the index
        idx += 1

        if idx == len(parcels):
            # Stop the loop
            break

    return parcels


def splitParcels(expr: str) -> (list[str]):
    # Change minus to plus minus 1
    expr = expr.replace('-','+-1*')

    # Get the parcels
    parcels = expr.split('+')

    # Trailling white spaces from parcels
    parcels = [parcel.strip() for parcel in parcels]

    # Adjust parenthesis occurrences
    adjustParenthesisOccurrences(parcels, inplace=True)

    return parcels


def setFunction(
    funcName: str,
    funcArgs: tuple[str],
    funcReturn: str = '',
    funcCode: str = '',
    funcDoc: str = '',
    funcAnnotations: tuple[str] = None,
    funcDefaults: dict[str: Any] = None,
    removeNullMultiplications: bool = False,
    indent: int = 4,
    **funcConstants: dict[str: Any]
        ) -> (Function):

    # Set the indentation
    indent = indent*' '

    if funcAnnotations is not None:

        if len(funcAnnotations) < len(funcArgs):
            # Adjust annotations size
            funcAnnotations += (len(funcArgs) - len(funcAnnotations))*('Any', )

        # Put the args annotations
        funcArgs = [
            f'{arg}: {annot}' for arg, annot in zip(funcArgs, funcAnnotations)
            ]

    # Join the function args
    funcArgs = ', '.join(funcArgs)

    if funcDefaults is not None:

        for k, v in funcDefaults.items():
            # Default structure
            defaultStruct = f'{k}={v}'

            # Turn respective var to default
            funcArgs = funcArgs.replace(k, defaultStruct)

            # # Add default value in function code
            # funcCode = f'{k}={v}\n{indent if any(funcCode) else ""}{funcCode}'

    # Evaluate the function constants
    funcCode = replaceProgressive(funcCode, funcConstants.items())
    funcReturn = replaceProgressive(funcReturn, funcConstants.items())

    if any(funcCode):
        # Adjust indentation from funcCode
        funcCode = funcCode.replace(indent, '$')\
                           .replace('  ', '')\
                           .replace('\n', f'\n{indent}')\
                           .replace('$', indent)

    if removeNullMultiplications:
        # Remove null multiplications
        funcReturn = str(sp.simplify(funcReturn))

    # Set the pyfunction definition
    funcDef = textwrap.dedent('''
        def {name}({args}):
        {indent}"""{doc}"""
        {code}

        {indent}return {value}
        ''').format(
            name=funcName,
            args=funcArgs,
            indent=indent,
            doc=funcDoc,
            code=funcCode,
            value=funcReturn
            )

    # Execute the func definition
    exec(funcDef, globals(), locals())

    return locals()[funcName]


def functionToStr(
    func: Function
        ) -> (str):
    '''Turn function return to str'''

    # Get the function varibables
    args = getVars(func)

    # Turn args to symbols
    args = sp.symbols(','.join(args))

    # Apply the function on symbols
    funcStr = func(*args)

    return funcStr


def replaceFormat(
    string: str,
    old: str,
    new: str
        ) -> (str):
    'Make replacement in string with regular expression'

    # Replace the marker to chars identifier
    chars = '([A-Za-z0-9\(\)\_\*\+\-\/\[\]\:\,\!\;\@\&\%\'\"\. ]*)'

    # Prepare the old format
    old = replaceProgressive(
        old, [('(', '\('), (')', '\)'),
              ('[', '\['), (']', '\]'),
              ('#', chars)]
        )

    # Prepare the new string
    new = new.replace('#', r'\1')

    # Make the replacement
    string = re.sub(old, new, string)

    return string


def replaceProgressive(
    string: str,
    replacements: list[tuple[str, str]],
    preserveds: list[str] = None
        ) -> (str):

    if preserveds is not None:

        # Looping in replacements
        for repl in replacements:
            # Set a default value
            restoreNeed = False

            # Get the old and new substrings
            old, new = repl

            # Verify the substring to be preserved
            if any([preserved in string for preserved in preserveds]):
                # Make the preservation
                string = preserveSubstring(string, preserveds)

                # Update the "restoreNeed" var
                restoreNeed = True

            if type(string) is str:
                # Turn to string type
                repl = map(str, repl)

                # Set the replacer
                replacer = lambda s, repl: s.replace(*repl)
                # # Make the replacement
                # string = string.replace(*repl)
            else:
                # Set the replacer
                replacer = lambda s, repl: uflReplace(s, dict([repl]))
                # # Make the replacement
                # string = uflReplace(string, dict([repl]))

            # Make the replacement
            string = replacer(string, repl)

            if restoreNeed:
                # Make the restoration
                string = restoreSubstring(string, preserveds)

    else:

        # Looping in replacements
        for repl in replacements:

            if type(string) is str:
                # Turn to string type
                repl = map(str, repl)

                # Set the replacer
                replacer = lambda s, repl: s.replace(*repl)
                # # Make the replacement
                # string = string.replace(*repl)
            else:
                # Set the replacer
                replacer = lambda s, repl: uflReplace(s, dict([repl]))
                # # Make the replacement
                # string = uflReplace(string, dict([repl]))

            # Make the replacement
            string = replacer(string, repl)

    return string


def preserveSubstring(string: str, preserveds: list[str], mark: str = '#'):
    'Protect the "preserved" substring in string'

    if type(preserveds) is not list:
        # Turn to list
        preserveds = [preserveds]

    # Looping in preserved substrings
    for i, preserved in enumerate(preserveds):
        # Make the preservation
        string = string.replace(preserved, f'{mark}{i}')

    return string


def restoreSubstring(string: str, preserveds: str, mark: str = '#'):
    'Restore the "preserved" substring in string'

    if type(preserveds) is not list:
        # Turn to list
        preserveds = [preserveds]

    # Get len of preserveds list
    tam = len(preserveds)-1

    # Looping in preserved substrings
    for i, preserved in enumerate(preserveds[::-1]):
        # Make the restoration
        string = string.replace(f'{mark}{tam - i}', preserved)

    return string


def replaceAll(
    string: str,
    target: Union[str, list[str]],
    replacement: str = ''
        ) -> str:

    if type(target) is str:
        # Turn in list
        target = [target]

    # Apply the replacement
    string = replaceProgressive(
        string,
        zip(target, len(target)*[replacement])
        )

    return string


def split(string: str, sub: str, count: int = None):
    if count is None:
        # Split all occurrences
        output = string.split(sub)
    else:
        # Split any count-th occurrences
        output = string.replace(sub, '#TARGET#', count).split('#TARGET#')
    return output


def adjustId(id):
    return f'_{id}' if id is not None else ''


def generateId(
    label: str,
    globalVars: dict[str: Any] = globals()
        ) -> (str):

    # Adjust label
    label = label.upper().replace(' ', '_')

    # Set a id global var name
    idGlobalVar = f"ID_{label}"

    if idGlobalVar in globalVars:
        globalVars[idGlobalVar] += 1

    else:
        globalVars[idGlobalVar] = 1

    # Get the respective id
    id = globalVars[idGlobalVar]

    return id


def showInfo(
    *infos: str,
    repeatMsg: Union[str, list[str]] = None,
    breakStart: bool = False,
    tab: int = 0,
    err: bool = False,
    color: str = None,
    end: str = '\n',
    delimiters: bool = True,
    alignment: str = None,
    cleanConsole: bool = False,
    startLine: int = 3,
    linesClean: int = '',
    initialCleanConsole: bool = False,
    outputName: str = None,
    copyTo: File = None,
    ignoreEmptyLines: bool = False
        ) -> (None):
    'Report a formated message'

    if cleanConsole:# and type(outputName) is not str:

        if initialCleanConsole:
            # Breaklines
            print(end=initialCleanConsole*'\n')

        # Clear the console
        print(f"\033[{startLine}H\033[{linesClean}J", end='\r')

    if alignment is not None:
        # Get the position of the all alignments char
        positions = list(
            map(lambda s: s.index(alignment)
                    if alignment in s
                    else len(s)//2,
                infos)
            )

        # Get the greater position
        maxPos = max(positions)

        # Calcule the compensation tab
        tabs = list(map(lambda v: (tab + maxPos - v)*' ', positions))

    else:
        # Set a default value
        tabs = len(infos)*[tab*' ']

    # Init the content
    content = ''

    for i, info in enumerate(infos):

        if ignoreEmptyLines and not any(info):
            # Skip the empty lines
            continue

        # Colorize info
        if err:
            info = colored(info, None, 'on_red')
        elif color is not None:
            info = colored(info, 'red', f'on_{color}')

        # Add info to content
        content += f'{tabs[i]}{info}\n'

    # Set a delimiter
    delimit = delimiter(
        max(map(len, infos))+tab, color='red' if err else color
        )

    # Add tab
    delimit = f'{tab*" "}{delimit}'

    if type(outputName) is str and 'outputTxt' not in globals():
        # Open a external file to print
        output = globals()['outputTxt'] = open(outputName, 'w')

    elif outputName is not None and 'outputTxt' in globals():
        # Get the opened external file to print
        output = globals()['outputTxt']

    elif outputName is not None:
        # Get the opened external file to print
        output = outputName

    else:
        # Get default internal output
        output = sys.stdout

    if repeatMsg is not None:
        if type(repeatMsg) not in [list, tuple]:
            repeatMsg = [repeatMsg]

        for msg in repeatMsg:
            # Make the repeat message replacement
            content = content.replace(msg, '#repeatMsg#', 1)\
                             .replace(msg, len(msg)*' ')\
                             .replace('#repeatMsg#', msg)

    if delimiters:
        # Set the print content
        content = [
            (breakStart - 1)*'\n', # Break line in start
            delimit,
            content.strip('\n'),
            delimit
            ]

        # Set the respective separator
        sep = '\n'

    else:
        # Set the print content
        content = [
            breakStart*'\n', # Break line in start
            content.strip('\n')
            ]

        # Set the separator
        sep = ''

    # Show the info
    print(
        *content,
        sep=sep,
        end=end,
        file=output
        )

    if copyTo is not None:
        # Write a copy of the print content
        print(
            *content,
            sep=sep,
            end=end,
            file=copyTo
            )

    if 'outputTxt' in globals() and globals()['outputTxt'].mode=='w':
        globals()['outputTxt'].close()
        globals()['outputTxt'] = open(outputName, 'a')

    return None


def formatBytes(
    bytesValues: int,
    toStr: bool = True
        ) -> (str):
    "Format the 'bytesValues'"

    # Separate in giga, mega, kbytes and byte values
    giga  = int(bytesValues//1e9)
    mega  = int(bytesValues%1e9//1e6)
    kbyte = int(bytesValues%1e9%1e6//1e3)
    byte  = int(bytesValues%1e9%1e6%1e3)

    if toStr:
        # Init the value formated
        formated = ''

        # Format value
        if   (giga  != 0):
            formated += f'{giga},{mega:03} Gb'
            #formated += str(giga) +','+(3-len(str(mega))) *'0'+str(mega)+' Gb'

        elif (mega  != 0):
            formated += f'{mega},{kbyte:03} Mb'
            #formated += str(mega) +','+(3-len(str(kbyte)))*'0'+str(kbyte)+' Mb'

        elif (kbyte != 0):
            formated += f'{kbyte},{byte:03} Kb'
            #formated += str(kbyte)+','+(3-len(str(byte))) *'0'+str(byte)+' Kb'

        else:
            formated += str(byte) + '  b'

        # Set the output
        output = formated.strip()

    else:
        # Set the output
        output = giga

    return output


def convertInt(
    num: Union[int, float],
    precision=3
        ) -> str:
    output = ('{:1.0%df} M'%precision).format(num/1e6)\
        if num > 1e6\
        else (('{:1.0%df} K'%precision).format(num/1e3)
            if num > 1e3
            else ('{}').format(num))
    return output


def convertTime(
    T: int,
    title: str = '',
    prec: int = 3
        ) -> (str):
    """
    Convert the time T into format 'HH:MM:SS.CS'
    """

    # Separate quantities
    hours, minutes = divmod(T, 3600)
    minutes, seconds = divmod(minutes, 60)
    seconds, centsSeconds = divmod(seconds, 1)

    # Convert to int
    hours, minutes, seconds = map(int, [hours, minutes, seconds])

    # Adjust the cents of seconds
    centsSeconds = int(10**prec*round(centsSeconds, prec))

    # Exhibition format
    T_format = ("{:02}:{:02}:{:02}.{:0"+str(prec)+"}").format(
        hours, minutes, seconds, centsSeconds
        )

    return T_format


def delimiter(
    titleTam: Union[str, int],
    symbol: str = '▬',
    tab: int = 0,
    color: str = None
        ) -> (str):
    'Return a string delimiter to text'

    # Set the tam of title
    tam = len(titleTam) if type(titleTam) is str else titleTam

    output = tab*'\t'
    output += colored(tam*symbol, None, f"on_{color}")\
        if color is not None else tam*symbol

    return output

def showProgress(
    message: str,
    k: int,
    total: int = None,
    label: str = None,
    printer: bool = True,
    formatter: str = '',
    markers: str = ["○", '◔', '◑', '◕'],
    suffix: str = ''
        ) -> (str):

    if k == 0:
        # Break a line
        print()

    # Progress structure
    if label is not None:
        structure = '{label} = {k:%s}'%formatter
        kwargs = {'k': k, 'label': label}
    elif total is not None:
        structure = '{k:0%s}/{total:0%s}'%(2*(formatter,))
        kwargs = {'k': k, 'total': total}
    else:
        # Set default values
        structure = ''
        kwargs = {}

    if total is not None:
        v = k/total
        if v < 0.25:
            marker = markers[0]
        elif 0.25 <= v and v < 0.5:
            marker = markers[1]
        elif 0.5 <= v and v < 0.75:
            marker = markers[2]
        elif 0.75 <= v and v <= 1:
            marker = markers[3]
        else:
            marker = markers[3]
    else:
        marker = ''

    # Format the progress
    progress = f'{message}{structure.format(**kwargs)}'

    if marker is not None and any(marker):
        # Add prefixed marker
        progress = f'{marker} ' + progress

    if suffix is not None:
        # Add suffix
        progress += f' {suffix}'

    if printer:
        print(progress, end='\r')

    return progress


def eraseProgress(message: str, printer: bool = True):
    eraser = len(message)*' '

    if printer:
        print(eraser, end='\033[F')

    return eraser


def replaceMathFunctions(
    string: str
        ) -> str:

    newString = replaceProgressive(
        string, [('exp', 'self.exp'),
                 ('log', 'self.log'),
                 ('ln', 'self.ln'),
                 ('sin', 'self.sin'),
                 ('cos', 'self.cos'),
                 ('tan', 'self.tan')]
        )
    return newString


def generateId(
    label: str,
    globalVars: dict[str: Any] = globals()
        ) -> (str):

    # Adjust label
    label = label.upper().replace(' ', '_')

    # Set a id global var name
    idGlobalVar = f"ID_{label}"

    if idGlobalVar in globalVars:
        globalVars[idGlobalVar] += 1

    else:
        globalVars[idGlobalVar] = 1

    # Get the respective id
    id = globalVars[idGlobalVar]

    return id


def showErrors(
    errors: dict[str:float],
    mode: str = 'horizontal', # ['vertical', 'horizontal']
    show: bool = False,
    indent: int = 0,
    preffix: str = ''
        ) -> str:

    # Set the indentation
    indent = indent*' '\
        if mode == 'vertical'\
        else ''

    # Set the preffix
    preffix = ''\
        if mode == 'vertical'\
        else preffix

    # Set the respective separator
    sep = '\n'\
        if mode == 'vertical'\
        else ' | '

    # Set the error structure
    error = len(errors)*(
        '%(indent)s{:8>} = {:1.03e}%(sep)s' % {'indent': indent, 'sep': sep}
        )

    # Add the preffix
    error = f'{preffix}{error}'

    # Fix the last term
    error = error.rstrip(' |')

    # Flatten the erros dict.items
    errorsFlatten = []
    [errorsFlatten.extend(kv) for kv in errors.items()]

    # Fill the error structure
    error = error.format(*errorsFlatten)

    if show:
        # Print the error
        print(error)

    return error


def getSetOperatorsPositions(
    string: str
        ) -> dict[str: str]:

    # Set the operators list
    operators = ['&', '|', '\\']

    # Init the operators positions dict
    operatorsPositions = {}

    # Split string characteres in an array
    chars = np.array(list(string))

    # Get the indexes array
    indexes = np.arange(len(string))

    # Looping in operators
    for op in operators:
        # Get the respective positions
        positions = indexes[chars == op]

        # Set the respective dict
        operatorsPositions.update(
            dict(zip(positions, positions.size*op))
            )

    return operatorsPositions


def getSetOperatorsOrder(
    string: str
        ) -> dict[str: str]:

    # Get the operators positions
    operatorsPositions = getSetOperatorsPositions(string)

    # Order operators by positions
    positions = sorted(operatorsPositions.items(), key=lambda x: x[0])

    # Order the operators
    operatorsOrdered = np.array(positions)[:, 1]

    return operatorsOrdered


def basename(filePath: str) -> (str):
    return os.path.basename(filePath)


def splitPathFile(filePath: str) -> Tuple[str, str]:
    # Get the diretory path
    path = os.path.dirname(filePath)
    # Get the file name
    file = os.path.basename(filePath)

    if os.path.isdir(filePath):
        # Join the "file" to path
        path = '/'.join([path, file])

        # Set a empty file name
        file = ''

    return path, file


def convertInt(
    num: Union[int, float],
    precision=3
        ) -> (str):
    output = ('{:1.0%df} M'%precision).format(num/1e6)\
        if num > 1e6\
        else (('{:1.0%df} K'%precision).format(num/1e3)
            if num > 1e3
            else ('{}').format(num))
    return output
