'''
Module for analytical proccessment methods
'''

__all__ = []

import numpy as np
import sympy as sp

from .profiler import tic, toc
from .tests import testLoop
from .mathFunctions import sin, cos, tan, exp, log, ln, pi
from .string import replaceProgressive, textwrap
from .parallel import parallelMP
from .types import (Union, Tuple, Function, Number, Iterable,
                    AnalyticalFunction)


def womersly(
    mu: float,
    terms: int = 20,
    omega: float = 3*sp.pi,
    string: bool = False
        ) -> (Union[Tuple[Function], Tuple[str]]):
    '''
    mu: float
    --
        Fluid viscosity

    terms: int = 20
    -----
        Number of womersly series term

    omega: float = 3*pi
    -----
        Senoid period in radians

    string: bool = False
    ------
        Return womersly expression in string format
    '''
    def gamma(k):
        # Set symbols
        pi = sp.pi
        exp = sp.exp
        sin = sp.sin

        numerator = textwrap.dedent('''({omega})*(
            exp(-{c}*pi**2*t) - cos({omega}*t)\
            ) + {c}*pi**2*sin({omega}*t)''').format(
                omega=omega, c=mu*(2*k+1)**2
                )

        denominator = '({c1}*pi**4 + {c2})*({c3})*pi'.format(
            c1=mu**2*(2*k+1)**4,
            c2=omega**2,
            c3=2*k+1
            )

        return f'{-8*mu}*({numerator})/({denominator})'

    # Summing the n-th series terms
    serieTerms = [
        f'({gamma(2*k+1)})*sin(({2*k+1})*pi*x[1])'
            for k in range(terms)
        ]

    # Set the velocity expression
    velocity = '+'.join(serieTerms)

    # Set the pressure expression
    pressure = f"-{2*mu}*sin({omega}*t)"

    if not string:
        # Set the velocity function
        exec(
            f'def velocityFunc(*x, t=0): return {velocity}',
            globals(),
            locals()
            )

        # Set the pressure function
        exec(
            f'def pressureFunc(*x, t=0): return {pressure}',
            globals(),
            locals()
            )

        # Get the python functions
        velocity = locals()['velocityFunc']
        pressure = locals()['pressureFunc']

    return velocity, pressure


class AnalyticalFunction:
    "Class to define a analytic function and your derivatives"

    # Name of all methods and atributes
    __all__ = [
        'definition', 'showDerivatives', 'changeDefinition', 'changeVariables',
        'copy', 'getDerivativeDefinitions', 'setAnalyticTransform'
        ]

    def __init__(
        self: object,
        def_string: str = 'None',
        vars: str = 'xyz',
        toccode: bool = False,
        parallel: bool = False
            ) -> (None):
        """
        Builder of object
        """

        # Set the attribute parallel
        self._parallel = parallel

        # Definition of function
        self.definition = str(
            sp.sympify(def_string)
            )

        # The variable this function
        self.vars = vars

        if (def_string != 'None'):
            # Process to calculate the derivatives and 
            # create the respective python functions
            self.__calculateDerivatives()

            if not self._parallel:
                # Set ccode symbols
                x, y, z, t = sp.symbols('x[0], x[1], x[2], t')\
                    if toccode\
                    else sp.symbols(','.join(self.vars + 't'))

                # Set the gradient and laplacian definitions
                self._grad = np.array(
                    [eval(self._dx), eval(self._dy), eval(self._dz)]
                    )
                self._div = sp.simplify(
                    eval(self._dx) + eval(self._dy) + eval(self._dz)
                    )
                self._divgrad = sp.simplify(
                    eval(self._dxx) + eval(self._dyy) + eval(self._dzz)
                    )
                self._bidivgrad = sp.simplify(
                    eval(self._dxxxx) + eval(self._dyyyy)
                    + eval(self._dzzzz) + eval(self._dxxyy)
                    + eval(self._dyyxx) + eval(self._dxxzz)
                    + eval(self._dzzxx) + eval(self._dyyzz)
                    + eval(self._dzzyy)
                    )
            else:
                # Set default value to gradient
                self._grad = self._div = self._divgrad = self._bidivgrad = None

            if toccode:
                # Turn all string definitions to c code notations
                self.toccode()

        else:
            # Set default value to gradient
            self._grad = self._div = self._divgrad = self._bidivgrad = None

        return None

    def __str__(
        self: object
            ) -> (str):
        """
        Result of command 'print(self)'

        Show the definitions of the function and your derivatives
        """

        # Init the list of the nulls derivatives
        derivativesNulls = []

        # Delimiter of exhibition and identation
        delimiter = 80*'─'
        tab = 4*' '

        # Init the deriviatives definitions
        derivativesDef = {
            'time': '',
            'space': '',
            'mixed': ''
            }

        # Looping in all derivatives
        for ind, attr in enumerate(self._derivativesNames):
            # Get the derivative
            der = self._derivativesNames_[ind]  # [2:]

            # Get the respective derivate definition
            definition = eval(f'self.{attr}')

            # Verify if this derivative is null (Store and go to next)
            if (definition == '0'):
                derivativesNulls.append(der)
                continue

            # # Separate to specific variables
            # if (der in self.vars):
            #     message += '\n'

            # Add the derivative definition to respective category
            if ('t' in der) \
                    and (any([var for var in self.vars if var in der])):
                derivativesDef['mixed'] += tab + f"{der:9}: {definition}\n"

            elif ('t' in der):
                derivativesDef['time'] += tab + f"{der:9}: {definition}\n"

            else:
                derivativesDef['space'] += tab + f"{der:9}: {definition}\n"

        # Get the specifics derivatives
        timeDer = derivativesDef['time'][:-1]
        spaceDer = derivativesDef['space'][:-1]
        mixDer = derivativesDef['mixed'][:-1]

        # Mount the properties text
        propertiesStr = f'''
        #{delimiter}
        #    Definitions of the function and your derivatives
        #{delimiter}

        #> Definition:

        #    {self.definition}

        #{'> Non-null derivatives in time:' if any(timeDer) else '*'}

        #{timeDer if any(timeDer) else '*'}

        #{'> Non-null derivatives in space-time:' if any(mixDer) else '*'}

        #{mixDer if any(mixDer) else '*'}

        #{'> Non-null derivatives in space:' if any(spaceDer) else '*'}

        #{spaceDer if any(spaceDer) else '*'}
        #'''

        # propertiesStr += f'''
        # #> Null derivatives:

        # #    {self._derivativesNullStructure(derivativesNulls)}
        # ''' if len(derivativesNulls) > 0 else ''

        propertiesStr += f'''
        #{delimiter}
        '''

        # Fix the indentation
        propertiesStr = propertiesStr.replace(8*' '+'#', '')

        # Remove excessives break lines
        propertiesStr = propertiesStr.replace('\n*\n', '')

        return propertiesStr

    def __newer__(
        self: object,
        newDefinition: str,
        vars: str = 'xyz',
        initialConditions: bool = False
            ) -> (AnalyticalFunction):
        "Define a new object with the new definition given"

        # Building the new object
        new_object = AnalyticalFunction(
            newDefinition,
            vars,
            initialConditions
            )

        return new_object

    def __neg__(
        self: object
            ) -> (AnalyticalFunction):
        "Negation of this object"

        # Multiply the definition by -1
        newDefinition = '-('+self.definition+')'

        # Apply an algebric simplification
        newDefinition = str(sp.simplify(newDefinition))

        # Generate a new object with the algebric sum of the definitions
        result = self.__newer__(newDefinition)

        return result

    def __add__(
        self: object,
        other: AnalyticalFunction
            ) -> (AnalyticalFunction):
        "Arithimetic operation of '+' between objects same type"

        if type(other) is not AnalyticalFunction:
            return None

        # Calculate the algebric sum of the definitions
        newDefinition = f'{self.definition}+{other.definition}'

        # Apply an algebric simplification
        newDefinition = str(sp.simplify(newDefinition))

        # Generate a new object with the algebric sum of the definitions
        result = self.__newer__(newDefinition)

        return result

    def __radd__(
        self: object,
        other: AnalyticalFunction
            ) -> (AnalyticalFunction):
        "Arithimetic operation of '+' between objects same type"

        # Calculate the algebric sum of the definitions
        newDefinition = other.definition+'+'+self.definition

        # Apply an algebric simplification
        newDefinition = str(sp.simplify(newDefinition))

        # Generate a new object with the algebric sum of the definitions
        result = self.__newer__(newDefinition)

        return result

    def __sub__(
        self: object,
        other: AnalyticalFunction
            ) -> (AnalyticalFunction):
        "Arithimetic operation of '-' between objects same type"

        if type(other) is not AnalyticalFunction:
            return None

        # Calculate the algebric sum of the definitions
        newDefinition = f'{self.definition}-({other.definition})'

        # Apply an algebric simplification
        newDefinition = str(sp.simplify(newDefinition))

        # Generate a new object with the algebric sum of the definitions
        result = self.__newer__(newDefinition)

        return result

    def __rsub__(
        self: object,
        other: AnalyticalFunction
            ) -> (AnalyticalFunction):
        "Arithimetic operation of '-' between objects same type"

        if type(other) is not AnalyticalFunction:
            return None

        # Calculate the algebric sum of the definitions
        newDefinition = other.definition+'-('+self.definition+")"

        # Apply an algebric simplification
        newDefinition = str(sp.simplify(newDefinition))

        # Generate a new object with the algebric sum of the definitions
        result = self.__newer__(newDefinition)

        return result

    def __pow__(
        self: object,
        other: "Number/AnalyticalFunction"
            ) -> (AnalyticalFunction):
        """
        Arithimetic operation of '**' between objects same type
        or multiply to scalar
        """

        # Set the new definition
        if type(other) in [int, float]:
            newDefinition = "(" + self.definition + ")**(" \
                             + str(other) + ')'

        else:
            newDefinition = "(" + self.definition + ")**(" \
                             + str(other.definition) + ')'

        # Apply an algebric simplification
        newDefinition = str(sp.simplify(newDefinition))

        # Generate a new object with the algebric sum of the definitions
        result = self.__newer__(newDefinition)

        return result

    def __rmul__(
        self: object,
        other: "Number/AnalyticalFunction"
            ) -> (AnalyticalFunction):
        """
        Arithimetic operation of '*' between objects same type
        or multiply to scalar
        """

        if type(other) not in [int, float, complex]:
            return None

        # Set the new definition
        if type(other) in [int, float]:
            newDefinition = str(other)\
                             + '*('+self.definition+')'\

        else:
            newDefinition = '('+other.definition+')*('\
                             + self.definition + ')'

        # Apply an algebric simplification
        newDefinition = str(sp.simplify(newDefinition))

        # Generate a new object with the algebric sum of the definitions
        result = self.__newer__(newDefinition)

        return result

    def __mul__(
        self: object,
        other: "Number/AnalyticalFunction"
            ) -> (AnalyticalFunction):
        """
        Arithimetic operation of '*' between objects same type
        or multiply to scalar
        """

        # Set the new definition
        if type(other) in [int, float]:
            newDefinition = '('+self.definition+')*'\
                             + str(other)
        else:
            newDefinition = '('+self.definition+')*('\
                             + other.definition + ')'

        # Apply an algebric simplification
        newDefinition = str(sp.simplify(newDefinition))

        # Generate a new object with the algebric sum of the definitions
        result = self.__newer__(newDefinition)

        return result

    def __call__(
        self: object,
        x: Number = None,
        y: Number = None,
        z: Number = None,
        t: Number = None
            ) -> (Number):
        """
        Calling of object
        """

        # Define the output
        if x is not None:
            output = self._u(x, y, z, t)

        else:
            output = self.definition

        return output

    def __checkVars(
        self: object,
        vars: str
            ) -> (str):
        'Get the variables contained in the definition'

        # Verifying if vars given, they are in function definition
        varsChecked = ''.join(
            set(vars).intersection(self.definition)
            )

        return varsChecked

    def __createDerivativeAttribute(
        self: object,
        order1: int,
        order2: int,
        derivativesNames: list,
        derivativesNames_: list
            ) -> (list, list):
        '''Create the attribute associated to derivative
           d{var1}{order1}{var2}{order2}'''
        # Importing it locally
        from .mathFunctions import sin, cos, tan, exp, log, ln

        # Get the definition of the exact function
        definition = self.definition

        # Get the variables
        var_s = self.vars + 't'
        x, y, z, t = var_s
        var_s_mixed = [
            (x, y), (x, z), (y, z), (x, t), (y, t), (z, t)
            ]

        # Title for python functions
        titleFunc = 'lambda {}, {}=None, {}=None, {}=None, mfuncs=globals(): (locals().update(globals()), %s)[1]'.format(
            *var_s
            )

        # Function null
        funcNull = "lambda {}, {}=None, {}=None, {}=None: 0*{}".format(
            *var_s, var_s[0]
            )

        # Dictionary to identify the var numbering
        varNumbering = {
            x: 1,
            y: 2,
            z: 3,
            t: 4
            }

        # Calcule the derivatives
        if (order1 == 1):
            for ind, var in enumerate(var_s):
                # Attribute path and name
                attr = 'self._d{}'.format(
                    order2*var
                    )

                # Store this derivative name
                derivativesNames .append(attr[5:])
                derivativesNames_.append(attr[6:])

                # Create the respective attribute
                exec(
                    "{} = str(sp.diff('{}','{}','{}'))".format(
                        attr, definition, var, order2
                        ),
                    globals(),
                    locals()
                    )

                # Evaluate the attr
                attrEval = eval(attr)

                # Construct the python function
                if (attrEval == '0'):
                    # The null functions
                    exec(
                        "self.d{} = eval('{}')".format(
                            attr[7:],
                            funcNull
                            ),
                        globals(),
                        locals()
                        )

                else:
                    # Define the functions not nulls
                    exec(
                        "self.d{} = {}".format(
                            attr[7:],
                            titleFunc % attrEval
                            ),
                        globals(),
                        locals()
                        )

                    # Save the derivative by the position of the variable
                    exec(
                        "self.d{} = {}".format(
                            order2*f'Var{varNumbering[var]}',
                            titleFunc % attrEval
                            ),
                        globals(),
                        locals()
                        )

        # Calcule the mixed derivatives
        for ind, var in enumerate(var_s_mixed):
            # Attribute path and name
            attr = 'self._d{}{}'.format(
                order1*var[0], order2*var[1]
                )

            attrComute = 'self._d{}{}'.format(
                order2*var[1], order1*var[0]
                )

            # Store this derivatives names
            derivativesNames .extend([attr[5:], attrComute[5:]])
            derivativesNames_.extend([attr[6:], attrComute[6:]])

            # Create the respectives attributes
            exec(
                "{} = str(sp.diff('{}','{}',{},'{}',{}))".format(
                    attr,
                    definition,
                    var[0],
                    order1,
                    var[1],
                    order2
                    ),
                globals(),
                locals()
                )

            exec(
                "{} = str(sp.diff('{}','{}',{},'{}',{}))".format(
                    attrComute,
                    definition,
                    var[1],
                    order2,
                    var[0],
                    order1
                    ),
                globals(),
                locals()
                )

            # Evaluate the attr and attrComute
            attrEval = eval(attr)
            attrComuteEval = eval(attrComute)

            # Construct the python function
            if attrEval == '0':
                exec(
                    "self.d{} = eval('{}')".format(
                        attr[7:],
                        funcNull
                        ),
                    globals(),
                    locals()
                    )

            else:
                exec(
                    "self.d{} = {}".format(
                        attr[7:],
                        titleFunc % attrEval
                        ),
                    globals(),
                    locals()
                    )

                # Save the derivative by the position of the variable
                exec(
                    "self.d{}{} = {}".format(
                        order1*f'Var{varNumbering[var[0]]}',
                        order2*f'Var{varNumbering[var[1]]}',
                        titleFunc % attrEval
                        ),
                    globals(),
                    locals()
                    )

            if attrComuteEval == '0':
                exec(
                    "self.d{} = eval('{}')".format(
                        attrComute[7:],
                        funcNull
                        ),
                    globals(),
                    locals()
                    )

            else:
                exec(
                    "self.d{} = {}".format(
                        attrComute[7:],
                        titleFunc % attrComuteEval
                        ),
                    globals(),
                    locals()
                    )

                # Save the derivative by the position of the variable
                exec(
                    "self.d{}{} = {}".format(
                        order2*f'Var{varNumbering[var[1]]}',
                        order1*f'Var{varNumbering[var[0]]}',
                        titleFunc % attrComuteEval
                        ),
                    globals(),
                    locals()
                    )

        return None

    def __calculateDerivatives(
        self: object
            ) -> (None):
        '''
        Calculate the definition of derivatives of function given
        and create the respectives python functions
        '''

        # Get the definition of the exact function
        definition = self.definition

        # Get the variables
        var_s = self.vars + 't'

        # Init a list to store the derivatives
        derivativesNames, derivativesNames_ = [], []

        # Title for python functions
        titleFunc = 'lambda {}, {}=0, {}=0, {}=0, mfuncs=globals(): (locals().update(globals()), %s)[1]'.format(
            *var_s
            )

        # Create a function with the definition given (to __call__)
        self._u = eval(
            titleFunc % f'{definition} + 0*{var_s[0]}',
            globals(), locals()
            )


        if not self._parallel:
            # Double looping in derivatives order
            for ord1 in [1, 2, 3, 4]:
                for ord2 in [1, 2, 3, 4]:
                    self.__createDerivativeAttribute(
                        ord1,
                        ord2,
                        derivativesNames,
                        derivativesNames_
                        )

        else:
            thrds = [[
                parallelMP(
                    self.__createDerivativeAttribute,
                    ord1, ord2, derivativesNames, derivativesNames_
                    ) for ord1 in [1,2,3,4]]
                for ord2 in [1,2,3,4]
                ]

            # Stop all threads
            [thrd.join() for thrd in np.ravel(thrds)]

        # Create the attribute (private)
        self._derivativesNames = sorted(derivativesNames)
        self._derivativesNames_ = sorted(derivativesNames_)

        return None

    def __changeVars(
        self: object,
        newVars: str,
        definition: str = None,
        vars: str = None
            ) -> (str):
        "Change the variables in definition"

        # Get the old variables
        if vars is None:
            x, y, z = self.vars
        else:
            x, y, z = vars

        # Get the newVars
        x_, y_, z_ = newVars

        # Preserve the 'exp' functions
        if definition is None:
            newDefinition = self.definition.replace('exp', 'e#p')

        else:
            newDefinition = definition.replace('exp', 'e#p')

        # Make the change
        newDefinition = newDefinition.replace(z, z_)\
                                     .replace(y, y_)\
                                     .replace(x, x_)

        # Restore the 'exp' functions
        newDefinition = newDefinition.replace('e#p', 'exp')

        return newDefinition

    def __derivativesNullStructure(
        self: object,
        derivativesList: 'list[str]'
            ) -> (str):
        'Mount the message to inform the derivative that are nulls'

        # Set the indentation
        tab = 4*' '

        # Init the message with the formatter
        message = '\n' + tab + len(derivativesList)*'{:9} = '

        # Number of the derivatives per line
        derivatives_per_line = 5

        # Put one break line limiting 50 characters per line
        message = message.replace(
            derivatives_per_line*'= {:9} ',
            tab + 10*' ' + derivatives_per_line*'= {:9} ' + '= 0\n'
        )

        # Fix the first line
        message = message.replace('{:9}' + 14*' ', '{:9}')

        # Fix the last line
        message = message.replace('\n= {:9}', '\n' + tab + 10*' ' + '= {:9}')

        # Add the '0' in last line
        message += '0\n'

        # Add the derivatives names
        message = message.format(*derivativesList)

        return message

    def toccode(self: object):
        for der in self._derivativesNames + ['definition']:
            exec(
                f"self.{der} = replaceProgressive(\
                    self.{der},\
                    [('z', 'x[2]'), ('y', 'x[1]'), ('x[', '#'),\
                    ('x', 'x[0]'), ('#', 'x[')],\
                    preserveds = ['exp']\
                    )",
                {'replaceProgressive': replaceProgressive},
                locals()
                )
        return None

    def grad(self, *x, t=None, dim=None):
        return np.array([self.dx(*x, t=t), self.dy(*x, t=t), self.dz(*x, t=t)])[:dim]

    def div(self, *x, t=None):
        return self.dx(*x, t=t) + self.dy(*x, t=t) + self.dz(*x, t=t)

    def divgrad(self, *x, t=None):
        return self.dxx(*x, t=t) + self.dyy(*x, t=t) + self.dzz(*x, t=t)

    def bidivgrad(self, *x, t=None):
        return self.dxxxx(*x, t=t) + self.dyyyy(*x, t=t) + self.dzzzz(*x, t=t)\
            + self.dxxyy(*x, t=t) + self.dyyxx(*x, t=t) + self.dxxzz(*x, t=t)\
            + self.dzzxx(*x, t=t) + self.dyyzz(*x, t=t)+ self.dzzyy(*x, t=t)

    def showDerivatives(
        self: object
            ) -> (None):
        'Return the all derivatives definitions'

        # Show the same message in function __str__
        print(self.__str__())

        return None

    def changeDefinition(
        self: object,
        newDefinition: str
            ) -> (None):
        "Change the definition and recalcule the derivatives"

        # Reconstruct this object with the new definition
        self.__init__(newDefinition, self.vars)

        return None

    def changeVariables(
        self: object,
        newVariables: str,
        newElement: bool = False
            ) -> (AnalyticalFunction):
        "Change the variables and recalcule the derivatives"

        # Change the variable in definition
        newDefinition = self.__changeVars(newVariables)

        # Create a new element
        if newElement:
            output = self.__newer__(newDefinition, newVariables)

        # Reconstruct this object with the new definition
        else:
            output = self.__init__(newDefinition, newVariables)

        return output

    def copy(
        self: object
            ) -> (str):
        "Return a copy of this object"

        # Create a new object with the same definition this
        object_copy = self.__newer__(self.definition)

        return object_copy

    def integral(
        self,
        u,
        t,
        vector=False,
        dimension=1,
        ps_detJ=None,
        tauGaussPts=None,
        factorAdjust=None,
        indGlb=None,
        funcGaussPts=None
            ) -> (float):
        'Calcule the integral of vector u given'

        # Turn to list (if didn't)
        if (type(tauGaussPts) != list):
            tauGaussPts = [tauGaussPts]

        if (vector) or ('function' not in str(type(u))):
            # The values of vector u by nodes of mesh add boundary values
            vect_gaussPts = np.append(
                u, [0, 0]*dimension*factorAdjust
                )[indGlb]

            # Linear combination: (u * basis finite element respective)
            vect_gaussPts = vect_gaussPts.dot(funcGaussPts)

            # Set the output
            output = vect_gaussPts

        else:
            # Evaluate the function in tranformation of gauss points
            # in Omega_e, for all e
            func_gaussPts = u(*tauGaussPts, t=t)

            # Set the output
            output = func_gaussPts

        # Turn value in row vector
        output = output.flatten()

        # Multiplying by gauss weight and jacobian
        output *= ps_detJ

        # Summing
        output = output.sum()

        return output

    def getDerivativeDefinitions(
        self: object,
        derivative: str
            ) -> (str):
        'Get the respective derivative definition'

        if not any(derivative):
            definition = self.definition

        else:
            try:
                # Get the respective definition
                definition = eval(f'self._{derivative}')

            except AttributeError:
                raise

        return definition

    def __getJacobianByDerivatives(
        self: object,
        derivatives: 'Iterable[str]',
        newVarLabel: str
            ) -> (Iterable):
        'Get the inverse jacobian associated to each derivatives given'

        # Get the new vars
        exec(
            f'{"y,z,w"[:2*len(newVarLabel)-1]} = "{newVarLabel}"',
            )

        # Adjust the derivatives value
        if type(derivatives) is str:
            derivatives = [derivatives]

        # Init the jacobians list
        jacobians = []

        # Looping in each derivatives
        for der in derivatives:
            # Calcule the jacobians
            jacobian = self.__analyticJacobian(
                sp.symbols(f'{newVarLabel}')
                )

            # Add to list
            jacobians.append(jacobian)

        return jacobians

    def __analyticTransformPdeTerm(
        self: object,
        term: str,
        solLabel: str = 'u',
        newSol: str = 'v',
        varLabel: str = 'x',
        newVar: str = 'y'
            ) -> (str):
        'Apply the variable transform in a pde term'

        # Get the derivative in term
        derivatives = getDerivativesFromPDE(term, solLabel)

        # Init the new terms list
        newTerms = []

        # Looping in derivatives
        for der in derivatives:
            # Apply the change vars in this derivatives
            newDerivative = der.replace(varLabel, newVar)

            # Calculate the respective inverse jacobians
            jacobian = self.__getJacobianByDerivatives(
                newDerivative,
                newVar
                )[0]

            # Calculate the respective new term
            newTerms.append(f'{jacobian}*{newSol}{newDerivative}')

        # Adjust if have one only term
        if len(newTerms) == 1:
            newTerms = newTerms[0]

        return newTerms

    def analyticTransform(
        self: object,
        termVar: [int, str]
            ) -> (str):
        """Apply the analytic transform in the term pde or
            var given"""

        if type(termVar) is str and self.__solutionLabel in termVar:
            output = self.__analyticTransformPdeTerm(
                termVar,
                self.__solutionLabel,
                self.__newSolLabel,
                self.__variableLabel,
                self.__newVarLabel
                )

        else:
            output = self.__analyticTransformVar(termVar)

        return output

    def setAnalyticTransform(
        self: object,
        transformDefinition: str,
        varLabel: str = 'x',
        newVar: str = 'y',
        solLabel: str = 'u',
        newSol: str = 'v'
            ) -> (None):
        'Set the analytic transform definition'

        # Set the indentation to structure definition
        indent = 4*' '

        # Set the new variable if don't given
        if varLabel in ['x', 'y', 'z', 't']:
            # Set from standard changes
            newVar = {
                'x': 'y',
                'y': 'z',
                'z': 'w',
                't': 't'
                }[varLabel]
        else:
            # Add emphasis
            newVar = varLabel + '_'

        # Set the new variable if don't given
        if newSol in ['u', 'v', 'w']:
            # Set from standard changes
            newSol = {
                'u': 'v',
                'v': 'w'
                }[solLabel]

        else:
            # Add emphasis
            newSol = solLabel + '_'

        # Mount the structure of the transform var function definition
        transformVarDefinition = f'''\
            #def analyticTransformVar({varLabel}):
            #    if type({varLabel}) is str:
            #        {varLabel} = sp.symbols('{varLabel}')
            #
            #    return {transformDefinition}'''.replace(3*indent+'#', '')

        # Mount the variable transform function
        exec(transformVarDefinition, None, locals())

        # Callcule the jacobian definition
        jacobianDefinition = sp.diff(transformDefinition, varLabel)

        # Mount the structure of the jacobian function definition
        jacobianDefinition = f'def analyticJacobian({varLabel}): ' \
            + f'return {jacobianDefinition}'

        # Mount the variable transform jacobian function
        exec(jacobianDefinition, None, locals())

        # Create the respectives attributes
        self.__analyticJacobian = locals()['analyticJacobian']
        self.__analyticTransformVar = locals()['analyticTransformVar']

        # Fixes the labels
        self.__solutionLabel = solLabel
        self.__newSolLabel = newSol
        self.__variableLabel = varLabel
        self.__newVarLabel = newVar

        # Add to methods list
        self.__all__.append('analyticTransform')

        return None

    # @classmethod
    # def sin(self, x): return np.sin(x)

    # @classmethod
    # def cos(self, x): return np.cos(x)

    # @classmethod
    # def exp(self, x): return np.exp(x)

    # @classmethod
    # def log(self, x): return np.log(x)

    # @classmethod
    # def tan(self, x): return np.tan(x)

    # @property
    # def pi(self): return np.pi

    # @property
    # def e(self): return np.e


class AnalyticalVectorFunction(AnalyticalFunction):
    "Class to define a analytic function and your derivatives"

    # Name of all methods and atributes
    __all__ = [
        'definition', 'showDerivatives', 'changeDefinition', 'changeVariables',
        'copy', 'getDerivativeDefinitions', 'setAnalyticTransform'
        ]

    def __init__(
        self: object,
        def_string: list[str] = ['None'],
        vars: str = 'xyz',
        toccode: bool = False,
        parallel: bool = False
            ) -> (None):
        """
        Builder of object
        """

        # Set the attribute parallel
        self._parallel = parallel

        # Generate analytical subfunctions
        self._subFunctions = [
            AnalyticalFunction(s, vars, toccode=False)
                for s in def_string
            ]

        # Definition of function
        self.definition = tuple(
            func.definition
                for func in self._subFunctions
            )

        # The variable this function
        self.vars = vars

        if (def_string != 'None'):
            # Process to calculate the derivatives and 
            # create the respective python functions
            self.__calculateDerivatives()

            if not parallel:
                # Set ccode symbols
                x, y, z, t = sp.symbols('x[0], x[1], x[2], t')\
                    if toccode\
                    else sp.symbols(','.join(self.vars + 't'))

                # Init the respective definitions list
                self._grad = []
                self._div = []
                self._divgrad = []
                self._bidivgrad = []

                # Set the gradient definition
                for func in self._subFunctions:
                    grad = [
                        eval(func._dx),
                        eval(func._dy),
                        eval(func._dz)
                        ]

                    div = eval(func._dx)\
                        + eval(func._dy)\
                        + eval(func._dz)

                    divgrad = eval(func._dxx)\
                        + eval(func._dyy)\
                        + eval(func._dzz)

                    bidivgrad = eval(func._dxxxx) + eval(func._dyyyy)\
                    + eval(func._dzzzz) + eval(func._dxxyy)\
                    + eval(func._dyyxx) + eval(func._dxxzz)\
                    + eval(func._dzzxx) + eval(func._dyyzz)\
                    + eval(func._dzzyy)

                    # Set the gradient and laplacian definitions
                    self._grad.append(np.array(grad))
                    self._div.append(sp.simplify(div))
                    self._divgrad.append(sp.simplify(divgrad))
                    self._bidivgrad.append(sp.simplify(bidivgrad))
            else:
                # Set default value to gradient
                self._grad = self._div = self._divgrad = self._bidivgrad = None

            if toccode:
                # Turn all string definitions to c code notations
                [func.toccode() for func in self._subFunctions]

                # Definition of function
                self.definition = tuple(
                    func.definition
                        for func in self._subFunctions
                    )

        else:
            # Set default value to gradient
            self._grad = self._div = self._divgrad = self._bidivgrad = None

        return None

    def __str__(
        self: object
            ) -> (str):
        """
        Result of command 'print(self)'

        Show the definitions of the function and your derivatives
        """

        # Init the list of the nulls derivatives
        derivativesNulls = []

        # Delimiter of exhibition and identation
        delimiter = 80*'─'
        tab = 4*' '

        # Init the deriviatives definitions
        derivativesDef = {
            'time': '',
            'space': '',
            'mixed': ''
            }

        # Looping in all derivatives
        for ind, attr in enumerate(self._derivativesNames):
            # Get the derivative
            der = self._derivativesNames_[ind]  # [2:]

            # Get the respective derivate definition
            definition = eval(
                f'tuple(func.{attr} for func in self._subFunctions)'
                )

            # Verify if this derivative is null (Store and go to next)
            if (np.array(definition) == '0').all():
                derivativesNulls.append(der)
                continue

            # Add the derivative definition to respective category
            if ('t' in der) \
                    and (any([var for var in self.vars if var in der])):
                derivativesDef['mixed'] += tab + f"{der:9}: ({(len(self.definition)*'%s, ').strip(', ')})\n" % definition

            elif ('t' in der):
                derivativesDef['time'] += tab + f"{der:9}: ({(len(self.definition)*'%s, ').strip(', ')})\n" % definition

            else:
                derivativesDef['space'] += tab + f"{der:9}: ({(len(self.definition)*'%s, ').strip(', ')})\n" % definition

        # Get the specifics derivatives
        timeDer = derivativesDef['time'][:-1]
        spaceDer = derivativesDef['space'][:-1]
        mixDer = derivativesDef['mixed'][:-1]

        # Mount the properties text
        propertiesStr = f'''
        #{delimiter}
        #    Definitions of the function and your derivatives
        #{delimiter}

        #> Definition:

        #    ({(len(self.definition)*'%s, ').strip(', ')})

        #{'> Non-null derivatives in time:' if any(timeDer) else '*'}

        #{timeDer if any(timeDer) else '*'}

        #{'> Non-null derivatives in space-time:' if any(mixDer) else '*'}

        #{mixDer if any(mixDer) else '*'}

        #{'> Non-null derivatives in space:' if any(spaceDer) else '*'}

        #{spaceDer if any(spaceDer) else '*'}
        #''' % tuple(d for d in self.definition)

        propertiesStr += f'''
        #{delimiter}
        '''

        # Fix the indentation
        propertiesStr = propertiesStr.replace(8*' '+'#', '')

        # Remove excessives break lines
        propertiesStr = propertiesStr.replace('\n*\n', '')

        return propertiesStr

    def __call__(
        self: object,
        x: Number = None,
        y: Number = None,
        z: Number = None,
        t: Number = None
            ) -> (Number):
        """
        Calling of object
        """

        # Define the output
        if x is not None:
            output = self._u(x, y, z, t)

        else:
            output = self.definition

        return output

    def __calculateDerivatives(
        self: object
            ) -> (None):
        '''
        Calculate the definition of derivatives of function given
        and create the respectives python functions
        '''

        # Get the definition of the exact function
        definition = self.definition

        # Get the variables
        var_s = self.vars + 't'

        # Init a list to store the derivatives
        derivativesNames, derivativesNames_ = [], []

        # Title for python functions
        titleFunc = 'lambda {}, {}=0, {}=0, {}=0, mfuncs=globals(): (locals().update(globals()), %s)[1]'.format(
            *var_s
            ) % f"np.array([{'%s, '*len(definition)}])"

        # Vectorize definitions
        definition = tuple(
            f'{d} + 0*{var_s[0]}' for d in definition
            )

        # Create a function with the definition given (to __call__)
        self._u = eval(
            titleFunc % definition,
            globals(), locals()
            )

        if not self._parallel:
            # Double looping in derivatives order
            for ord1 in [1, 2, 3, 4]:
                for ord2 in [1, 2, 3, 4]:
                    self.__createDerivativeAttribute(
                        ord1,
                        ord2,
                        derivativesNames,
                        derivativesNames_
                        )

        else:
            thrds = [[
                parallelMP(
                    self.__createDerivativeAttribute,
                    ord1, ord2, derivativesNames, derivativesNames_
                    ) for ord1 in [1,2,3,4]]
                for ord2 in [1,2,3,4]
                ]

            # Stop all threads
            [thrd.join() for thrd in np.ravel(thrds)]

        # Create the attribute (private)
        self._derivativesNames = sorted(derivativesNames)
        self._derivativesNames_ = sorted(derivativesNames_)

        return None

    def __createDerivativeAttribute(
        self: object,
        order1: int,
        order2: int,
        derivativesNames: list,
        derivativesNames_: list
            ) -> (list, list):
        '''Create the attribute associated to derivative
           d{var1}{order1}{var2}{order2}'''
        # Importing it locally
        from .mathFunctions import sin, cos, tan, exp, log, ln

        # Get the definition of the exact function
        definition = self.definition

        # Get the variables
        var_s = self.vars + 't'
        x, y, z, t = var_s
        var_s_mixed = [
            (x, y), (x, z), (y, z), (x, t), (y, t), (z, t)
            ]

        # Title for python functions
        titleFunc = 'lambda {}, {}=None, {}=None, {}=None, mfuncs=globals(): (locals().update(globals()), %s)[1]'.format(
            *var_s
            ) % f"np.array([{'%s, '*len(definition)}])"

        # Function null
        funcNull = "lambda {}, {}=None, {}=None, {}=None: \
            np.array({}*[0*{}])".format(
            *var_s, len(definition), var_s[0]
            )

        # Dictionary to identify the var numbering
        varNumbering = {
            x: 1,
            y: 2,
            z: 3,
            t: 4
            }

        # Calcule the derivatives
        if (order1 == 1):
            for ind, var in enumerate(var_s):
                # Attribute path and name
                attr = 'self._d{}'.format(
                    order2*var
                    )

                # Store this derivative name
                derivativesNames .append(attr[5:])
                derivativesNames_.append(attr[6:])

                # Create the respective attribute
                exec(
                    "{} = tuple(\
                        str(sp.diff(d,'{}','{}'))\
                            for d in {}\
                        )".format(attr, var, order2, definition),
                    globals(),
                    locals()
                    )

                # Evaluate the attr
                attrEval = eval(attr)

                # Construct the python function
                if (np.array(attrEval) == '0').all():
                    # The null functions
                    exec(
                        "self.d{} = eval('{}')".format(
                            attr[7:],
                            funcNull
                            ),
                        globals(),
                        locals()
                        )

                else:
                    # Define the functions not nulls
                    exec(
                        "self.d{} = {}".format(
                            attr[7:],
                            titleFunc % attrEval
                            ),
                        globals(),
                        locals()
                        )

                    # Save the derivative by the position of the variable
                    exec(
                        "self.d{} = {}".format(
                            order2*f'Var{varNumbering[var]}',
                            titleFunc % attrEval
                            ),
                        globals(),
                        locals()
                        )

        # Calcule the mixed derivatives
        for ind, var in enumerate(var_s_mixed):
            # Attribute path and name
            attr = 'self._d{}{}'.format(
                order1*var[0], order2*var[1]
                )

            attrComute = 'self._d{}{}'.format(
                order2*var[1], order1*var[0]
                )

            # Store this derivatives names
            derivativesNames .extend([attr[5:], attrComute[5:]])
            derivativesNames_.extend([attr[6:], attrComute[6:]])

            # Create the respectives attributes
            exec(
                "{} = tuple(\
                    str(sp.diff(d,'{}',{},'{}',{}))\
                        for d in {}\
                    )".format(attr, var[0], order1,
                              var[1], order2, definition),
                globals(),
                locals()
                )

            exec(
                "{} = tuple(\
                    str(sp.diff(d,'{}',{},'{}',{}))\
                        for d in {}\
                    )".format(attrComute, var[1], order2,
                              var[0], order1, definition
                    ),
                globals(),
                locals()
                )

            # Evaluate the attr and attrComute
            attrEval = eval(attr)
            attrComuteEval = eval(attrComute)

            # Construct the python function
            if (np.array(attrEval) == '0').all():
                exec(
                    "self.d{} = eval('{}')".format(
                        attr[7:],
                        funcNull
                        ),
                    globals(),
                    locals()
                    )

            else:
                exec(
                    "self.d{} = {}".format(
                        attr[7:],
                        titleFunc % attrEval
                        ),
                    globals(),
                    locals()
                    )

                # Save the derivative by the position of the variable
                exec(
                    "self.d{}{} = {}".format(
                        order1*f'Var{varNumbering[var[0]]}',
                        order2*f'Var{varNumbering[var[1]]}',
                        titleFunc % attrEval
                        ),
                    globals(),
                    locals()
                    )

            if (np.array(attrComuteEval) == '0').all():
                exec(
                    "self.d{} = eval('{}')".format(
                        attrComute[7:],
                        funcNull
                        ),
                    globals(),
                    locals()
                    )

            else:
                exec(
                    "self.d{} = {}".format(
                        attrComute[7:],
                        titleFunc % attrComuteEval
                        ),
                    globals(),
                    locals()
                    )

                # Save the derivative by the position of the variable
                exec(
                    "self.d{}{} = {}".format(
                        order2*f'Var{varNumbering[var[1]]}',
                        order1*f'Var{varNumbering[var[0]]}',
                        titleFunc % attrComuteEval
                        ),
                    globals(),
                    locals()
                    )

        return None

    def grad(self, *x, t=None, dim=None):
        return np.array([func.grad(*x, t=t, dim=dim)
                            for func in self._subFunctions])

    def div(self, *x, t=None):
        return np.array([func.div(*x, t=t)
                            for func in self._subFunctions])

    def divgrad(self, *x, t=None):
        return np.array([func.divgrad(*x, t=t)
                            for func in self._subFunctions])

    def bidivgrad(self, *x, t=None):
        return np.array([func.bidivgrad(*x, t=t)
                            for func in self._subFunctions])

# Test section
if (__name__ == '__main__'):
    import sys

    # Init the progress to show
    print(
        '\n(    ) Building the instance', end='\r'
        )

    PARALLEL = sys.argv[1] if len(sys.argv) > 1 else True

    u = AnalyticalFunction(
        'sin(w)*sin(t)**2*cos(y*z)',
        'yzw',
        progress=True,
        parallel=PARALLEL
        )

    print(u)

    # u.setAnalyticTransform('1/(1+t)', varLabel='t')

    # print(
    #     '\n',
    #     u.analyticTransform(0),
    #     u.analyticTransform('u_xx'),
    #     sep=', '
    #     )
