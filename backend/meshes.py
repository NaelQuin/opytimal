'''
Module of the meshes proccessment methods
'''

__all__ = ['readXDMFMesh', 'getInnerNodes', 'getSubMeshes',
           'getSubMeshesComplement', 'getCoordinates', 'getNormal'
           'inBoundary', 'inInletOnly', 'inOutletOnly',
           'getFiniteElementData', 'getBoundaryData']

import os

import dolfin as df
import numpy as np
from mpi4py import MPI

from .profiler import tic, toc
from .tests import testLoop
from .string import (showProgress, eraseProgress, getSetOperatorsOrder,
                     replaceAll, convertInt)
from .types import (Mesh, MeshFunction, Array, MeshFunctionSizet, Tuple, Union,
                    Element)
from .arrays import split


def readXDMFMesh(
    meshFilename: str,
    mfFilename: str = None,
    cfFilename: str = None,
    comm: MPI.Intracomm = None
        ) -> Tuple[
                Mesh,
                MeshFunctionSizet,
                MeshFunctionSizet
                ]:
    # Set the progress message
    _progMsg = 'Reading the "xdmf" file '

    # Show the progress message
    progMsg = showProgress(_progMsg, 0, 3, suffix='mesh')

    # Strip the xdmf extension
    meshFilename = meshFilename.replace('.xdmf', '')
    if mfFilename is not None:
        mfFilename = mfFilename.replace('.xdmf', '')
    if cfFilename is not None:
        cfFilename = cfFilename.replace('.xdmf', '')

    # Init a fenics mesh
    mesh = df.Mesh()\
        if comm is None\
        else df.Mesh(comm)
    mesh.rename('Omega', 'Omega')

    # Erase previous progress message
    eraseProgress(progMsg)

    # Show the progress message
    progMsg = showProgress(_progMsg, 1, 3, suffix='mf')

    # Get the mesh.xdmf file access
    meshFile = df.XDMFFile(f"{meshFilename}.xdmf")\
        if comm is None\
        else df.XDMFFile(comm, f"{meshFilename}.xdmf")

    # Open the mesh file
    with meshFile as infile:
        # Write the file content in mesh
        infile.read(mesh)

    if mfFilename is None:
        # Get the default name to "mf" file
        mfFilename = f'{meshFilename}_mf.xdmf'

        # Verify if the file exists
        if not os.path.exists(f'{meshFilename}_mf.xdmf'):
            # Raise a error
            raise FileNotFoundError(
                f'File "{meshFilename}_mf.xdmf" not found'
                )
    else:
        # Add the file's extension
        mfFilename += '.xdmf'

    # Get the mf.xdmf file access
    mfFile = df.XDMFFile(mfFilename)\
        if comm is None\
        else df.XDMFFile(comm, mfFilename)

    # Get the mesh topology dim
    dim = mesh.topology().dim()

    # Init a value collection elemen
    mvc = df.MeshValueCollection("size_t", mesh, dim-1)

    # Reading the cf file
    with mfFile as infile:
        # Write the file content in
        mfFile.read(mvc)

    # Turn to vector
    mf = df.cpp.mesh.MeshFunctionSizet(mesh, mvc)
    mf.rename('mf', 'mf')

    # Erase previous progress message
    eraseProgress(progMsg)

    # Show the progress message
    progMsg = showProgress(_progMsg, 2, 3, suffix='cf_file')

    if cfFilename is not None:
        # Init a value collection element
        mvc = df.MeshValueCollection("size_t", mesh, dim)

        # Get the cf.xdmf file access
        cfFile = df.XDMFFile(f"{cfFilename}.xdmf")\
            if comm is None\
            else df.XDMFFile(comm, f"{cfFilename}.xdmf")

        # Reading the cf file
        with cfFile as infile:
            # Write the file content in
            infile.read(mvc)

        # Turn to vector
        cf = df.cpp.mesh.MeshFunctionSizet(mesh, mvc, name='cf')
        cf.rename('cf', 'cf')

    else:
        # Set a default value
        cf = None

    # Erase the progress message
    eraseProgress(progMsg)

    return mesh, mf, cf


def getInnerNodes(
    mesh: Mesh,
    bmesh: Mesh = None
        ) -> Array:

    if bmesh is None:
        # Get the respective boundary mesh
        bmesh = df.BoundaryMesh(mesh, 'local', False)

    # Get the inner nodes
    innerNodes = map(tuple, mesh.coordinates())
    innerNodes = set(innerNodes) - set(map(tuple, bmesh.coordinates()))
    innerNodes = np.array(list(innerNodes))

    return innerNodes


def getSubMeshes(
    boundData: MeshFunction,
    boundMarks: dict[str: int]
        ) -> dict[str: Mesh]:
    # Init the submeshes map
    subMeshes = {}

    # Looping in bound name and mark
    for boundName, boundMark in boundMarks.items():
        # Get the respective submesh
        subMeshes[boundName] = df.MeshView.create(
            boundData, boundMark
            )

    return subMeshes


def getMeshFunctionType(data: MeshFunction) -> str:

    # Get the data class name
    mtype = type(data).__name__.replace('MeshFunction', '').lower()

    if mtype == 'sizet':
        # Add underscore
        mtype = 'size_t'

    return mtype


def meshFunctionCopy(data: MeshFunction) -> MeshFunction:
    # Identify the mesh function type
    mtype = getMeshFunctionType(data)

    # Create a new mesh function element
    dataCopy = df.MeshFunction(mtype, data.mesh(), data.dim(), 0)

    # Set the copy content
    dataCopy.array()[:] = data.array()[:]

    return dataCopy


def getSubMeshesComplement(
    boundData: MeshFunction,
    aimMark: int,
    otherMarks: Union[int, tuple[int]]
        ) -> Mesh:

    if type(otherMarks) is int:
        # Put in list
        otherMarks = [otherMarks]

    # Get a bound data copy
    boundData = meshFunctionCopy(boundData)

    # Get the bound data array copy
    _arr = boundData.array()[:]

    # Turn to zero the other marks
    arr = np.where(lambda x: x in tuple(otherMarks), 0, _arr)

    # if (arr == 0).all():
    #     # Turn to zero the other marks (Another way)
    #     arr = eval(
    #         f'np.vectorize(lambda x: x not in {otherMarks})(_arr) * _arr'
    #         )

    # Update the bound data
    boundData.array()[:] = arr

    # Get the respective complement submesh
    submeshComplement = df.MeshView.create(boundData, aimMark)

    return submeshComplement


def getVertexCoordinates(
    *meshes: df.Mesh,
    type: type = tuple
        ) -> list[df.Point]:

    # Init a list
    vertexCoordinates = []

    # Looping in meshes
    for mesh in meshes:
        # Add respective coordinates array to list
        vertexCoordinates.extend(
            list(map(type, mesh.coordinates()))
            )

    return vertexCoordinates


def getCoordinates(
    type: type = tuple,
    **name_meshes: dict[str:df.Mesh]
        ) -> Array:

    # Init the output map
    coordinates = {}

    # Looping in names and meshes
    for name, mesh in name_meshes.items():
        # Get the respective mesh coordinates and
        # turn one-by-one to tuple
        coordinates[name] = list(map(type, mesh.coordinates()))

    return coordinates


def inBoundary(
    x: Array,
    boundaryOperation: str,
    boundMesh: dict[str: Mesh]
        ) -> bool:
    # Get the operators positions
    operatorsOrdered = getSetOperatorsOrder(boundaryOperation)

    # Replace operators by #
    boundaryOperation = replaceAll(
        boundaryOperation, ['&', '\\', '|'], '%s'
        )

    # Get the boundary tags
    boundaryTags = boundaryOperation.split('%s')

    # Turn operators back
    boundaryOperation = boundaryOperation % tuple(operatorsOrdered)

    # Looping in tags
    for tag in boundaryTags:
        # Replace the tag by the respective mesh coordinates
        boundaryOperation = boundaryOperation.replace(
            tag, f'(x in boundMesh["{tag}"].coordinates())'
            )

    # Replace the complement operator to equivalent
    boundaryOperation = boundaryOperation.replace('\\(x in', ' &(x not in')\
                                         .replace('\\ (x in', ' & (x not in')

    return eval(boundaryOperation, {'x': x, 'boundMesh': boundMesh})


def inInletOnly(
    x: Array,
    boundaryMeshes: dict[str: df.Mesh]
        ) -> bool:
    # Verify if x in inlet boundary only
    answer = (x in boundaryMeshes['inlet'].coordinates())\
            & (x not in boundaryMeshes['wall'].coordinates())

    return answer


def inOutletOnly(
    x: Array,
    boundaryMeshes: dict[str: df.Mesh],
    id: int = None
        ) -> bool:
    # Set the outlet boundary name
    outletName = 'outlet'\
        if id is None\
        else f'outlet_{id}'

    # Verify if x in respective outlet boundary only
    answer = (x in boundaryMeshes[outletName].coordinates())\
        & (x not in boundaryMeshes['wall'].coordinates())

    return answer


def getNormal(boundMesh: Mesh) -> Array:
    # Get the first mesh cell
    cell = next(df.cells(boundMesh))

    # Get the normalize
    normal = cell.cell_normal().array()

    return normal


def getBoundaryData(
    S: df.FunctionSpace,
    bcs: list[df.DirichletBC],
    boundData: MeshFunction,
    boundMarks: dict[str: int]
        ) -> (dict[str:str]):

    # Init the dofs dict
    boundDofs = {}

    # Set a test value
    testValue = df.Function(S)

    # Looping in pair (name, mark)
    for bound, mark in boundMarks.items():
        # Calcule the inlet's dofs
        dofs = df.DirichletBC(
            S, testValue, boundData, mark
            ).get_boundary_values().keys()

        # Store the respective dof
        boundDofs[bound] = len(dofs)

    # Init the amount dict
    boundAmount = (len(bcs), len(boundMarks) - len(bcs))

    # Set the inlet mode
    inletMode = 'dirichlet'\
        if boundMarks['inlet'] in [bc.domain_args[1] for bc in bcs]\
        else 'neumann'

    # Join the data
    boundDataDict = {
        'boundDofs': boundDofs,
        'boundAmount': boundAmount,
        'boundMarks': boundMarks,
        'inletMode': inletMode.title()
    }

    return boundDataDict



def setMeasureComplement(
    boundData: MeshFunction,
    i: int
        ) -> (str):
    # Get the boundary data marks
    boundMarks = set(boundData.array())

    # Build the measure complement
    complement = '+'.join(f'ds({j})' for j in (boundMarks - {i}))

    return complement


def getBoundaryArea(
    mesh: df.Mesh,
    ds: df.Measure,
    sub: int = None
        ) -> (float):

    # Set a basice function space
    V = df.FunctionSpace(mesh, "P", 1)

    # Get the geometric dimension of the mesh
    gdim = mesh.geometric_dimension()

    # Get the boundary data
    boundData = ds.subdomain_data()

    # Set a scalar nonnull function
    one = scalarFunction(mesh, 1, 'One')

    # Set the boundary measure complement
    dsComplement = eval(
        setMeasureComplement(boundData, sub)
        )

    # Calcule the boundary measure via complement
    output = df.assemble(one*ds(sub))

    if output == 0:
        # Calcule the boundary measure via complement
        output = 10*df.assemble(
            0.1*one*ds - 0.1*one*dsComplement
            )

    return output


def scalarFunction(
    mesh: Mesh,
    scalar: float,
    name: str = None
        ) -> (df.Function):

    # Set a function space
    V = df.FunctionSpace(mesh, 'P', 1)

    # Set the function
    F = df.Function(V, name=name)

    # Update your values
    F.vector()[:] = scalar

    return F
