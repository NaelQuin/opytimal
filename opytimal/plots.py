'''
Module of the plots proccessment methods
'''

__all__ = ['setColorbar', 'sca', 'subplot', 'plotComparison',
           'plotMesh', 'addCheckButtons', 'dynamicComparison']

import time
import tkinter as tk

import matplotlib
import easygui
import tikzplotlib
import matplotlib.pyplot as plt
import dolfin as df
import numpy as np
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.widgets import CheckButtons
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .parallel import parallel
from .types import (Union, Tuple, Any, Function, Axe, HostAxes, Colorbar,
                    Figure, Plot, Array, Color, Collection, Mesh)
from .string import generateId, showProgress
from .numeric import ceil
from .tests import testLoop
from .arrays import splitSlice, zeros, removeEmpty, array, getMidPoint
from .files import dirname, basename
from .modules import isInstalled

SET_AXES_ASPECT = False

try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('WebAgg')

def is3d(ax: Axe) -> (bool):
    return '3D' in type(ax).__name__


def colorbarPositioner(
    ax: Axe,
    size: float = 1,
    position: str = 'lower right'
        ) -> (HostAxes):

    # Adjust colorbar size and positioning him
    if not is3d(ax):
        # Get the axes location
        divider = make_axes_locatable(ax)

        # Adjust the position string
        position = position.replace('center', '')\
                           .strip()

        # Set the properties
        cax = divider.append_axes(
            position,
            size=f"{size*100}%",
            pad=0.1
            )

    else:
        # Fix positions wrong to 3D axes
        positionsConverter = {
            'bottom': 'lower center',
            'top': 'upper center',
        }

        # Fix the position
        position = positionsConverter.get(position, position)

        cax = inset_axes(
            ax,
            width="5%",
            height="50%",
            loc=position,
            bbox_to_anchor=(0.125, 0.04, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=-2,
            )

    # Set colorbar axe label
    cax.set_label('<colorbar>')

    return cax


def getColorLimits(*collections) -> Tuple[float, float]:

    # if all(hasattr(col, 'get_clim') for col in collections):
    #     clim = array(
    #         [col.get_clim()
    #             for col in collections if hasattr(col, 'get_clim')]
    #         )
    #     clim = clim[:, 0].min(), clim[:, 1].max()
    #     if clim != (None, None):
    #         return clim

    # Set initial values to limits
    vmin = np.inf
    vmax = -np.inf

    # Looping in axes collections
    for col in collections:
        # Get the collections array colors
        colorsArray = getColorValues(col)

        if colorsArray is not None:
            # Get the colorbar limits
            vmin = min(colorsArray.min(), vmin)
            vmax = max(colorsArray.max(), vmax)

    # Set the output vales limits
    output = (vmin, vmax)\
        if vmin != np.inf and vmax != -np.inf\
        else None

    return output


def getColoredCollections(
    *axes: Axe,
    withoutEmpty: bool = False,
    turn2row: bool = True
        ) -> list[Collection]:
    # Init a collections list
    collections = []

    # Looping in axes
    for ax in axes:
        # Add the respective collections to list
        collections.extend([
            col for col in ax.collections
                if 'Line' not in type(col).__name__
            ])

    if withoutEmpty:
        # Remove empty list
        removeEmpty(collections, inplace=True)

    if turn2row:
        # Turn to row list
        collections = np.ravel(collections)

    return collections


def getCollections(
    *axes: Axe,
    withoutEmpty: bool = False,
    turn2row: bool = True
        ) -> (list[Plot]):
    # Init a collections list
    collections = []

    # Looping in axes
    for ax in axes:
        # Add the respective collections to list
        collections.extend([col for col in ax.collections])

    if withoutEmpty:
        # Remove empty list
        removeEmpty(collections, inplace=True)

    if turn2row:
        # Turn to row list
        collections = np.ravel(collections)

    return collections


def normalizeColors(
    *collections: Collection,
    vlim: Tuple[float, float],
    clip: bool = True
        ) -> (None):

    if type(collections[0]) is Figure:
        # Get all figures axes
        axes = [fig.axes for fig in collections]

        # Turn to row array
        axes = np.ravel(axes)

        # Get the all axes collections
        collections = getColoredCollections(*axes)

    # Set the normalizator
    norm = matplotlib.colors.Normalize(*vlim, clip=clip)

    # Looping in axes
    for col in collections:
        # Get the axe array
        array = getColorValues(col)

        if array is not None:
            # Normalize colors
            col.set_array()

    return None


def setColorbar(
    ax: Axe,
    label: str = None,
    scale: float = 1,
    tickfont: int = 10,
    position: str = 'bottom',
    cmap: Color = matplotlib.cm.viridis,
    vlim: Tuple[float, float] = None,
    formatter: Union[str, Function] = '{x:1.02e}',
        ) -> (Colorbar):

    # Get all figure axes colored collections
    coloredCollections = getColoredCollections(
        ax,
        #*ax.figure.axes,
        withoutEmpty=True
        )

    if not any(coloredCollections):
        # This axe haven't colors as legend
        return None

    if vlim is None:
        # Get the values limits
        vlim = getColorLimits(*coloredCollections)

    # Calcule the color range
    vRange = vlim[1] - vlim[0]\
        if vlim is not None\
        else None

    # Set the colorbar location
    if 'upper' in position or position == 'top':
        location = 'top'
    elif 'lower' in position or position == 'bottom':
        location = 'bottom'
    elif 'left' in position:
        location = 'left'
    elif 'right' in position:
        location = 'right'

    # Set the colorbar positioner
    cax = colorbarPositioner(ax, scale, position)

    # Plot the colorbar
    cbar = ax.figure.colorbar(
        ScalarMappable(
            norm=matplotlib.colors.Normalize(*vlim, clip=False),
            cmap=cmap
            ),
        ax=ax, cax=cax, location=location,
        ticks=np.linspace(*vlim, 6)
            if vRange is not None
            else None
        )

    # Adjust the font size of the colorbar
    cbar.ax.tick_params(labelsize=tickfont)

    if formatter is not None:
        # Format the ticks labels
        cbar.ax.yaxis.set_major_formatter(formatter)

    # Set the colorbar label
    cbar.set_label(
        '$\mathbf{%s}$' % label.strip('$')
            if label is not None
            else ''
        )

    # Adjust the colorbar label
    cbar.ax.set_ylabel(
        cbar.ax.get_ylabel(),
        rotation='horizontal',
        labelpad=10
        )

    # Set the colorbar edge color
    cbar.outline.set_edgecolor('black')

    # Set the tick labels color
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

    return cbar


def sca(figLabel: str, axId: int = 0):
    'Set the axId-th axe of the respective figure as current'

    # Set the respective axe
    plt.sca(
        plt.figure(figLabel).axes[axId]
        )

    return None


def subplot(
    rows: int = 1,
    cols: int = 1,
    sharex: bool = True,
    sharey: bool = True,
    label: str = None,
    projection: str = None,
    triDim: bool = False,
        ) -> (Tuple[plt.Figure, plt.Axes]):
    'Generate a figure subplot'

    if type(projection) is not list:
        # Adjust the projection shape
        projection = array(
            rows*[cols*[projection]]
            )

    # Get the respective figure or create one
    fig = plt.figure(label)

    # Set the width and height ratios
    widthRatio = (cols - cols//2, cols//2)\
        if cols > 1\
        else (1,)
    heightRatio = (rows - rows//2, rows//2)\
        if rows > 1\
        else (1,)

    # Create a gridspec of size rows x cols
    gs = fig.add_gridspec(
        rows, cols, width_ratios=widthRatio, height_ratios=heightRatio,
        left=0.1, right=0.9, bottom=0.1, top=0.9,
        wspace=0.05, hspace=0.05
        )

    # Set the axes structure matrix
    axes = array(rows*[cols*[0.]], dtype=object)

    # Looping in columns
    for j in range(cols):
        # Looping in rows
        for i in range(rows):
            # Associate the respective grid position to a new axe
            ax = fig.add_subplot(
                gs[i, j], projection=projection[i, j],
                # **({'sharex': axes[0, j]} if i > 0 else {}),
                # **({'sharey': axes[i, 0]} if j > 0 else {}),
                )

            if sharex and i < rows-1 and rows > 1 and not triDim:
                # Share x axis with the first in respective column
                ax.set_xticks([])

            if sharey and j < cols-1 and cols > 1 and not triDim:
                # Share y axis with the first in respective row
                ax.set_yticks([])

            # Add to respective position in axes matrix
            axes[i, j] = ax

    if axes.size == 1:
        # Get the unique axe
        axes = axes[0, 0]

    return fig, axes


def simulation(
    func: Function,
    markevery: list[int] = None,
    personalPlot: bool = False,
    projection3D: bool = False,
    markerSize: list[int] = None,
    color: list[str] = None,
    show: bool = True,
    strides: int = None,
    normalize: bool = False,
    mesh: Mesh = None,
    position: str = 'right',
    setAxesAspectRatio: bool = True,
        ) -> (Figure):
    '''Plot the respective graphical'''

    if personalPlot:
        # Handling new argument
        scatters = personalPlot

    # Set the keyword args to plot
    plotKwargs = {
        'lw': 3,
        'ms': 10,
        'markevery': markevery,
        'ls': '-',
        'marker': 'o',
        'label': func.name(),
        'markersize': markerSize,
        'color': color
        } if personalPlot or not projection3D\
          else {}

    if mesh is None:
        # Set the plot mesh
        mesh = func.function_space().mesh()

    # Get the geometric dimension
    tdim = mesh.topology().dim()

    # Adjust plot kwargs if quiver
    if any(func.value_shape()) or tdim == 3:
        # Remove specific kwargs
        plotKwargs.pop('ms', None)
        plotKwargs.pop('markevery', None)
        plotKwargs.pop('marker', None)
        plotKwargs.pop('markersize', None)
        plotKwargs.pop('markeredgecolor', None)

    # Set the figure label
    figLabel = f'Plot {func.name()}'

    # Set a subplot figure
    fig, ax = subplot(
        label=figLabel,
        projection='3d' if projection3D else None
        )

    # Plot the exact solution
    funcPlot = plot(
        func, dolfin=not personalPlot, mesh=mesh,
        **plotKwargs, strides=strides,
        )

    # Get values limits
    vlim = (func.vector().min(),
            func.vector().max())

    if any(plt.gca().collections) and not personalPlot:
        # Put colorbar
        setColorbar(plt.gca(), scale=0.05, vlim=vlim,
                    label=func.name(), position=position)

    elif personalPlot and tdim < 3:
        # Put legend in axes
        plt.legend(loc='best')

    elif tdim == 3:
        # Put colorbar in axes
        setColorbar(ax, scale=0.05, position=position,
                    label=func.name(), vlim=vlim)

    else:
        # Show the plot legend in the bestest local
        plt.legend(loc='best')

    if normalize and colorbarAmount(fig) > 1:
        # Get the colored collections
        coloredCollections = getColoredCollections(*fig.axes)

        # Get the color limits
        vlim = getColorLimits(coloredCollections)

        if vmin is not None and vmax is not None:
            # Normalize the colors
            normalizeColors(*coloredCollections, vlim=vlim)

    # Adjust axis limits
    ax.autoscale()

    # Set the plot figure title
    fig.suptitle(f'Plot ${func.name()}$')

    if show:
        # Show plot
        plt.show()

    if setAxesAspectRatio:
        # Set the figure's axes aspect ratio
        setFigAxesAspectRatio(fig)

    return fig


def plotComparison(
    exact: Function,
    numerical: Function = None,
    splitSols: bool = False,
    markevery: list[int] = None,
    scatters: bool = False,
    projection3D: bool = False,
    markerSizes: list[int] = None,
    colors: list[str] = None,
    interactive: bool = True,
    show: bool = True,
    personalPlot: bool = False,
    strides: int = None,
    normalize: bool = False,
    mesh: Mesh = None,
    position: str = 'right',
    setAxesAspectRatio: bool = False,
        ) -> (Figure):
    '''Plot the comparison graphical'''

    if personalPlot:
        # Handling new argument
        scatters = personalPlot

    # if mesh is not None:
    #     # Adjust the projection 3D parameter
    #     projection3D = (
    #         mesh.topology().dim() > 1
    #             and not any(exact.value_shape())
    #             and personalPlot
    #         or mesh.topology().dim() > 2
    #             and any(exact.value_shape())
    #     )

    if numerical is None:
        # Plot only one function
        fig = simulation(
            exact,
            markevery,
            personalPlot,
            projection3D,
            markerSizes[0] if markerSizes is not None else None,
            colors[0] if colors is not None else None,
            show,
            strides,
            normalize,
            mesh,
            position,
            setAxesAspectRatio
            )

        return fig

    # Set the keyword args to plot
    plotKwargs = {
        'common': {
            'lw': 3,
            'ms': 60 if mesh is not None and mesh.topology().dim() > 1 else 15,
            #'markevery': markevery
            },
        'exact': {
            'ls': '',
            'marker': '.',
            # 'markeredgecolor': 'k',
            'label': exact.name()
            },
        'numerical': {
            'ls': '',
            'marker': '+',
            # 'markeredgecolor': 'k',
            'label': numerical.name()
            },
        } if scatters or not projection3D\
          else {k: {} for k in ['common', 'exact', 'numerical']}

    # Adjust any plot settings
    adjustPlotSettings(plotKwargs, markerSizes=markerSizes, colors=colors)

    # Get the geometric dimension
    tdim = numerical.function_space().mesh().topology().dim()\
        if mesh is None\
        else mesh.topology().dim()

    # Adjust plot kwargs if quiver
    if any(numerical.value_shape()) or tdim == 3:
        # Remove specific kwargs
        plotKwargs['common'].pop('ms', None)
        plotKwargs['common'].pop('markevery', None)
        plotKwargs['exact'].pop('marker', None)
        plotKwargs['exact'].pop('markeredgecolor', None)
        plotKwargs['numerical'].pop('marker', None)
        plotKwargs['numerical'].pop('markeredgecolor', None)

    # Set the figure label
    figLabel = f'Plot comparison {numerical.name()} x {exact.name()}'

    if splitSols:
        # Set a subplot figure
        fig, axes = subplot(
            2, 1, label=figLabel,
            projection='3d' if projection3D else None
            )

        if projection3D:
            # Linking the axes movements
            connectAxesMovements(fig)

        # Set the firts axe as the current
        sca(figLabel, 0)

    else:
        # Set a subplot figure
        fig, axes = subplot(
            label=figLabel,
            projection='3d' if projection3D else None
            )

    # Plot the exact solution
    exactPlot = plot(
        exact, dolfin=not scatters, **plotKwargs['common'],
        **plotKwargs['exact'], strides=strides, mesh=mesh
        )

    if splitSols:
        # Get the values limits
        vlim = (exact.vector().min(), exact.vector().max())

        if any(plt.gca().collections) and not scatters:
            # Put colorbar
            setColorbar(plt.gca(), label=exact.name(), scale=0.05,
                        vlim=vlim, position=position)

        # Set the second axe as the current
        sca(figLabel, 1)

    # Plot the numerical solution
    numericalPlot = plot(
        numerical, dolfin=not scatters, **plotKwargs['common'],
        **plotKwargs['numerical'], strides=strides, mesh=mesh
        )

    # Get the values limits
    vlim = (numerical.vector().min(),
            numerical.vector().max())

    if splitSols and any(plt.gca().collections) and not scatters:
        # Put colorbar
        setColorbar(plt.gca(), label=numerical.name(), scale=0.05,
                    vlim=vlim, position=position)

    elif any(plt.gca().collections) and not scatters:
        # Put colorbar
        setColorbar(plt.gca(), scale=0.05, vlim=vlim, position=position)

    elif scatters and tdim < 3:
        # Put legend in axes
        [(plt.sca(ax), plt.legend(loc='best')) for ax in plt.gcf().axes]

    elif tdim == 3:
        # Put colorbar in axes
        [setColorbar(ax, scale=0.05, position='center right',
                     label=getLabel(ax), vlim=vlim)
            for ax in plt.gcf().axes]

    else:
        # Show the plot legend in the bestest local
        plt.legend(loc='best')

    if normalize and colorbarAmount(fig) > 1:
        # Get the colored collections
        coloredCollections = getColoredCollections(*fig.axes)

        # Get the color limits
        vlim = getColorLimits(coloredCollections)

        if vlim is not None:
            # Normalize the colors
            normalizeColors(*coloredCollections, vlim=vlim)

    # Adjust axis limits
    [ax.autoscale() for ax in np.ravel(axes)]

    # Set the plot figure title
    fig.suptitle(f'Comparison: ${exact.name()}$ x ${numerical.name()}$')

    if interactive:
        # Show the interactive plot
        addCheckButtons(
            fig,
            [exactPlot, numericalPlot]
                if tdim > 1 or any(numerical.value_shape())
                else [exactPlot[0], numericalPlot[0]],
            show=show
            )

    if show:
        # Show plot
        plt.show()

    if setAxesAspectRatio:
        # Set the figure's axes aspect ratio
        setFigAxesAspectRatio(fig)

    return fig


def colorbarAmount(
    fig: Figure
        ) -> (int):
    return len([1 for ax in fig.axes if ax.get_label() == '<colorbar>'])


def plotMesh(
    *nodes: Array,
    labels: list[str],
    projection3D: bool = False,
    normals: dict[str:Array] = None,
    setAxesAspectRatio: bool = SET_AXES_ASPECT
        ) -> (None):

    # Create the figure and axe
    fig = plt.figure('Mesh')
    ax = fig.add_subplot(projection='3d')\
        if projection3D\
        else fig.add_subplot()

    # Init the plots list
    plots = []

    # Looping in pair (nodes, label)
    for node, label in zip(nodes, labels):
        # Plot the mesh nodes and append to list
        plots.append(
            ax.scatter(*node, label=label)
        )

        if normals is not None and label in normals:
            # Get the mesh center node
            centerNode = getCenterNode(node.T)

            # Plot the normal vectors in mesh center
            plots.append(
                ax.quiver(*centerNode, *normals[label], label=f'n_{label[0]}')
            )

    if setAxesAspectRatio:
        # Set the axe aspect
        ax.set_box_aspect(getBestAspect(ax))

    # Automatic figure adjust
    fig.tight_layout()

    # Add check buttons
    addCheckButtons(fig, plots, show=True)

    return None


def addCheckButtons(
    fig: plt.Figure,
    lines: list[Any] = None,
    show: bool = True
        ) -> (None):

    if lines is None:
        lines = []
        for ax in fig.axes:
            lines += [line for line in ax.lines]
            lines += [col for col in ax.collections]

    rax = fig.add_axes([0.05, 0.4, 0.1, 0.15])
    labels = [str(line.get_label()) for line in lines]
    visibility = [line.get_visible() for line in lines]
    check = CheckButtons(rax, labels, visibility)

    def func(label):
        index = labels.index(label)
        lines[index].set_visible(not lines[index].get_visible())
        plt.draw()

    check.on_clicked(func)

    # Positioning the check button axe
    positioningAxe(check.ax)

    if show:
        plt.show()

    return None


def positioningAxe(
    ax: Axe,
    x: float = None,
    y: float = None
        ) -> (None):

    return None

    if x is None:
        x = "?"

    # Positioning the checkbutton axe
    pos = ax.get_position()
    posXRange = pos.x1 - pos.x0
    posYRange = pos.y1 - pos.y0
    pos.x0 = fig.axes[0].get_xlim()[0]
    pos.x1 = pos.x0 + posXRange
    pos.y0 = fig.axes[0].get_ylim()[0]
    pos.y1 = pos.y0 + posYRange
    check.ax.set_position(pos)

    return None


def getBestAspect(
    ax: Figure
        ) -> (Union[float, tuple[int]]):
    'Set the aspect-ratio to the axe'

    # Get the axe's axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()\
        if is3d(ax)\
        else (0, 0)

    # Calcule the axis width
    xWidth = xlim[1] - xlim[0]
    yWidth = ylim[1] - ylim[0]
    zWidth = zlim[1] - zlim[0]

    # Set the aspect
    aspect = yWidth/xWidth\
        if zWidth == 0\
        else array([xWidth, yWidth, zWidth])

    return aspect


def plot(
    *args,
    dolfin: bool = True,
    strides: int = None,
    mesh: Mesh = None,
    setAxesAspectRatio: bool = SET_AXES_ASPECT,
    **kwargs
        ) -> (Plot):
    'Plot a graph with matplotlib module'

    global KWARGS_COPY

    if dolfin:
        # if any(args[0].value_shape()):
        #     # Add border to arrows
        #     kwargs['edgecolor'] = 'k'

        # Make plot from dolfin module
        p = df.plot(*args, **kwargs, mesh=mesh)

        # if mesh is not None:
        #     df.plot(mesh, lw=0.3)

        # Get the respective axe
        ax = p.axes\
            if type(p) is not list\
            else p[0].axes

    else:
        # Get the current axe
        ax = plt.gca()

        # Set the strides
        strides = slice(None, None, strides)

        # Init any lists
        vector = []
        tdim = []
        gdim = []
        coord = []
        coordStrides = []
        values = []
        valuesStrides = []

        # Set default value
        dRange = 0

        # Looping in args
        for arg in args:
            # Verify if function is a vector
            vector.append(any(arg.value_shape()))

            # Get the function mesh
            _mesh = arg.function_space().mesh()\
                if mesh is None\
                else mesh

            # Get the topological dimensions
            _tdim = _mesh.topology().dim()\
                if mesh is None\
                else mesh.topology().dim()
            tdim.append(_tdim)

            # Get the geometric dimensions
            _gdim = _mesh.geometric_dimension()\
                if mesh is None\
                else mesh.geometric_dimension()
            gdim.append(_gdim)

            # Get the mesh coordinates from args
            coord.append(
                [c for c in _mesh.coordinates().T]
                )

            # Turn args to array
            if mesh is None:
                values.append(
                    arg.compute_vertex_values()
                        if not any(arg.value_shape())
                        else [argi.compute_vertex_values()
                                for argi in arg.split()]
                    )
            else:
                if _tdim < _gdim:
                    # Consider only the topology coordinates
                    coord = [c[1:] for c in coord]
                    if _tdim == 1:
                        coord = [coord]
                _arg = df.project(arg, mesh=mesh)
                values.append(
                    _arg.compute_vertex_values()
                        if not any(_arg.value_shape())
                        else [argi.compute_vertex_values()
                                for argi in _arg.split()]
                    )
                if _tdim < _gdim and not vector[-1]:
                    dRange = max(
                        dRange, abs(values[-1].max() - values[-1].min())
                        )

                    if dRange < 1e-5:
                        # Set the yaxis margin
                        dy = 1/(10*dRange)
                        if _tdim == 1:
                            ax.set_ylim((dRange*(1 - dy), dRange*(1 + dy)))
                        elif _tdim == 2:
                            # ax.set_xticklabels(
                            #     [#f'{values[-1].max()-2*dy:1.02e}',
                            #      f'{values[-1].max()+:1.02e}',
                            #      #f'{values[-1].max()+2*dy:1.02e}']
                            #     ]
                            # )
                            ax.set_xlim((dRange*(1 - 2*dy), dRange*(1 + 2*dy)))

            if strides != slice(None):
                for _coord in coord:
                    __coord = []
                    for c in _coord:
                        __coord.append(c[strides])
                    # Get the mesh coordinates with strides
                    coordStrides.append(__coord)

                for _values in values:
                    # Get args array with strides
                    valuesStrides.append(
                        _values[strides]
                            if not any(arg.value_shape())
                            else [v[strides]
                                    for v in _values]
                    )

        if strides == slice(None):
            # Get the original elements
            coordStrides = coord
            valuesStrides = values

        if is3d(ax) and 3:
            # Rename the marker size kwarg
            kwargs['s'] = kwargs.pop('ms', 10)

            # Remove any kwarg
            [kwargs.pop(v)
                for v in ['markevery', 'markeredgecolor']
                if v in kwargs]

        # Looping in args
        for i, value in enumerate(valuesStrides):

            if tdim[i] == 3 and not vector[i]:
                # Plot the scatters
                p = ax.scatter(*coordStrides[i], **kwargs)

                # Set the scatter's color
                p.set_array(value)

            elif tdim[i] != 3 and vector[i]:
                # Remove the 's' keyword
                kwargs.pop('s', None)

                # Plot the velocity field
                p = ax.quiver(*coordStrides[i], *value, **kwargs)
                

                # Set the arrow's color
                p.set_array((array(value)**2).sum(axis=0)**0.5)

            elif tdim[i] == 3 and vector[i]:
                # Remove the 's' keyword
                kwargs.pop('s', None)

                # Get the vector norm from values without strides
                norm = (array(values[i])**2).sum(axis=0)**0.5

                # Get the current figure
                fig = ax.figure

                if 'KWARGS_COPY' not in globals():
                    # Clean the figure properties
                    fig.clf()

                # Add the subplot
                [fig.add_subplot(
                    221 + 2*k + ('KWARGS_COPY' in globals()),
                    label=lbl,
                    projection='3d'
                    ) for k, lbl in enumerate(['field', 'norm'])]

                # Get the axe of the velocity field
                ax = fig.axes[i + 2*('KWARGS_COPY' in globals())]

                # Get a kwargs dict copy
                KWARGS_COPY = kwargsCopy = kwargs.copy()

                # Remove the linewidth argument
                kwargsCopy.pop('lw', None)
                kwargsCopy.pop('ls', None)

                # Plot the velocity field
                p = ax.quiver(
                    *coordStrides[i], *value, **kwargs,
                    linewidth=1, linestyle='-', arrow_length_ratio=0.75,
                    )

                # # Set the axis labels
                # ax.set_xlabel('$x$')
                # ax.set_ylabel('$y$', rotation='horizontal')
                # ax.set_zlabel('$z$', rotation='horizontal')

                if setAxesAspectRatio:
                    # Set the axe box aspect
                    ax.set_box_aspect(getBestAspect(ax))

                if not any(fig.axes[-1].collections):
                    # Plot the magnitude dots
                    p = fig.axes[-1].scatter(
                        *coord[i],
                        label=f' |{kwargs.get("label", ".")}|_0'
                        )

                    # Colorize the norm
                    p.set_array(norm)

                    if setAxesAspectRatio:
                        # Set the axe box aspect
                        fig.axes[-1].set_box_aspect(
                            getBestAspect(fig.axes[-1])
                            )

                # Enable some event handlings
                connectAxesMovements(fig)
                enableFocalMode(fig)
                disableFocalMode(fig)

                # testLoop(globals(), locals())

            else:

                try:
                    # Plot the scatters
                    p = ax.scatter(value, *coordStrides[i], **kwargs)\
                        if tdim[i] == 2 and gdim[i] == 3\
                        else ax.scatter(*coordStrides[i], value, **kwargs)
                except AttributeError:
                    if 'ms' in kwargs:
                        kwargs.pop('ms')
                    if 'markerfacecolor' in kwargs:
                        kwargs.pop('markerfacecolor')
                    if 'markersize' in kwargs:
                        kwargs.pop('markersize')
                    if 'markevery' in kwargs:
                        kwargs.pop('markevery')
                    #p = ax.scatter(value, *coordStrides[i], **kwargs)
                    p = ax.scatter(*coordStrides[i], value, **kwargs)

    # Set the axis labels
    ax.set_xlabel('$x$', labelpad=10)
    ax.set_ylabel('$y$', rotation='horizontal')
    if hasattr(ax, 'set_zlabel'):
        ax.set_zlabel('$z$', rotation='horizontal')

    if setAxesAspectRatio:
        # Set the axe aspect
        ax.set_box_aspect(getBestAspect(ax))

    return p


def adjustPlotSettings(plotKwargs, **kwargs):
    if kwargs.get('markerSizes', None) is not None:
        # Get the markerSizes value
        markerSizes = kwargs['markerSizes']

        # Update the exact plot settings
        plotKwargs['exact']['ms'] = markerSizes[0]\
            if markerSizes[0] is not None\
            else plotKwargs['common']['ms']

        # Update the numerical plot settings
        plotKwargs['numerical']['ms'] = markerSizes[1]\
            if markerSizes[1] is not None\
            else plotKwargs['common']['ms']

        # Remove from common args
        plotKwargs['common'].pop('ms', None)

    if kwargs.get('colors', None) is not None:
        # Get the colors value
        colors = kwargs['colors']

        if colors[0] is not None:
            # Update the exact plot settings
            plotKwargs['exact']['color'] = colors[0]

        if colors[1] is not None:
            # Update the numerical plot settings
            plotKwargs['numerical']['color'] = colors[1]

    return None


def dynamicComparison(
    solutions,
    iterator: list[float],
    iteratorFormatter: str = 't = {:1.02}/',
    labels: list[str] = None,
    linestyles: list[str] = ['--', ''],
    markers: list[str] = ['o', '*'],
    markersizes: list[int] = [10, 5],
    markevery: list[int] = None,
    multipleViews: bool = False,
    splitSols: bool = False,
    show: bool = True
        ) -> (None):
    # Set the iterator string format
    iteratorFormat = iteratorFormatter + str(iterator[-1])

    # Plot evolutive solutions
    fig = _dynamicPlot(
        *solutions,
        suptitles=[iteratorFormat.format(k) for k in iterator],
        labels=labels,
        linestyle=linestyles,
        marker=markers,
        markevery=markevery,
        markersizes=markersizes,
        fps=max(ceil(len(iterator)//10), 2),
        globalVars=globals(),
        multipleViews=multipleViews,
        splitSols=splitSols
        )

    if show:
        # Distribute figures in screen
        adjustFiguresInScreen(fig)

        # Show the graphs
        plt.show()

    return fig


def _dynamicPlot(*sols, titles: str = None,
                suptitles: str = None, freezeVAxis: bool = True,
                labels: str = '', linestyle: str = '-', marker: str = 'o',
                fps: int = 1, show: bool = False, numRepeats: int = 100,
                globalVars: dict = None, videoOutput: str = None,
                adjustStride: bool = True, graphsOutput: str = None,
                suptitlePos: str = 'lower center', figTitle: str = None,
                markevery: list[int] = None, markersizes: list[int] = None,
                multipleViews: bool = False, splitSols: bool = False,
                ticksFontSize: int = None
                ) -> (plt.Figure):

    if globalVars is None:
        globalVars = globals()

    if type(linestyle) is not list:
        linestyle = len(sols)*[linestyle]

    if type(marker) is not list:
        linestyle = len(sols)*[marker]

    if type(labels) is not list:
        labels = len(sols)*[labels]

    # Convert sols in list
    sols = list(sols)

    # Set the animated figure label
    figureLabel = 'Animated Plot'\
        if figTitle is None\
        else figTitle

    # Generate a id to figure label
    id = generateId(figureLabel, globals())

    if figureLabel in plt.get_figlabels():
        # Add the id number to figure label
        figureLabel += f'_{id}'

    # Set the global repeat label
    repeatLabel = f'REPEATS_{id}'

    # Get the mesh
    mesh = sols[0][n:=0].function_space().mesh()

    # Get mesh domain dimension
    tdim = mesh.topology().dim()

    if tdim < 3 and not multipleViews:
        # Set a generator to each follow plot properties
        colors = (c for c in ['tab:blue', 'tab:orange', 'tab:green'])
        alphas = (a for a in np.linspace(0.5, 1, 3))

    elif tdim < 3 and multipleViews:
        # Set a generator to each follow plot properties
        colors = [
            (c for c in ['tab:blue', 'tab:orange', 'tab:green'])
                for _ in range(3)
            ]
        alphas = [
            (a for a in np.linspace(0.5, 1, 3))
                for _ in range(3)
            ]

    elif tdim == 3 and multipleViews:
        # Set default values
        colors = alphas = [None, None, None]

    else:
        # Set default values
        colors = alphas = None

    # Create a figure
    fig = plt.figure(figureLabel, dpi=100)

    # Generate the axe(s)
    axes = generateSubplots(
        fig, tdim, multipleViews,
        splitSols=len(sols)*splitSols
        )

    if tdim > 1:
        # Linking the axes movements
        connectAxesMovements(fig)

    # Set the main axe
    if type(axes) is list and type(axes[0]) is list:
        ax = axes[-1][0]
    elif type(axes) is list:
        ax = axes[0] if not splitSols else axes[-1]
    else:
        ax = axes

    # Set the text position
    textPos = {
        'upper left': (0.02, 0.95, 0.95),
        'upper center': (0.425, 0.95, 0.95),
        'upper right': (0.8, 0.95, 0.95),
        'lower center': (0.425, -0.02, 0.95),
        }[suptitlePos][:tdim+1]

    if suptitles is not None:
        # Set a text
        suptitle = ax.text(*textPos, s='', transform=ax.transAxes,
                           backgroundcolor='0.8', color='black')

    # Get mesh coordinates
    X = mesh.coordinates()

    # Split coordinates to each axis
    X = [X[:, i] for i in range(tdim)]

    # Init lines
    lines = []\
        if not multipleViews\
        else [[] for _ in range(3)]

    # Looping in sols
    for k, sol in enumerate(sols):
        # Adjust plot label to latex notation
        label = f'${labels[k]}$'

        # Set the kwargs
        kwargs = {
            'visible': False,
            'label': label,
            'marker': marker[k]
            }

        if tdim == 1:
            # Add others mark arguments
            kwargs['markevery'] = markevery
            kwargs['markersize'] = markersizes[k]\
                if markersizes is not None\
                else 10
            kwargs['markeredgecolor'] = 'k'

        # Turn fenics all time functions/expressions to array
        sol = sols[k] = [
            s.compute_vertex_values(mesh)
                for s in sol
            ]

        if not multipleViews and not splitSols:
            # Make the plot
            line = makePlot(
                axes, X, sol[0], alphas, colors, **kwargs
                )

            # Add to lines list
            lines.append(line)

        elif not multipleViews:
            # Make the plot
            line = makePlot(
                axes[k], X, sol[0], alphas, colors, **kwargs
                )

            # Add to lines list
            lines.append(line)

        else:
            # Looping in axes
            for j, ax in enumerate(axes[k] if splitSols else axes):
                # Make the plot
                line = makePlot(ax, X, sol[0], alphas[j], colors[j], **kwargs)

                # Add to respective axe lines list
                lines[j].append(line)

        # Turn current time solutions to array
        solArray = array(sol)

        if 'lim' not in locals():
            # Get the max and min of all values in array
            lim = (solArray.min(), solArray.max())

        else:
            # Check the min and max and update
            lim = (
                min(lim[0], solArray.min()),
                max(lim[1], solArray.max()),
                )

    if multipleViews and splitSols:
        # Adjust axes aspect
        [[ax.set_box_aspect(getBestAspect(ax))
            for ax in axes[k]]
         for k in range(len(sols))]

    elif multipleViews:
        # Adjust axes aspect
        [ax.set_box_aspect(getBestAspect(ax)) for ax in axes]

    if type(lines[0]) is not list:
        # Set plot properties
        setPlotProperties(
            *lines, linewidth=4
            )
        [setPlotProperties(line, linestyle=ls, zorder=zo, marker=m)
            for line, ls, m, zo in zip(
                lines, linestyle, marker, range(len(sols))[::-1]
                )]

    else:
        for _lines in lines:
            # Set plot properties
            setPlotProperties(
                *_lines, linewidth=4
                )
            [setPlotProperties(line, linestyle=ls, zorder=zo, marker=m)
                for line, ls, m, zo in zip(
                    _lines, linestyle, marker, range(len(sols))[::-1]
                    )]

    if ticksFontSize is not None:
        # Resize the ticks label
        resizeTicksFont(ax, fontSize=ticksFontSize)\
            if type(axes) is not list\
            else [resizeTicksFont(ax, fontSize=ticksFontSize)
                    for ax in np.ravel(axes)]

    # Enable the grid
    plt.grid()

    if freezeVAxis:
        # Set vertical pad
        vPad = 0.1*(lim[1] - lim[0])

        # Add the pad
        lim = (lim[0] - vPad, lim[1] + vPad)

        # Update the plot vertical limits
        ax.set_ylim(lim)

    def initPlot():
        # Change the plot visibility
        thrds = [
            parallel(line.set_visible, True)[0]
                for line in np.ravel(lines)
            ]

        # Stop threads
        [thrd.join() for thrd in thrds]

        if suptitles is not None:
            suptitle.set_text('Starting')

        return lines, suptitle

    def updatePlotOneAux(
        plot: Plot,
        data: list[float]
            ) -> None:
        # Get the setter function associated to this plot
        setter = plotDataSetter(plot)

        if ani.wait:
            # Slow animation
            time.sleep(0.5)

        # Update the plot data
        setter(
            plot, data,
            mesh.coordinates()
                if tdim == 2
                else slice(None)
            )

        return None

    def updatePlotOne(
        lines: list[Plot],
        frame: int,
        data: list[float]
            ) -> None:

        # Update the plots
        thrds = [parallel(updatePlotOneAux, lines[k], data_k[frame])[0]
            for k, data_k in enumerate(data)]

        # Stop threads
        [thrd.join() for thrd in thrds]

        if suptitles is not None:
            # Update the suptitle
            suptitle.set_text(suptitles[frame])

        return None

    def updatePlotSubplot(
        lines: list[Plot],
        frame: int,
        data: list[float],
        numSubplots: int = 1
            ) -> None:

        # Make the update one by one
        thrds = [parallel(updatePlotOne, lines[k], frame, data)[0]
            for k in range(numSubplots)]

        # Stop threads
        [thrd.join for thrd in thrds]

        return None

    def updatePlot(frame, data):

        try:

            if not multipleViews:
                # Apply the respective plot update
                updatePlotOne(lines, frame, data)

            else:
                # Apply the respective plot update
                updatePlotSubplot(
                    lines, frame, data, 3
                    )

        except KeyboardInterrupt:
            # Close animation
            plt.close(fig.number)

        return lines, suptitle

    def toggle_pause(event):
        # Get figure toolbar manager
        tm = fig.canvas.manager.toolmanager
        pauseTool = tm.get_tool('Pause')

        # Get the global var
        paused = pauseTool.paused

        if not paused:
            # Pause animation
            ani.pause()

            # Update the toggled var
            pauseTool.paused = True

        elif paused:
            # Resume animation
            ani.resume()

            # Update the toggled var
            pauseTool.paused = False

    # Create the animation
    ani = FuncAnimation(
        fig,               # The base figure
        updatePlot,        # The animation function
        len(sol),          # number of the rates
        fargs=(sols,),     # Args to animation function
        init_func=initPlot,# The initial animation
        interval=1000/fps, # Delay between frames (mileseconds)
        repeat=True,       # If repeat animation or not
        repeat_delay=10,   # Delay between repetitions (mileseconds)
        blit=False,        # Otimize the animation (Do not work to this case)
        )

    # Add any attributes
    ani.lastFrame = ani.totalFrames = len(sol)
    ani.paused = False
    ani.initialInterval = ani._interval
    ani.videoOutput = videoOutput
    ani.fps = fps
    ani.wait = True

    if tdim < 3:
        # Get the vertical limits from all solutions given
        maxs = []
        mins = []
        for sol in sols:
            # Append to respective lists the maximal and the minimal
            [maxs.append(s.max()) for s in sol]
            [mins.append(s.min()) for s in sol]

        # Get the major and minor values
        vmax = max(maxs)
        vmin = min(mins)

        # Adjust plots in axes
        autoAdjustAxis(fig, vmax=vmax, vmin=vmin)

    try:
        # Set the legend/colorbar position
        position = {
            1: 'upper right',
            2: 'upper center',
            3: 'upper center'
            }[tdim]

        # Set the correspondently axes parameters
        if multipleViews:
            _axes = axes
        elif splitSols:
            _axes = np.reshape(axes, (-1, 1)).tolist()
        else:
            _axes = []

        # Put legends/colorbars
        putLegends(
            fig, colorbar=(tdim == 3),
            axMainId=0 if multipleViews or splitSols else None,
            position=position, axes=_axes
            )

        # Adjust plots in axes
        autoAdjustAxis(fig)

    except AttributeError:
        pass

    def changeFrame(text, **kwargs):

        if any(text) and (text.isnumeric() or text[0] == '-'):
            try:
                # Evaluate the frame number
                frame = eval(text)

                # Adjust the frame numbering
                if frame < 0:
                    frame = ani.lastFrame + frame
                elif frame >= ani.lastFrame:
                    frame = 0

                # Pause de animation
                ani.pause()

                if 'pauseButton' in kwargs:
                    kwargs['pauseButton'].config(text='>')

                # Draw the respective frame
                ani._draw_frame(frame)

                # Update the frame sequence
                ani.frame_seq = (f for f in range(frame, ani.lastFrame))

                if suptitles is not None:
                    # Change the suptitle frame
                    suptitle.set_text(suptitles[frame])

                # Refresh the window
                ani._fig.canvas.draw()

            except Exception:
                pass

        return None

    def pauseAnimation(text):
        ani.pause()
        return None

    # Add a text box to change frames
    addPlayer(
        fig, 'Frame', submitFunc=changeFrame,
        textChangeFunc=pauseAnimation, ani=ani,
        globalVars=globalVars,
        )

    if 'animations' not in globalVars:
        # Create a global animations list
        globalVars['animations'] = [ani]
    else:
        # Add to global animations list
        globalVars['animations'].append(ani)

    # Adjust widgets in window
    fig.tight_layout()

    if videoOutput is not None:
        # Save this animation
        saveAnimation(ani, videoOutput, fps=fps)

    if show:
        # Adjust figures in screeen
        adjustFiguresInScreen(fig)

        # Show the animation
        plt.show()

    return fig


def adjustFiguresInScreen(
    *figures: Figure,
    monitor: int = 0,
    subplots: bool = False,
    fullscreen: bool = False,
    dx: float = 1e-2,
    dy: float = 5e-2,
    pad: int = 1
        ) -> (None):
    'Positioning matplotlib figures in respective monitor screen'

    if matplotlib.get_backend() == 'WebAgg':
        return None

    # Get the monitor info
    width, height = getScreenInfo(monitor)

    # Set the fig amount
    figAmount = len(figures)

    # Set the figures size
    sizes = figAmount*[
        (width*(2/figAmount - dx), height*(1/2 - dy))
        ] if figAmount > 1\
          else [(width*(1 - dx), height*(1 - dy))]

    # Set the window step size Mesh
    hw = 2/figAmount
    hh = 0.5

    # Set the figures position
    positions = zip(
        2*[*(width*np.arange(0, 1, hw))],
        height*np.repeat([0, hh], figAmount//2)
    ) if figAmount > 1\
      else [(width*0.1, height*0.1)]

    # Convert respective sizes and positions to int
    sizes = list(map(lambda x: tuple(map(int, x)), sizes))
    positions = list(map(lambda x: tuple(map(int, x)), positions))

    # Looping in figures
    for fig, size, position in zip(figures, sizes, positions):
        # Turn respective figure to current
        fig = plt.figure(fig.number)

        # Get the figure manager
        man = plt.get_current_fig_manager()

        if fullscreen:
            # Turn figure to fullscreen
            man.full_screen_toggle()

            # Next figure
            continue

        # Set the geometry of this figure
        if type(man.window.geometry()) is not str:
            man.window.geometry().setRect(
                *size, *position
                )

        else:
            man.window.wm_geometry(
                '{}x{}+{}+{}'.format(*size, *position)
                )

        # Show respective figure
        fig.canvas.draw()

        # Adjust the figures
        fig.tight_layout(pad=pad)

    # Add close all button
    #addMyButtons(*figures, onlyClose=True)
    enableCloseAllTwoClicks(*figures)

    return None


def enableCloseAllTwoClicks(
    *figures: Figure
        ) -> (None):

    # Get all figures labels
    figLabels = getFigureLabel(*figures)

    def onclick(event):

        if event.dblclick:
            return [plt.close(label) for label in figLabels]

    # Enable the event handler to figures
    [fig.canvas.mpl_connect('button_press_event', onclick)
        for fig in figures]

    return None


def getFigureLabel(
    *figures: Figure
        ) -> (Union[str, list[str]]):

    # Get the figure labels
    labels = [fig.get_label() for fig in figures]

    if len(labels) == 1:
        # Get the unique value
        labels = labels[0]

    return labels


def addMyButtons(
    *figs: plt.Figure,
    globalVars: dict = globals(),
    onlyClose: bool = True,
    interactive: bool = False,
    varName: str = 'SHOW_GRAPH',
    animation: FuncAnimation = None,
    playPauseButton: tk.Button = None,
    fps: int = None
        ) -> (None):
    'Add buttons to toolbar figures'

    # Get the figures label
    figuresLabel = [fig.get_label() for fig in figs]

    # Looping in figures
    for fig in figs:

        try:
            fig.canvas.manager.toolbar.add_tool
        except AttributeError:
            continue

        if not onlyClose:
            # Add a button to show or hide contours
            fig.canvas.manager.toolmanager.add_tool(
                'Contour', GroupHideTool, gid=None, kmap='c',
                title='Contour'
                )
            fig.canvas.manager.toolbar.add_tool(
                'Contour', 'toolgroup'
                )

            # Add a button to show or hide solution
            fig.canvas.manager.toolmanager.add_tool(
                'Solution', GroupHideTool, gid='solution', kmap='s',
                title='Solution'
                )
            fig.canvas.manager.toolbar.add_tool(
                'Solution', 'toolgroup'
                )

        if interactive:
            # Make a dropdown to select the Area, or "All"
            area = widgets.Dropdown(
                options=['All'] + list(df['Area'].unique()),
                value='All',
                description='Area:',
            )

            # Add a button to show or hide contours
            fig.canvas.manager.toolmanager.add_tool(
                'Contour', GroupHideTool, gid=None, kmap='c',
                title='Contour'
                )
            fig.canvas.manager.toolbar.add_tool(
                'Contour', 'toolgroup'
                )

        # Add a button to disable graphs exhibition
        fig.canvas.manager.toolmanager.add_tool(
            'DisableGraphs', ChangeVar, varName=varName,
            onValue=False, offValue=True, globalVars=globalVars
            )
        fig.canvas.manager.toolbar.add_tool(
            "DisableGraphs", "toolgroup"
            )

        # Add a button to close all
        fig.canvas.manager.toolmanager.add_tool(
            ' CloseAll ', CloseAllTool, labels=figuresLabel
            )
        fig.canvas.manager.toolbar.add_tool(
            " CloseAll ", "toolgroup"
            )

        if animation is not None:
            # # Add a button to pause animation
            # fig.canvas.manager.toolmanager.add_tool(
            #     'SaveAni', SaveAnimationTool, animation=animation[-1],
            #     fps=fps if fps is not None else 30
            #     )
            # fig.canvas.manager.toolbar.add_tool(
            #     "SaveAni", "toolgroup"
            #     )

            # Add a button to pause animation
            fig.canvas.manager.toolmanager.add_tool(
                ' PauseAll ', PauseAnimationTool, animation=animation,
                playPauseButton=playPauseButton
                )
            fig.canvas.manager.toolbar.add_tool(
                " PauseAll ", "toolgroup"
                )

    return None


def plotDataSetter(plot: Plot) -> (None):
    # Get the plot type
    plotType = plot.__class__.__name__

    # Define the data getter convenently
    setter = eval({
        'TriContourSet': 'lambda p, v, coord: updateContours(p, v, coord)',
        'Quiver': 'lambda p, v, _: p.set_offsets(v)',
        'Line2D': 'lambda p, v, strides: p.set_ydata(v)',
        'Line3DCollection': 'lambda p, v, mCoord: updateVerts3D(p, v, mCoord)',
        'Poly3DCollection': 'lambda p, v, mCoord: updateVerts3D(p, v, mCoord)',
        'Path3DCollection': 'lambda p, v, _: updateArray(p._offsets3d[-1], v)'
            if hasattr(plot, 'get_array') and plot.get_array() is None
            else 'lambda p, v, _: p.set_array(v)',
        'PathCollection': 'lambda p, v, _: p.set_offsets(v)'
        }[plotType], globals(), locals())

    return setter


def autoAdjustAxis(
    *figures,
    padx=None,
    pady=None,
    pad=0.1,
    vmin=None,
    vmax=None
        ) -> (None):

    # Looping in plots
    for fig in figures:
        # Looping in figure axes
        for ax in fig.axes:
            # Adjust the axe scale
            ax.autoscale()
            if vmin != vmax is not None:
                dv = (vmax - vmin)*pad
                if hasattr(ax, 'set_zlim'):
                    ax.set_zlim((vmin-dv, vmax+dv))
                else:
                    ax.set_ylim((vmin-dv, vmax+dv))

    return None


def putLegends(
    *figures: plt.Figure,
    axes: list[Axe] = None,
    outsideAxe: bool = True,
    tickFont: int = 12,
    colorbar: bool = False,
    position: str = 'upper right',
    axMainId: int = None
        ) -> (None):

    if not colorbar:
        # Get the anchor and position to legend
        anchor, position, ncol = getLegendAnchor(position) \
            if outsideAxe \
            else (None, 'best')

    # Looping in figures
    for fig in figures:
        # Set current figure
        plt.figure(fig.get_label())

        if axes is None:
            # Get the axes
            axes = fig.get_axes()

        elif type(axes) is not list:
            # Put in list
            axes = [axes]

        if axMainId is not None:
            # Anchor legend to respective axe
            axes = [axes[axMainId]]\
                if type(axes[0]) is not list\
                else [ax[axMainId] for ax in axes]

        # Looping in axes
        for ax in axes:

            #if containsClassName('Line', ax.collections, aller=True):
            if not colorbar:
                # Make the legends
                leg = ax.legend(bbox_to_anchor=anchor, loc=position,
                                handlelength=2, ncol=ncol)

                # Looping in legend lines
                for i, item in enumerate(leg.legendHandles):
                    # Set the visibility of the legend lines
                    item.set_visible(True)

                    # Set the linewidth of the legend lines
                    item.set_linewidth(2.25)

                    if item.__class__.__name__ == 'Rectangle':
                        if None not in item.get_linestyle():
                            style = item.get_linestyle()
                            temp = array(style[1])
                            temp *= 0.5
                            style = (style[0], temp)
                            item.set_linestyle(style)

            elif any(ax.collections):
                # Create the colorbar and add to axe
                setColorbar(
                    ax, ax.collections[0].get_label(),
                    position=position
                    )

    return None


def addPlayer(
    fig: plt.Figure,
    label: str,
    submitFunc: Function,
    textChangeFunc: Function = None,
    initialText: str = '0',
    position: str = 'right',
    boxSize: tuple[float] = (0.07, 0.05),
    inside: bool = False,
    hovercolor: str = None,
    velocityField: 'Quiver' = None,
    tooltipText: str = None,
    ani: FuncAnimation = None,
    globalVars: dict[str: Any] = globals(),
    vertical: bool = False,
    float: bool = True,
    fluid: bool = False,
    splitedFig: bool = False
        ) -> (None):

    # Set the current figure
    fig = plt.figure(fig.get_label())

    # Get the figure manager
    man = plt.get_current_fig_manager()

    try:
        # Get the tk window
        root = man.window
    except AttributeError:
        globalVars['playPauseButtons'] = None
        return None, None

    # Create a new canvas from figure canvas
    canvas = fig.canvas

    if float:
        # Create the player window
        root = tk.Toplevel(root, relief=tk.FLAT)

        # Associate the player to root window
        root.transient(root.master)

        # Enable player over all windows
        # root.attributes('-topmost', True)

        # Positioning the player on root window
        root.geometry('+775+60')

        # Disable the 'x' button to exit window
        root.protocol("WM_DELETE_WINDOW", lambda: None)

        # Make don't resizable
        root.resizable(0, 0)

        # Call the root window to show
        root.master.lift()

    # Create a frame to iteractive widgets
    frameMain = tk.Frame(root)
    frameMain.pack(
        side=tk.BOTTOM if not vertical else tk.RIGHT,
        fill=tk.BOTH,
        expand=1,
        anchor=tk.CENTER
        )

    # Set three new frames
    frame1 = tk.Frame(frameMain)
    frame1.grid(row=0, sticky=tk.W)
    frame2 = tk.Frame(frameMain)
    frame2.grid(row=1, sticky=tk.W)
    frame3 = tk.Frame(frameMain)
    frame3.grid(row=2, sticky=tk.W)

    # Create a tk string var
    numFrame = tk.StringVar()

    # Set the submit function
    def submitFuncHandle():
        submitFunc(numFrame.get())
        textChangeFunc(numFrame.get())
        return None

    def closeRoot():
        plt.close(fig.number)
        return None

    # Create a tk label
    label = tk.Label(frame1, text=label)
    label.grid(row=0, column=0)

    # Create a tk text box
    textBox = tk.Entry(frame1, textvariable=numFrame, width=5)
    textBox.grid(row=0, column=1)
    textBox.bind('<Return>', lambda ev: submitFuncHandle())

    # Create a tk button
    submitButton = tk.Button(frame1, text='OK', command=submitFuncHandle)
    submitButton.grid(row=0, column=2)

    # Create a empty space
    emptyLabel = tk.Label(frame1, text=20*' ')
    emptyLabel.grid(row=0, column=3)

    # Create a tk button
    closeButton = tk.Button(frame1, text='    X    ', command=closeRoot)
    closeButton.grid(row=0, column=4)

    # playImg = tkinter.PhotoImage(file=f'{ICONS_FOLDER}/play.png')
    # pauseImg = tkinter.PhotoImage(file=f'{ICONS_FOLDER}/pause.png')
    # firstImg = tkinter.PhotoImage(file=f'{ICONS_FOLDER}/first.png')
    # lastImg = tkinter.PhotoImage(file=f'{ICONS_FOLDER}/last.png')
    # nextImg = tkinter.PhotoImage(file=f'{ICONS_FOLDER}/next.png')
    # previousImg = tkinter.PhotoImage(file=f'{ICONS_FOLDER}/previous.png')

    firstButton = tk.Button(
        frame2, text='|<',
        command=lambda: submitFunc(
            '0', pauseButton=playPauseButton
            ),# image=firstImg
        )
    firstButton.grid(row=0, column=3)

    previousButton = tk.Button(
        frame2, text='<<',
        command=lambda: submitFunc(
            str(getCurrentFrame(ani)-1), pauseButton=playPauseButton
            ),# image=previousImg
        )
    previousButton.grid(row=0, column=4)

    def submitFuncPlayPause():
        if ani.paused:
            playPauseButton.config(text='||')
            ani.resume()
            ani.paused = False
        else:
            playPauseButton.config(text='>')
            ani.pause()
            ani.paused = True

    playPauseButton = tk.Button(
        frame2, text='||',
        command=submitFuncPlayPause,# image=pauseImg
        )
    playPauseButton.grid(row=0, column=5)

    if 'playPauseButtons' not in globalVars:
        # Create a global animations list
        globalVars['playPauseButtons'] = [playPauseButton]
    else:
        # Add to global animations list
        globalVars['playPauseButtons'].append(playPauseButton)

    nextButton = tk.Button(
        frame2, text='>>',
        command=lambda: submitFunc(
            str(getCurrentFrame(ani)+1), pauseButton=playPauseButton
            ),# image=nextImg
        )
    nextButton.grid(row=0, column=6)

    lastButton = tk.Button(
        frame2, text='>|',
        command=lambda: submitFunc(
            str(ani.lastFrame-1), pauseButton=playPauseButton
            ),# image=lastImg
        )
    lastButton.grid(row=0, column=7)

    # def changeSpeed():
    #     if speedStr.get() == '1x':
    #         ani.pause()
    #         playPauseButton.config(text='||')
    #         ani.wait = True
    #         ani._draw_frame(getCurrentFrame(ani))
    #         ani.resume()
    #         playPauseButton.config(text='>')
    #         speedStr.set('2x')
    #         speedButton.configure(text='2x')
    #     else:
    #         ani.pause()
    #         playPauseButton.config(text='||')
    #         ani.wait = False
    #         ani._draw_frame(getCurrentFrame(ani))
    #         ani.resume()
    #         playPauseButton.config(text='>')
    #         speedStr.set('1x')
    #         speedButton.configure(text='1x')

    # speedStr = tk.StringVar(frame, name='1x')
    # speedStr.set('1x')

    # speedButton = tk.Button(
    #     frame, text=speedStr.get(),
    #     command=lambda: changeSpeed(),# image=lastImg
    #     )
    # speedButton.grid(row=0, column=8)

    saveButton = tk.Button(
        frame2, text='mp4', font='bold',
        command=lambda: saveAnimation(ani, ani.videoOutput, ani.fps),
        # image=lastImg
        )
    saveButton.grid(row=0, column=9)

    if fluid:
        arrowEdge = tk.IntVar()
        checkButton = tk.Checkbutton(
            frame3, text='ArrowEdge', variable=arrowEdge,
            onvalue=1, offvalue=0,
            command=lambda: [submitFuncPlayPause() for _ in range(2)]
            #command=lambda: addEdgeToArrow(velocityField, edgeArrow)
            )
        checkButton.grid(row=0, column=0)
        revertColors = tk.IntVar()
        checkButton = tk.Checkbutton(
            frame3, text='RevertColors', variable=revertColors,
            onvalue=1, offvalue=0,
            command=lambda: [submitFuncPlayPause() for _ in range(2)]
            )
        checkButton.grid(row=0, column=1)

        # Store the previous check vars
        checkVars = {
            'arrowEdge': arrowEdge,
            'revertColors': revertColors,
            }

        if not splitedFig:
            pressureVisible = tk.IntVar()
            checkButton = tk.Checkbutton(
                frame3, text='Pressure', variable=pressureVisible,
                onvalue=1, offvalue=0,
                command=lambda: [submitFuncPlayPause() for _ in range(2)]
                )
            checkButton.select()
            checkButton.grid(row=1, column=0)

            velocityVisible = tk.IntVar()
            checkButton = tk.Checkbutton(
                frame3, text='Velocity', variable=velocityVisible,
                onvalue=1, offvalue=0,
                command=lambda: [submitFuncPlayPause() for _ in range(2)]
                )
            checkButton.select()
            checkButton.grid(row=1, column=1)

            # Store the previous check vars
            checkVars.update({
                'velocityField': velocityVisible,
                'pressure': pressureVisible
                })

    else:
        checkVars = None

    # if tooltipText is not None:
    #     #addCanvasLatex(frame2, (7, 4), tooltipText.replace('; ', '\n'))
    #     scrolledText = scrolledtext.ScrolledText(
    #         frame2,
    #         wrap=tk.WORD,
    #         width=80,
    #         height=1.5,
    #         font=("Times New Roman", 11),
    #         )
    #     scrolledText.insert(
    #         tk.INSERT,
    #         tooltipText.replace('; ','\n')
    #         )
    #     scrolledText.config(state=tk.DISABLED)
    #     scrolledText.grid(row=1, column=0)

    # Adjust widgets in screen
    fig.tight_layout()

    return fig, checkVars


def saveAnimation(
    animation: FuncAnimation,
    videoName: str,
    fps: int = 30,
    parallel: bool = False
    ):

    # Pause respective animation
    animation.pause()

    # Generate the video maker
    writervideo = FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=['-pix_fmt', 'yuv420p'],
        bitrate=-1
        )

    # Get the video output folder
    videoFolder = './'

    if videoName is None:
        # Get the video file name
        videoName = easygui.enterbox(
            "What's the animation output name?",
            'OpytimalControl PDE',
            'AnimationPlot'
            )

        if videoName is None:
            return None
        else:
            videoName = videoName.split('.')[0]

    # Get the animation figure
    fig = animation._fig

    # Get current figure dpi
    dpi = fig.get_dpi()

    # Increase the figure dpi
    fig.set_dpi(300)

    # Save the animation
    animation.save(
            f'{videoFolder}/{videoName}.mp4',
            writer=writervideo,
            progress_callback=lambda frame, total: showProgress(
                '\rSaving plot animation: ', frame+1, total=total
                )
            )

    # Rotate the axis in 120 degrees (To 3D plots)
    triDim = rotateAxes(fig, deg=120)

    if triDim:
        # Update the figure
        fig.canvas.draw()

        # Save the animation (Rotated 120 degrees)
        animation.save(
            f'{videoFolder}/{videoName}_120deg.mp4',
            writer=writervideo,
            progress_callback=lambda frame, total: showProgress(
                '\rSaving plot animation: ', frame+1, total=total
                )
            )

        # Restore the initial rotation
        rotateAxes(fig, deg=0)

    plt.close(fig.get_label())

    # # Reset the figure dpi
    # fig.set_dpi(dpi)

    #globals()['saveAniThread'].join()

    return None


def rotateAxes(fig, deg=0):
    # Set a default value
    triDim = False

    # Looping in figure axes
    for ax in fig.axes:

        if '3D' in ax.__class__.__name__:
            # Rotate the figure
            ax.view_init(30, -60+deg)

            # Update this var
            triDim = True

    return triDim


def setPlotProperties(*plots, id: int = None, **kwargs):

    # Looping in plots
    for i, plot in enumerate(plots):
        # Set the plot id
        _id = i if id is None else id

        # Verify the plot existence
        if plot is None:
            return None

        if type(plot) is list:
            plot = plot[0]

        # Get the plot class name
        className = type(plot).__name__

        # Verify if axe is 3d
        triDim = '3D' in className  # plot.__class__

        # Vars possibilities
        vars = ['linewidth', 'linestyle', 'markerfacecolor', 'markeredgecolor',
                'alpha', 'zorder', 'markersize', 'cmap', 'markevery',
                'markeredgewidth', 'marker']

        # Get vars from kwargs
        (lw, ls, markerfacecolor, markeredgecolor, alpha, zorder,
         markersize, cmap, markevery, markeredgewidth, marker) \
            = map(kwargs.get, vars, len(vars)*[None])

        if ls is not None:
            _ls = ls[_id] \
                if type(ls) is list and _id < len(ls)\
                else (ls[-1]
                    if type(ls) is list
                    else ls)
            plot.set_linestyle(_ls)

        if lw is not None and className != 'TriContourSet':
            _lw = lw[_id] \
                if type(lw) is list and _id < len(lw)\
                else (lw[-1]
                    if type(lw) is list
                    else lw)
            plot.set_linewidth(_lw)

        if markerfacecolor is not None:
            _markerfacecolor = markerfacecolor[_id] \
                if type(markerfacecolor) is list and _id < len(markerfacecolor)\
                else (markerfacecolor[-1]
                    if type(markerfacecolor) is list
                    else markerfacecolor)
            if not hasattr(plot, 'cmap'):
                plot.set_markerfacecolor(_markerfacecolor)

        if markeredgecolor is not None:
            _markeredgecolor = markeredgecolor[_id] \
                if type(markeredgecolor) is list and _id < len(markeredgecolor)\
                else (markeredgecolor[-1]
                    if type(markeredgecolor) is list
                    else markeredgecolor)

            if hasattr(plot, 'set_markeredgecolor'):
                plot.set_markeredgecolor(_markeredgecolor)

        if alpha is not None:
            _alpha = alpha[_id] \
                if type(alpha) is list and _id < len(alpha)\
                else (alpha[-1]
                    if type(alpha) is list
                    else alpha)
            plot.set_alpha(_alpha)

        if zorder is not None:
            _zorder = zorder[_id] \
                if type(zorder) is list and _id < len(zorder)\
                else (zorder[-1]
                    if type(zorder) is list
                    else zorder)

            plot.set_zorder(_zorder)

        if markersize is not None:
            _markersize = markersize[_id] \
                if type(markersize) is list and _id < len(markersize)\
                else (markersize[-1]
                    if type(markersize) is list
                    else markersize)

            if hasattr(plot, 'set_markersize'):
                plot.set_markersize(_markersize)

            elif hasattr(plot, 'set_sizes'):
                plot.set_sizes([_markersize])

        if cmap is not None:
            _cmap = cmap[_id] \
                if type(cmap) is list and _id < len(cmap)\
                else (cmap[-1]
                    if type(cmap) is list
                    else cmap)

            if hasattr(plot, 'set_cmap'):
                plot.set_cmap(_cmap)

        if markevery is not None and not triDim:
            _markevery = markevery[_id] \
                if type(markevery) is list and _id < len(markevery)\
                else (markevery[-1]
                    if type(markevery) is list
                    else markevery)

            if hasattr(plot, 'set_markevery'):
                plot.set_markevery(_markevery)

        if markeredgewidth is not None:
            _markeredgewidth = markeredgewidth[_id] \
                if type(markeredgewidth) is list and _id < len(markeredgewidth)\
                else (markeredgewidth[-1]
                    if type(markeredgewidth) is list
                    else markeredgewidth)

            if hasattr(plot, 'set_markeredgewidth'):
                plot.set_markeredgewidth(_markeredgewidth)

        if marker is not None:
            _marker = marker[_id] \
                if type(marker) is list and _id < len(marker)\
                else (marker[-1]
                    if type(marker) is list
                    else marker)

            if hasattr(plot, 'set_marker'):
                plot.set_marker(_marker)

    return None


def resizeTicksFont(fig, axeId: [int, str] = 'all', fontSize: int = 12):
    'Reset the font size of the respective figure axe ticks labels'

    # Resize font size of the tick labels
    resize = lambda tick: tick.label.set_fontsize(fontSize)

    # Ideintify the respective axe
    if 'Axes' in type(fig).__name__:
        # Get the the axe given ("fig" is a axe in this case)
        axes = [fig]
    else:
        # Adjust the axe number
        axeId = axeId if axeId != 'all' else slice(0, len(fig.axes))

        # Get the axe from figure
        axes = fig.axes[axeId]

    # Looping in axes
    for ax in axes:
        # Get the xtick label
        xticks = ax.xaxis.get_major_ticks()

        # Verify if axe ticks is not empty
        if any(xticks):
            # Resize the x ticks
            # resize(xticks)
            list(map(resize, xticks))

            # Get the xtick label
            yticks = ax.yaxis.get_major_ticks()

            # Resize the y ticks
            # resize(yticks)
            list(map(resize, yticks))

            # Resize the z ticks
            if '3D' in type(ax).__name__:
                # Get the xtick label
                zticks = ax.zaxis.get_major_ticks()

                #resize(ax.zaxis.get_major_ticks())
                list(map(resize, zticks))

    return None


def getLegendAnchor(position: str) -> (tuple[tuple[float], str]):
    output = {
        'upper right': ((1.04, 1), 'upper left', 1),
        'lower right': ((1.04, 0), 'lower left', 1),
        'upper left': ((-0.1, 1), 'upper right', 1),
        'lower left': ((-0.1, 0), 'lower right', 1),
        'upper center': ((0.5, 1), 'lower center', 10),
        'lower center': ((0.5, -0.1), 'upper center', 10),
        }[position]
    return output


def getScreenInfo(monitor=0):
    'Return the screen size of the respective monitor'

    # Import respective module
    import screeninfo

    # Get principal monitor
    monitor = screeninfo.get_monitors()[0]

    # Get the monitor info
    width, height = monitor.width, monitor.height

    return width, height


def getCurrentFrame(ani):
    ani.pause()
    try:
        currentFrame = next(ani.frame_seq)
    except StopIteration:
        currentFrame = 0
    return currentFrame


def updateVerts3D(p, v, meshCoord):
    # Get the plot type
    pType = p.__class__.__name__

    # Get current Xs, Ys and Zs
    vec = eval({
        'Poly3DCollection': 'p._vec[:-1].T',
        'Line3DCollection': 'p._vec[:-1].T',
        }[pType])

    # Identify the element type
    element = 'rectangle'

    # Identify the element vertex numbering
    elementVerts = {
        'triangle': 3,
        'rectangle': 4
        }[element]

    # Set the convenently shape
    shape = (vec.shape[0]//elementVerts, elementVerts, 3)

    # Update the v_data
    vec = updateData(vec, v.flatten(), meshCoord)

    p.set_verts(vec.reshape(shape))

    return None


def updateData(vec, newData, meshCoord):
    # Get the transpose
    vec = vec.T

    # Get the plot coordinates
    vecCoord = array(list(zip(vec[0], vec[1])))

    # Get the current data
    currentData = vec[2]

    # Looping in mesh vertex coordinate
    for i, coord in enumerate(meshCoord):
        # Get the positions associated to each vertex
        vecArgs = matchValue(vecCoord, coord)

        # Replace the value corresponding to each vertex
        currentData[vecArgs] = newData[i]

    # Get the transpose
    vec = vec.T

    return vec


def matchValue(arr, v):
    positions = (arr == v).sum(axis=1) == 2
    return positions


def updateArray(arr1, arr2):
    # Make the update
    arr1[:] = arr2
    return None


def makePlot(ax: Axe, X, sol, alphas, colors, **kwargs) -> Plot:
    # Set the plot function
    plotFunc = ax.plot\
        if not is3d(ax)\
        else ax.scatter

    if len(X) < 3:
        # Make the plot
        line = plotFunc(*X, sol, **kwargs)

        if type(line) is list:
            # Get the from list
            line = line[0]

    else:
        # Make the plot
        line = plotFunc(*X, **kwargs)

        # Set the colors
        line.set_array(sol)

    if is3d(ax):
        # Set any plot parameters
        line.set_alpha(next(alphas)) if alphas is not None else None
        line.set_edgecolor(next(colors)) if colors is not None else None
        line.set_alpha(None)
        line.set_facecolor((0, 0, 0, 0))

    if not hasattr(line, '_offset_zordered'):
        # Handling a matplotlib problem
        line._offset_zordered = None

    return line


def generateSubplots(
    fig: Figure,
    tdim: int,
    multipleViews: bool = False,
    splitSols: int = 1
        ) -> list[Axe]:

    if splitSols in [0, None, False, True]:
        # Set a default value
        splitSols = 1

    if not multipleViews or tdim == 1:
        # Generate one axe
        axes = fig.add_subplot(label='Axe') \
            if tdim == 1 \
            else (fig.add_subplot(projection='3d', label='Axe')
                    if splitSols == 1
                    else [fig.add_subplot(
                        splitSols*100 + 11 + k,
                        projection='3d',
                        label=f'Axes_{k}'
                        ) for k in range(splitSols)])

    elif multipleViews:
        # Init the axes list
        axes = []

        # Looping in splitSols
        for k in range(splitSols):
            # Add a empty list
            axes.append([])

            # Looping in axes
            for i in range(3):
                # Create respective axe
                axes[k].append(
                    ax:=fig.add_subplot(
                        splitSols*100 + 31 + i + 3*k,
                        projection='3d',
                        label=f'Axe_{k}_{i}'
                        )
                    )

                # Set the axis labels
                ax.set_xlabel('$x$')
                ax.set_ylabel('$y$')
                ax.set_zlabel('$z$')

                # Set the view position parameters
                params = {
                    0: dict(azim=-90, elev=0), # XZ plan
                    1: dict(azim=0, elev=0), # YZ plan
                    2: dict(azim=-90, elev=90) # XY plan
                    }[i]

                # Set the axe view position
                ax.view_init(**params)

                # Adjust the excedent axis
                if i == 0:
                    ax.set_yticks([])
                    ax.set_ylabel('')
                    ax.set_xlabel('$x$', labelpad=3)

                elif i == 1:
                    ax.set_xticks([])
                    ax.set_xlabel('')
                    ax.set_ylabel('$y$', labelpad=3)

                elif i == 2:
                    ax.set_zticks([])
                    ax.set_zlabel('')
                    ax.set_xlabel('$x$', labelpad=20)

            # Set the middle axe as the main
            axes[k] = axes[k][1:2] + axes[k][0::2]

    if splitSols == 1 and type(axes) is list:
        # Get the unique list
        axes = axes[0]

    return axes


def figure(label: str = None, **kwargs):
    return plt.figure(label, **kwargs)


def imageTypes() -> (list[str]):
    return ['png', 'jpeg', 'jpg', 'eps']


def show(
    output: str = None,
    type: str = 'png',
    dpi: int = 150,
    grid: bool = True,
    label: str = '',
    props: dict[str:Any] = None
        ) -> (None):

    # Looping in figures number
    for figNum in plt.get_fignums():
        # Get the figure axes
        axes = plt.figure(figNum).axes

        if grid:
            # Add grid to respective axes
            addGrid(axes)

        if props is not None:
            # Update the axes props
            updateAxProps(*axes, **props)

    if output is None:
        return plt.show()

    # Join the output and extension append a string formatter
    # to consider the fignums
    outputName = '_%02d.'.join([output, type])

    if any(label):
        # Add label to output filename
        outputName = outputName.replace('_', f'_{label}_')

    if type in imageTypes():
        # Get the figures amount
        figAmount = len(plt.get_fignums())

        # Looping in all created figures
        for figNum in plt.get_fignums():
            # Get the respective figure
            fig = plt.figure(figNum, dpi=dpi)

            # Save the figure
            plt.savefig(outputName % figNum)

            # Rotate axes to YZ plan view
            rotations = [ax.view_init(0, 0) for ax in fig.axes if is3d(ax)]

            if any(rotations):
                # Save the figure with rotated axes
                plt.savefig(outputName % (figNum+figAmount))

        # Set the output name format
        outputName = outputName.replace('%02d', '*')

        # Print the figure savement
        print(
            '\Figure {} saved on {}'.format(
            basename(outputName),
            dirname(outputName)
            )
        )

    elif type == 'tex' and isInstalled(tikzplotlib):
        # Save the tikz tex files
        [(plt.figure(figNum),
          tikzplotlib.save(outputName % figNum))
            for figNum in plt.get_fignums()]

        # Set the output name format
        outputName = outputName.replace('%02d', '*')

        # Print the tex generation
        print(
            '\nTex file {} generated on {}'.format(
            basename(outputName),
            dirname(outputName)
            )
        )

    return None


def enableFocalMode(*figs):

    '''Connect all axes movements in each figures'''

    # Get all the axes
    axes = []
    [axes.extend(fig.axes) for fig in figs]
    axes = [ax for ax in axes if hasattr(ax, 'view_init')]

    def on_focal(event, fig):

        if event.key == 'control':
            # Change the all figure axes perspective
            [(ax.set_proj_type('persp', focal_length=0.1),
              ax.figure.canvas.draw())
                for ax in axes]

        return None

    # Looping in figures
    for fig in figs:
        # Enable the event handling
        fig.canvas.mpl_connect('key_press_event', lambda e: on_focal(e, fig))

    return None


def disableFocalMode(*figs):

    '''Connect all axes movements in each figures'''

    # Get all the axes
    axes = []
    [axes.extend(fig.axes) for fig in figs]
    axes = [ax for ax in axes if hasattr(ax, 'view_init')]

    def off_focal(event, fig):

        if event.key == 'control':
            # Change the all figure axes perspective
            [(ax.set_proj_type('ortho'),
              ax.figure.canvas.draw())
                for ax in axes]

        return None

    # Looping in figures
    for fig in figs:
        # Enable the event handling
        fig.canvas.mpl_connect('key_release_event', lambda e: off_focal(e, fig))

    return None


def connectAxesMovements(*figs):
    '''Connect all axes movements in each figures'''

    # Get all the axes
    axes = []
    [axes.extend(fig.axes) for fig in figs]
    axes = [ax for ax in axes if hasattr(ax, 'view_init')]

    def on_move(event, fig):
        # Get the current axe
        currentAx = event.inaxes

        if currentAx is not None:

            # Handling the rotate action
            if hasattr(currentAx, 'button_pressed') \
                    and currentAx.button_pressed in currentAx._rotate_btn:
                [ax.view_init(elev=currentAx.elev, azim=currentAx.azim)
                    for ax in axes]

                # Update the figure canvas
                fig.canvas.draw_idle()

            # # Handling the zoom action
            # elif currentAx.button_pressed in currentAx._zoom_btn:
            #     for ax in axes:
            #         ax.set_xlim(ax.get_xlim())
            #         ax.set_ylim(ax.get_ylim())
            #         ax.set_zlim(ax.get_zlim())

        return None

    # Looping in figures
    for fig in figs:
        # Enable the event handling
        fig.canvas.mpl_connect('motion_notify_event', lambda e: on_move(e, fig))

    return None


def title(
    msg: str,
    fig: Figure = None
        ) -> (None):

    plt.title(msg)\
        if fig is None\
        else fig.suptitle(msg)

    return None


def getColorValues(
    col: Collection
        ) -> (Array):

    if hasattr(col, 'cvalues'):
        cValues = col.cvalues()
    elif hasattr(col, 'get_array'):
        cValues = col.get_array()
    elif hasattr(col, 'get_clim'):
        cValues = col.get_clim()

    return cValues


def getLabel(
    ax: Axe
        ) -> (str):

    if any(ax.collections):
        label = ax.collections[0].get_label()
    elif any(ax.lines):
        label = ax.lines[0].get_label()
    else:
        label = None

    return label


def getCenterNode(
    nodes: Union[Array, list]
        ) -> (list):

    if type(nodes) is not Array:
        # Turn to array
        nodes = array(nodes)

    # Get the midpoint in each column
    center = [
        getMidPoint(nodes[:, i])
            for i in range(nodes.shape[1])
    ]

    return center


def setPltStyle(style: str) -> (None):
    return plt.style.use(style)


def settingColorbar(
    cbar: Colorbar,
    position: str = 'bottom',
    scale: float = 1,
    tickfont: int = 10,
    label: str = None,
    cmap: Color = matplotlib.cm.viridis,
    formatter: Union[str, Function] = '{x:1.04e}',
        ) -> (None):

    # Get cbar axe
    ax = cbar.ax

    testLoop(globals(), locals())

    # Set the colorbar location
    if 'upper' in position or position == 'top':
        location = 'top'
    elif 'lower' in position or position == 'bottom':
        location = 'bottom'
    elif 'left' in position:
        location = 'left'
    elif 'right' in position:
        location = 'right'

    # Set the colorbar positioner
    cax = colorbarPositioner(ax, scale, position)\
        if 'left' in position or 'right' in position\
        else None

    # Plot the colorbar
    cbar = ax.figure.colorbar(
        ScalarMappable(
            norm=matplotlib.colors.Normalize(*vlim, clip=False),
            cmap=cmap
            ),
        ax=ax, cax=cax, location=location,
        ticks=np.linspace(*vlim, 6)
            if vRange is not None
            else None
        )

    # Adjust the font size of the colorbar
    cbar.ax.tick_params(labelsize=tickfont)

    if formatter is not None:
        # Format the ticks labels
        cbar.ax.yaxis.set_major_formatter(formatter)

    # Set the colorbar label
    cbar.set_label(
        '$\mathbf{%s}$' % label.strip('$')
            if label is not None
            else ''
        )

    # Adjust the colorbar label
    cbar.ax.set_ylabel(
        cbar.ax.get_ylabel(),
        rotation='horizontal',
        labelpad=10
        )

    # Set the colorbar edge color
    cbar.outline.set_edgecolor('black')

    # Set the tick labels color
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

    return None


def addGrid(
    fig_ax: Union[Figure, Axe],
    x: bool = True,
    y: bool = True
        ) -> (None):

    if type(fig_ax) is Figure:
        # Get axes from figure
        axes = fig_ax.axes

    elif type(fig_ax) not in [list, tuple, Array]:
        # Put axe in list
        axes = [fig_ax]

    else:
        axes = fig_ax

    # Set the grid props
    props = {
        'axis': 'both' if x and y
                       else x*'x' + y*'y',
        'linestyle': '-'
    }

    # Looping in respective figure axes
    for ax in axes:
        # Put grid on background
        ax.grid(**props)

    return None


def updateAxProps(
    *axes: Axe,
    **props: dict[str: Any]
        ) -> (None):

    # Update the my personal properties
    updateMyAxProps(*axes, **props)

    # Remove props that don't apply to the axes
    props = popNoProps(axes, **props)

    # Looping in axes
    for ax in axes:
        # Update the proprieties in respective axe
        ax.set(**props)

    return None


def popNoProps(
    ax: Axe,
    **props: dict[str: Any]
        ) -> (dict[str: Any]):

    # Get a props copy
    newProps = props.copy()

    # Looping in proprieties
    for prop in props.keys():
        # Verify if is a axe attribute
        if not hasattr(ax, prop):
            # Remove the propriety
            newProps.pop(prop)

    return newProps


def updateMyAxProps(
    *axes: Axe,
    **props: dict[str: Any]
        ) -> (None):

    # Get the personal properties
    ticksFont = props.get('ticksFont', None)
    labelsFont = props.get('labelsFont', None)
    noTitles = props.get('noTitles', None)
    noColorbarLabel = props.get('noColorbarLabel', None)

    # Looping in axes
    for ax in axes:

        if noColorbarLabel is not None and noColorbarLabel\
                and any(ax.get_xlabel()) and not any(ax.get_ylabel()):
            # Remove the colorbar label
            ax.set_xlabel(None)

        if noTitles is not None and noTitles:
            # Remove the figure suptitle
            ax.figure.suptitle(None)

            # Remove the axe title
            ax.set_title(None)

        if ticksFont is not None:
            # Set the ticks labels
            ticksLabels = ax.get_xticklabels() + ax.get_yticklabels()

            if hasattr(ax, 'get_zticklabels'):
                ticksLabels += ax.get_zticklabels()

            # Looping in ticks label
            for label in ticksLabels:
                # Set the ticks label font
                label.set_fontproperties(
                    font_manager.FontProperties(
                        size=ticksFont
                    )
                )

        if labelsFont is not None:
            # Set the axis label font
            ax.xaxis.label.set_size(labelsFont)
            ax.yaxis.label.set_size(labelsFont)

        # # Fit the figure layout
        # ax.figure.tight_layout()

    return None


def setFigAxesAspectRatio(
    *figs: Figure
        ) -> (None):

    # Check the args that are list
    areList = filter(lambda i: hasattr(figs[i], 'len'), range(len(figs)))

    if any(areList):
        # Flat the args that are list
        _figs = np.array(figs)[set(range(len(figs))) - set(areList)]
        [_figs.extend(figs[i]) for i in areList]
        figs = _figs

    # Looping in figures
    for fig in figs:
        # Looping in figure's axes
        for ax in fig.axes:
            if 'colorbar' not in ax.get_label():
                # Set the respective axe aspect ratio
                ax.set_box_aspect(getBestAspect(ax))

    return None
