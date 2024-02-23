### Import Statements


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from ..model.node import PickupNode, DropoffNode, Node, NodeInfo, NodeType
from ..solution.Solution import SolutionNode, Solution
from src.data.instance import Instance
import seaborn as sns
import pandas as pd


### Constant Colors

node_color = {
    NodeType.D_DEPOT: "k",
    NodeType.O_DEPOT: "k",
    NodeType.DO: "r",
    NodeType.PU: "g",
}


#### Get Colormap


def get_cmap(n, name="Set1"):
    """
    Get a colormap function.
    """
    return plt.cm.get_cmap(name, n)


### Plot Functions

#### Plot Arrows


def plot_arrows(
    axis, nodes: list[NodeInfo], route_color, arrowstyle, linestyle, linewidth
):
    """
    Plot arrows between nodes.
    """
    for p, d in zip(nodes[:-1], nodes[1:]):
        arrow = patches.FancyArrowPatch(
            p.xy_coord,
            d.xy_coord,
            edgecolor=route_color,
            facecolor=route_color,
            arrowstyle=arrowstyle,
            linestyle=linestyle,
            linewidth=linewidth,
            mutation_scale=10,
        )
        axis.add_artist(arrow)


#### Plot Line Collection


def plot_line_collection(
    axis, node_xy_coords, route_color, linestyle, linewidth
):
    """
    Plot a line collection.
    """
    lc_vehicle = mc.LineCollection(
        [node_xy_coords],
        linewidths=linewidth,
        linestyles=linestyle,
        edgecolors=route_color,
    )
    axis.add_collection(lc_vehicle)


#### Plot Nodes


def plot_nodes(axis, node_xy_coords, node_colors, node_types):
    """
    Plot nodes on the axis.
    """
    x, y = zip(*node_xy_coords)
    # TODO This part will brack for vehicles that do not leave the depot
    axis.scatter(x[1:-1], y[1:-1], color=node_colors[1:-1], marker="o", s=15)
    axis.scatter(x[0], y[0], color="k", marker="s", s=15)

    # data = pd.DataFrame(
    #     dict(x=x,
    #     y=y,
    #     colors=node_colors,
    #     markers=node_types))

    # print(data)
    # sns.scatterplot(data=data, x='x', y='y', hue='colors', ax=axis)


#### Plot Node Labels


def plot_node_labels(axis, node_labels, ignore_depot=True):
    """
    Plot labels for nodes.
    """
    labels = node_labels[1:-1] if ignore_depot else node_labels
    for label, xy in labels:
        axis.annotate(label, xy=xy, fontsize=9, xytext=np.array(xy) + 0.05)


#### Set Axis Limits


def set_axis_limits(axis, coord_box, x, y):
    """
    Set limits for the axis.
    """
    if coord_box:
        axis.set_xlim(coord_box[0], coord_box[1])
        axis.set_ylim(coord_box[2], coord_box[3])
    else:
        axis.set_xlim(min(x), max(x))
        axis.set_ylim(min(y), max(y))


#### Plot Legend


def plot_legend(axis):
    """
    Plot a custom legend.
    """
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="Depot",
            markerfacecolor="k",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Pickup",
            markerfacecolor="g",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Drop off",
            markerfacecolor="r",
            markersize=10,
        ),
    ]
    axis.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        fancybox=False,
        frameon=False,
        shadow=False,
        ncol=3,
        fontsize="small",
    )
    plt.subplots_adjust(bottom=0.1)


### Main Plotting Functions

#### Plot Vehicle Route


def plot_vehicle_route(
    axis,
    visits: list[SolutionNode],
    node_info: list[NodeInfo],
    route_color="k",
    coord_box=(-10, 10, -10, 10),
    show_arrows=True,
    show_nodes=True,
    show_node_labels=True,
    arrowstyle="-|>",
    linestyle="-",
    linewidth=0.5,
    show_legend=True,
    x_title="x",
    y_title="y",
    tw=False,
    title=None,
):
    """
    Plot a single vehicle route.
    """
    (
        node_xy_coords,
        node_colors,
        nodes,
        node_labels,
        node_types,
    ) = process_visits(visits, node_info, tw=tw)

    if show_arrows:
        plot_arrows(axis, nodes, route_color, arrowstyle, linestyle, linewidth)
    else:
        plot_line_collection(
            axis, node_xy_coords, route_color, linestyle, linewidth
        )

    if show_nodes:
        plot_nodes(axis, node_xy_coords, node_colors, node_types)
    if show_node_labels:
        plot_node_labels(axis, node_labels, ignore_depot=True)

    axis.set_xlabel(x_title)
    axis.set_ylabel(y_title)
    set_axis_limits(axis, coord_box, *zip(*node_xy_coords))

    if show_legend:
        plot_legend(axis)
    if title:
        axis.set_title(title)


#### Process Visits


def process_visits(
    visits: list[SolutionNode], node_info: list[NodeInfo], tw=False
):
    """
    Process visit nodes.
    """
    node_xy_coords = []
    node_colors = []
    nodes = []
    node_labels = []
    node_types = []

    for n in visits:
        node = node_info[n.id]
        nodes.append(node)
        node_colors.append(node_color[node.type])

        tw_str = (
            f"({node.tw.earliest}/{round(n.b,1)}/{node.tw.latest})"
            if tw
            else ""
        )
        label = f"{node.alias}{tw_str}"
        node_labels.append((label, node.xy_coord))
        node_types.append(
            "s" if node.type in (NodeType.O_DEPOT, NodeType.D_DEPOT) else "o"
        )
        node_xy_coords.append(node.xy_coord)

    return node_xy_coords, node_colors, nodes, node_labels, node_types


#### Plot Vehicle Routes


def plot_vehicle_routes(
    instance,
    solution,
    jointly=False,
    coord_box=(-10, 10, -10, 10),
    figsize=(10, 100),
    show_arrows=True,
    show_nodes=True,
    show_node_labels=True,
    arrowstyle="->",
    linestyle="--",
    linewidth=1,
):
    """
    Plot routes for multiple vehicles.
    """
    n_plots, ax = setup_plots(jointly, len(instance.vehicles), figsize)
    plot_multiple_routes(
        ax,
        instance,
        solution,
        jointly,
        coord_box,
        show_arrows,
        show_nodes,
        show_node_labels,
        arrowstyle,
        linestyle,
        linewidth,
    )
    return n_plots, ax


#### Setup Plots


def setup_plots(jointly, n_vehicles, figsize):
    """
    Setup plot figures and axes.
    """
    n_plots = 1 if jointly else n_vehicles
    fig, ax = plt.subplots(n_plots, figsize=figsize, squeeze=False)
    return n_plots, ax.flatten()


#### Plot Multiple Routes


def plot_multiple_routes(
    ax,
    instance: Instance,
    solution: Solution,
    jointly,
    coord_box,
    show_arrows,
    show_nodes,
    show_node_labels,
    arrowstyle,
    linestyle,
    linewidth,
):
    """
    Plot routes for multiple vehicles.
    """
    v_nodes = {v.id: v.route for v in solution.vehicle_routes}
    cmap = get_cmap(len(v_nodes))

    for i, (v, visits) in enumerate(v_nodes.items()):
        axis = ax[0] if jointly else ax[i]
        v_color = cmap(i)
        if not jointly:
            axis.set_title(instance.vehicle_id_dict[v].alias)
        plot_vehicle_route(
            axis,
            visits,
            instance.nodes,
            route_color=v_color,
            coord_box=coord_box,
            show_arrows=show_arrows,
            show_nodes=show_nodes,
            show_node_labels=show_node_labels,
            arrowstyle=arrowstyle,
            linestyle=linestyle,
            linewidth=linewidth,
            show_legend=False,
        )

    if jointly:
        add_joint_legend(ax[0], v_nodes.keys(), cmap, instance)


#### Add Joint Legend


def add_joint_legend(axis, vehicle_ids, cmap, instance):
    """
    Add a joint legend for all vehicles.
    """
    legends = [
        patches.Patch(color=cmap(i), label=instance.vehicle_id_dict[v].alias)
        for i, v in enumerate(vehicle_ids)
    ]
    axis.legend(handles=legends, frameon=False, loc="lower center", ncol=5)
