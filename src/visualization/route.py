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

node_features = {
    NodeType.D_DEPOT.name: {"color": "k", "marker": "s", "size": 100, "font-color": "w", "edgecolor": "w"},
    NodeType.O_DEPOT.name: {"color": "k", "marker": "s", "size": 100, "font-color": "w", "edgecolor": "w"},
    NodeType.DO.name: {"color": "k", "marker": "o", "font-color": "w", "edgecolor": "w", "size": 150},
    NodeType.PU.name: {"color": "w", "marker": "o", "font-color": "k", "edgecolor": "k", "size": 150},
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
    axis, df: pd.DataFrame, route_color, arrowstyle, linestyle, linewidth
):
    """
    Plot arrows between nodes using DataFrame.
    """
    for idx in range(len(df) - 1):
        p = np.array(df.iloc[idx][["x", "y"]].to_list())
        d = np.array(df.iloc[idx + 1][["x", "y"]].to_list())
        
        # Circle radius
        radius = 0.8  # Half of your diameter

        # Determine if the line is horizontal or vertical
        horizontal_line = p[1] == d[1]
        vertical_line = p[0] == d[0]

        if horizontal_line:
            if d[0] > p[0]:
                p_adjusted = [p[0] + radius, p[1]]
                d_adjusted = [d[0] - radius, d[1]]
            else:
                p_adjusted = [p[0] - radius, p[1]]
                d_adjusted = [d[0] + radius, d[1]]
        elif vertical_line:
            if d[1] > p[1]:
                p_adjusted = [p[0], p[1] + radius]
                d_adjusted = [d[0], d[1] - radius]
            else:
                p_adjusted = [p[0], p[1] - radius]
                d_adjusted = [d[0], d[1] + radius]
        else:
            # For non-strictly horizontal/vertical lines, adjust as before
            direction = d - p
            norm_direction = direction / np.linalg.norm(direction)
            p_adjusted = p + norm_direction * radius
            d_adjusted = d - norm_direction * radius
        
        if not np.array_equal(p, d):
            arrow = patches.FancyArrowPatch(
                p_adjusted,
                d_adjusted,
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
    axis, df: pd.DataFrame, route_color, linestyle, linewidth
):
    """
    Plot a line collection using DataFrame.
    """

    lc_vehicle = mc.LineCollection(
        [df[["x", "y"]].values.reshape(-1, 2)],
        linewidths=linewidth,
        linestyles=linestyle,
        edgecolors=route_color,
    )
    axis.add_collection(lc_vehicle)


#### Plot Nodes


def plot_nodes(axis, df, size_node=150):
    """
    Plot nodes on the axis using DataFrame.
    """
    # Depot nodes
    depot_df = df[
        df["node_type"].isin([NodeType.O_DEPOT.name, NodeType.D_DEPOT.name])
    ]
    axis.scatter(
        depot_df["x"],
        depot_df["y"],
        color=node_features[NodeType.O_DEPOT.name]["color"],
        marker=node_features[NodeType.O_DEPOT.name]["marker"],
        s=node_features[NodeType.O_DEPOT.name].get("size", size_node),
    )

    # Pickup nodes
    pu_df = df[df["node_type"] == NodeType.PU.name]
    axis.scatter(
        pu_df["x"],
        pu_df["y"],
        color=node_features[NodeType.PU.name]["color"],
        marker=node_features[NodeType.PU.name]["marker"],
        edgecolors=node_features[NodeType.PU.name]["edgecolor"],
        s=node_features[NodeType.PU.name].get("size", size_node),
    )

    # Dropoff nodes
    du_df = df[df["node_type"] == NodeType.DO.name]
    axis.scatter(
        du_df["x"],
        du_df["y"],
        color=node_features[NodeType.DO.name]["color"],
        marker=node_features[NodeType.DO.name]["marker"],
        edgecolors=node_features[NodeType.DO.name]["edgecolor"],
        s=node_features[NodeType.DO.name].get("size", size_node),
    )


#### Plot Node Labels


def plot_node_labels(axis, df, ignore_depot=True, fontsize=7):
    """
    Plot labels for nodes using DataFrame.
    """
    if ignore_depot:
        df = df[
            ~df["node_type"].isin(
                [NodeType.O_DEPOT.name]
            )
        ]

    for _, row in df.iterrows():
        xy = row[["x", "y"]].to_list()
         # Add text annotations for the nodes
    for _, row in df.iterrows():
        axis.text(
            row['x'],
            row['y'],
            str(row['alias']),
            fontsize=fontsize,
            ha='center',
            va='center',
            fontfamily='Consolas',
            color=node_features[row["node_type"]]["font-color"]
        )
        


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
            marker=node_features[NodeType.O_DEPOT.name]["marker"],
            color="w",
            label="Depot",
            markerfacecolor=node_features[NodeType.O_DEPOT.name]["color"],
            markersize=10,
            markeredgecolor="k", 
        ),
        Line2D(
            [0],
            [0],
            marker=node_features[NodeType.PU.name]["marker"],
            color="w",
            label="Pickup",
            markerfacecolor=node_features[NodeType.PU.name]["color"],
            markersize=10,
            markeredgecolor=node_features[NodeType.PU.name]["edgecolor"],
        ),
        Line2D(
            [0],
            [0],
            marker=node_features[NodeType.DO.name]["marker"],
            color="w",
            label="Drop off",
            markerfacecolor=node_features[NodeType.DO.name]["color"],
            markersize=10,
            markeredgecolor=node_features[NodeType.DO.name]["edgecolor"],
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
    df,
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
    title=None,
    size_node=150,
    fontsize=7
):
    """
    Plot a single vehicle route using DataFrame.
    """

    if not axis:
        fig, axis = plt.subplots()

    if show_arrows:
        plot_arrows(axis, df, route_color, arrowstyle, linestyle, linewidth)
    else:
        plot_line_collection(axis, df, route_color, linestyle, linewidth)

    if show_nodes:
        plot_nodes(axis, df, size_node = size_node)
    if show_node_labels:
        plot_node_labels(axis, df, ignore_depot=False, fontsize=fontsize)

    axis.set_xlabel(x_title)
    axis.set_ylabel(y_title)
    set_axis_limits(axis, coord_box, df["x"], df["y"])

    if show_legend:
        plot_legend(axis)
    if title:
        axis.set_title(title)


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
