### Import Statements


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from ..model.node import PickupNode, DropoffNode, Node


### Constant Colors


COLOR_PICKUP = 'green'
COLOR_DELIVERY = 'red'
COLOR_ORIGIN = 'black'


### Helper Functions

#### Node Color


def node_color(n):
    """
    Determine the color of a node.
    """
    if isinstance(n, DropoffNode):
        return COLOR_DELIVERY
    elif isinstance(n, PickupNode):
        return COLOR_PICKUP
    return COLOR_ORIGIN


#### Get Colormap


def get_cmap(n, name='Set1'):
    """
    Get a colormap function.
    """
    return plt.cm.get_cmap(name, n)


### Plot Functions

#### Plot Arrows


def plot_arrows(axis, nodes, route_color, arrowstyle, linestyle, linewidth):
    """
    Plot arrows between nodes.
    """
    for p, d in zip(nodes[:-1], nodes[1:]):
        arrow = patches.FancyArrowPatch(p.xy_coord, d.xy_coord, edgecolor=route_color,
                                        facecolor=route_color, arrowstyle=arrowstyle,
                                        linestyle=linestyle, linewidth=linewidth,
                                        mutation_scale=10)
        axis.add_artist(arrow)


#### Plot Line Collection


def plot_line_collection(axis, node_xy_coords, route_color, linestyle, linewidth):
    """
    Plot a line collection.
    """
    lc_vehicle = mc.LineCollection([node_xy_coords], linewidths=linewidth,
                                   linestyles=linestyle, edgecolors=route_color)
    axis.add_collection(lc_vehicle)


#### Plot Nodes


def plot_nodes(axis, node_xy_coords, node_colors):
    """
    Plot nodes on the axis.
    """
    x, y = zip(*node_xy_coords)
    axis.scatter(x, y, color=node_colors, marker='o', s=10)


#### Plot Node Labels


def plot_node_labels(axis, nodes:list[Node]):
    """
    Plot labels for nodes.
    """
    for n in nodes:
        axis.annotate(f"{n.alias}({n.tw.earliest}/{n.arrival}/{n.tw.latest})", xy=n.xy_coord, fontsize=9, xytext=np.array(n.xy_coord) + 0.05)


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
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Origin',
                              markerfacecolor='black', markersize=5),
                       Line2D([0], [0], marker='o', color='w', label='Pickup',
                              markerfacecolor='green', markersize=5),
                       Line2D([0], [0], marker='o', color='w', label='Drop off',
                              markerfacecolor='red', markersize=5)]
    axis.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                fancybox=False, shadow=False, ncol=3, fontsize='small')
    plt.subplots_adjust(bottom=0.1)


### Main Plotting Functions

#### Plot Vehicle Route


def plot_vehicle_route(axis, visits, node_id_dict:dict[int, Node], route_color='k', coord_box=(-10, 10, -10, 10), 
                       show_arrows=True, show_nodes=True, show_node_labels=True, 
                       arrowstyle='->', linestyle='--', linewidth=1, show_legend=True, 
                       x_title="x", y_title="y"):
    """
    Plot a single vehicle route.
    """
    node_xy_coords, node_colors, nodes = process_visits(visits, node_id_dict)

    if show_arrows:
        plot_arrows(axis, nodes, route_color, arrowstyle, linestyle, linewidth)
    else:
        plot_line_collection(axis, node_xy_coords, route_color, linestyle, linewidth)

    if show_nodes:
        plot_nodes(axis, node_xy_coords, node_colors)
    if show_node_labels:
        plot_node_labels(axis, nodes)

    axis.set_xlabel(x_title)
    axis.set_ylabel(y_title)
    set_axis_limits(axis, coord_box, *zip(*node_xy_coords))

    if show_legend:
        plot_legend(axis)


#### Process Visits


def process_visits(visits, node_id_dict):
    """
    Process visit nodes.
    """
    node_xy_coords = []
    node_colors = []
    nodes = []

    for n in visits:
        try:
            node = node_id_dict[n.id]
        except KeyError:
            node = node_id_dict[int(n.id.replace("*", ""))]
        nodes.append(node)
        node_colors.append(node_color(node))
        node_xy_coords.append(node.xy_coord)
    
    return node_xy_coords, node_colors, nodes


#### Plot Vehicle Routes


def plot_vehicle_routes(instance, solution, jointly=False, coord_box=(-10, 10, -10, 10),
                        figsize=(10, 100), show_arrows=True, show_nodes=True,
                        show_node_labels=True, arrowstyle='->', linestyle='--', linewidth=1):
    """
    Plot routes for multiple vehicles.
    """
    n_plots, ax = setup_plots(jointly, len(instance.vehicles), figsize)
    plot_multiple_routes(ax, instance, solution, jointly, coord_box, show_arrows,
                         show_nodes, show_node_labels, arrowstyle, linestyle, linewidth)
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


def plot_multiple_routes(ax, instance, solution, jointly, coord_box, show_arrows,
                         show_nodes, show_node_labels, arrowstyle, linestyle, linewidth):
    """
    Plot routes for multiple vehicles.
    """
    v_nodes = {v.id: v.visits for v in solution.vehicle_routes}
    cmap = get_cmap(len(v_nodes))

    for i, (v, visits) in enumerate(v_nodes.items()):
        axis = ax[0] if jointly else ax[i]
        v_color = cmap(i)
        if not jointly:
            axis.set_title(instance.vehicle_id_dict[v].alias)
        plot_vehicle_route(axis, visits, instance.node_id_dict, route_color=v_color,
                           coord_box=coord_box, show_arrows=show_arrows,
                           show_nodes=show_nodes, show_node_labels=show_node_labels,
                           linestyle=linestyle, linewidth=linewidth, show_legend=False)

    if jointly:
        add_joint_legend(ax[0], v_nodes.keys(), cmap, instance)


#### Add Joint Legend


def add_joint_legend(axis, vehicle_ids, cmap, instance):
    """
    Add a joint legend for all vehicles.
    """
    legends = [patches.Patch(color=cmap(i), label=instance.vehicle_id_dict[v].alias)
               for i, v in enumerate(vehicle_ids)]
    axis.legend(handles=legends, frameon=False, loc='lower center', ncol=5)
