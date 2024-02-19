import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections  as mc
from ..model.Node import PickupNode, DropoffNode
import matplotlib.patches as patches

COLOR_PICKUP = 'green'
COLOR_DELIVERY = 'red'
COLOR_ORIGIN = 'black'

def node_color(n):
    if type(n) == DropoffNode: return COLOR_DELIVERY
    elif type(n) == PickupNode: return COLOR_PICKUP
    else: return COLOR_ORIGIN
    

def get_cmap(n, name='Set1'):
    '''
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.
    
    https://matplotlib.org/stable/tutorials/colors/colormaps.html
    
    '''
    return plt.cm.get_cmap(name, n)

def plot_vehicle_route(
    axis,
    visits,
    node_id_dict,
    route_color='k',
    coord_box=(-10,10,-10,10),
    show_arrows=True,
    show_nodes=True,
    show_node_labels=True,
    arrowstyle='->',
    linestyle='--',
    linewidth=1):
    
    node_xy_coords = []
    node_colors = []
    node_aliases = []
    nodes = []

    for n in visits:
        try:
            node = node_id_dict[n.id]
        except:
            node = node_id_dict[int(n.id.replace("*",""))]
        nodes.append(node)
        node_colors.append(node_color(node))
        node_xy_coords.append(node.xy_coord)
        node_aliases.append(node.alias)

    if show_arrows:
        for p, d in zip(nodes[:-1], nodes[1:]):

            arrow = patches.FancyArrowPatch(
                p.xy_coord,
                d.xy_coord,
                edgecolor=route_color,
                facecolor=route_color,
                arrowstyle=arrowstyle,
                linestyle=linestyle,
                mutation_scale=10)
            axis.add_artist(arrow)
    else:
        # Plot vehicle route
        lc_vehicle = mc.LineCollection(
            [node_xy_coords],
            linewidths=linewidth,
            linestyles=linestyle,
            edgecolors=route_color,
            facecolors=route_color
        )

        axis.add_collection(lc_vehicle)
        
    
    # Plot pickup and delivery nodes
    x, y = zip(*node_xy_coords)
    
    if show_nodes:
        axis.scatter(x, y, color=node_colors, marker='o', s=10)

    if show_node_labels:
        for n in nodes:
            axis.annotate(
                xy=n.xy_coord,
                text=n.alias,
                fontsize=9,
                xytext=np.array(n.xy_coord)+0.05)
    
    if coord_box:
        min_x, max_x, min_y, max_y = coord_box
        axis.set_xlim(min_x, max_x)
        axis.set_ylim(min_y, max_y)
    else:
        axis.set_xlim(min(x), max(x))
        axis.set_ylim(min(y), max(y))
    
def plot_vehicle_routes(
    instance,
    solution,
    jointly=False,
    coord_box=(-10,10,-10,10),
    figsize=(10, 100),
    show_arrows=True,
    show_nodes=True,
    show_node_labels=True,
    arrowstyle='->',
    linestyle='--',
    linewidth=1):
    
    
    if jointly:
        n_plots =  1
    else:
        n_plots = len(instance.vehicles)
    

    fig, ax = plt.subplots(
        n_plots,
        figsize=figsize)

    v_nodes=dict()
    for v in solution.vehicle_routes:
        v_nodes[v.id] = v.visits

    cmap = get_cmap(len(v_nodes))
    
    v_ids = []
    v_colors = []
    for i, (v, visits) in enumerate(v_nodes.items()):
        v_id = instance.vehicle_id_dict[v].alias
        if jointly:
            axis = ax
            v_ids.append(v_id)
            
        else:
            axis = ax[i]
            axis.set_title(v_id)
        v_color = cmap(i)
        v_colors.append(v_color)
        plot_vehicle_route(axis, visits, instance.node_id_dict, route_color=v_color)
        

    if jointly:
        legends = [patches.Patch(color=v_color, label=v_id)
                   for v_id, v_color in zip(v_ids, v_colors)]
        axis.legend(handles=legends, frameon=False, loc='lower center', ncol=5)
        
    return fig, ax