from model.Request import Request
from model.Vehicle import Vehicle
from model.Node import Node, OriginNode, PickupNode, DestinationNode, DropoffNode

class Instance:
    def __init__(
        self, vehicles, requests, nodes, dist_matrix, config_dict=None
    ):
        self.config_dict = config_dict

        self.vehicles = vehicles
        self.vehicle_id_dict = {v.id:v for v in self.vehicles}

        self.requests = requests
        self.request_id_dict = {r.id:r for r in self.requests}

        self.nodes = nodes
        self.node_id_dict = {n.pos:n for n in self.nodes}
        
        self.node_id_pos_dict = {n.id:n.pos for n in self.nodes}
        
        self.pickup_nodes = []
        self.dropoff_nodes = []
        self.destination_nodes = []
        self.origin_nodes = []
        
        for n in self.nodes:
            if type(n) == PickupNode:
                self.pickup_nodes.append(n)
            elif type(n) == DropoffNode:
                self.dropoff_nodes.append(n)
            elif type(n) == DestinationNode:
                self.destination_nodes.append(n)
            elif type(n) == OriginNode:
                self.origin_nodes.append(n)
        
        dist = {}
        for o in self.node_id_pos_dict:
            dist[o] = {}
            o_pos = self.node_id_pos_dict[o]
            for d in self.node_id_pos_dict:
                d_pos = self.node_id_pos_dict[d]
                dist[o][d] = dist_matrix[o_pos][d_pos]
                    
        self.dist_matrix_id = dist
        self.dist_matrix = dist_matrix
        
        
    def get_data(self):
        TOTAL_HORIZON = 1440
        data = dict(
            origin_depot=self.vehicles[0].pos,
            K=[v.id for v in self.vehicles],
            Q={v.id:v.capacity for v in self.vehicles},
            P=[n.pos for n in self.pickup_nodes],
            D=[n.pos for n in self.dropoff_nodes],
            L={r.pickup_node.pos:r.max_ride_time for r in self.requests},
            el={n.pos:n.el for n in self.nodes},
            d={n.pos:n.service_delay for n in self.nodes},
            q={n.pos: n.load for n in self.nodes},
            dist_matrix=self.dist_matrix,
            total_horizon=TOTAL_HORIZON
        )
        
        return data

    def __del__(self):
        '''Resets the id count for requests, vehicles, and nodes'''
        Request.cleanup()
        Vehicle.cleanup()
        Node.cleanup()

    def __str__(self) -> str:
        output = "\n".join([f"{k} = {v}" for k, v in self.config_dict.items()])
        output += "\n\n### Nodes:\n" + "\n".join(map(str, self.nodes))
        output += "\n\n### Requests:\n"
        output += "\n".join(map(str,self.requests))
        output += "\n\n### Vehicles:\n"
        output += "\n".join(map(str,self.vehicles))
        return output