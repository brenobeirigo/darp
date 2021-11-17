from model.Request import Request
from model.Vehicle import Vehicle
from model.Node import Node

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

        self.dist_matrix = dist_matrix
    
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