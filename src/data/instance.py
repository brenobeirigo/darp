from ..model.Request import Request
from ..model.Vehicle import Vehicle
from ..model.node import Node, OriginNode, PickupNode, DestinationNode, DropoffNode, NodeInfo
import pandas as pd
from dataclasses import dataclass

@dataclass
class InstanceConfig:
    n_vehicles: int
    n_customers: int
    time_horizon_min: int
    vehicle_capacity: int
    maximum_ride_time_min: int
  
class Instance:
    def __init__(
        self,
        vehicles:list[Vehicle],
        requests:list[Request],
        nodes:list[NodeInfo],
        config_dict:InstanceConfig,
        instance_filepath:str,
        instance_parser:str
    ):
        self.config_dict = config_dict
        self.vehicles = vehicles
        self.requests = requests
        self.nodes = nodes
        
        self.vehicle_id_dict = {v.id:v for v in self.vehicles}
        self.request_id_dict = {r.id:r for r in self.requests}
        # self.node_id_id_dict = {n.id:n.id for n in self.nodes}

        self.pickup_nodes = []
        self.dropoff_nodes = []
        self.destination_nodes = []
        self.origin_nodes = []
        
        for r in requests:
            self.pickup_nodes.append(r.pickup_node)
            self.dropoff_nodes.append(r.dropoff_node)
        for v in vehicles:
            self.destination_nodes.append(v.destination_node)
            self.origin_nodes.append(v.origin_node)
            
                # self.node_id_dict = {n.id:n for n in self.nodes}
        self.dist_matrix_id = {
            o.id: {
                d.id: o.point.distance(d.point)
                for d in self.nodes
            }
            for o in self.nodes
        }
        
        self.__del__()
    
    @property
    def nodeset_df(self):
        # Directly initialize DataFrame with specified data types
        columns = ["id", "alias", "x", "y", "earliest", "latest", "service_duration"]
        data = [[n.id, n.alias, n.x, n.y, n.tw.earliest, n.tw.latest, n.service_duration] for n in self.nodes]
        dtype = {'id': 'int32', "earliest": "int32", "latest": "int32"}

        df = pd.DataFrame(data, columns=columns).astype(dtype)

        # Consider if setting and sorting by the index is necessary
        # If it is, keep this line
        return df.set_index("id").sort_index()
        
    def get_data(self):
        TOTAL_HORIZON = 1440
        return dict(
            origin_depot=self.nodes[0].id,
            K=[v.id for v in self.vehicles],
            Q={v.id: v.capacity for v in self.vehicles},
            P=[n.id for n in self.pickup_nodes],
            D=[n.id for n in self.dropoff_nodes],
            L={r.pickup_node.id: r.max_ride_time for r in self.requests},
            el={n.id: (n.tw.earliest, n.tw.latest) for n in self.nodes},
            d={n.id: n.service_duration for n in self.nodes},
            q={n.id: n.load for n in self.nodes},
            destination_depot=self.nodes[-1].id,
            dist_matrix=self.dist_matrix_id,
            total_horizon=TOTAL_HORIZON,
        )

    def __del__(self):
        '''Resets the id count for requests, vehicles, and nodes'''
        Request.cleanup()
        Vehicle.cleanup()
        Node.cleanup()

    def __str__(self) -> str:
        output = str(self.config_dict)
        output += "\n\n### Nodes:\n" + "\n".join(map(str, self.nodes))
        output += "\n\n### Requests:\n"
        output += "\n".join(map(str,self.requests))
        output += "\n\n### Vehicles:\n"
        output += "\n".join(map(str,self.vehicles))
        return output
    
    
    