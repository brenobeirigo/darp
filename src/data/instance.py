from ..model.Request import Request
from ..model.Vehicle import Vehicle
from ..model.node import Node, NodeInfo
import pandas as pd
from dataclasses import dataclass


@dataclass
class InstanceConfig:
    n_vehicles: int
    n_customers: int
    vehicle_capacity: int
    maximum_driving_time_min: int = None
    time_horizon_min: int = None
    maximum_ride_time_min: int = None
    n_depots: int = 1

    @property
    def label(self):
        return f"Depots: {self.n_depots} / Vehicles: {self.n_vehicles} / Customers: {self.n_customers}"


class Instance:
    def __init__(
        self,
        vehicles: list[Vehicle],
        requests: list[Request],
        nodes: list[NodeInfo],
        config: InstanceConfig,
        instance_filepath: str,
        instance_parser: str,
    ):
        self.config = config
        self.vehicles = vehicles
        self.requests = requests
        self.nodes = nodes
        self.instance_filepath = instance_filepath

        self.vehicle_id_dict = {v.id: v for v in self.vehicles}
        self.request_id_dict = {r.id: r for r in self.requests}
        # self.node_id_id_dict = {n.id:n.id for n in self.nodes}

        self.origin_nodes = self.nodes[:self.config.n_depots]
        self.pickup_nodes = self.nodes[self.config.n_depots:self.config.n_depots+self.config.n_customers]
        self.dropoff_nodes = self.nodes[self.config.n_depots+self.config.n_customers:self.config.n_depots+2*self.config.n_customers]
        self.destination_nodes = self.nodes[self.config.n_depots+2*self.config.n_customers:]

        # for r in requests:
        #     self.pickup_nodes.append(r.pickup_node)
        #     self.dropoff_nodes.append(r.dropoff_node)
        # for v in vehicles:
        #     self.destination_nodes.append(v.destination_node)
        #     self.origin_nodes.append(v.origin_node)

            # self.node_id_dict = {n.id:n for n in self.nodes}
        self.dist_matrix_id = {
            o.id: {d.id: o.point.distance(d.point) for d in self.nodes}
            for o in self.nodes
        }

        self.__del__()

    @property
    def nodeset_df(self):
        # Directly initialize DataFrame with specified data types
        columns = [
            "id",
            "alias",
            "node_type",
            "x",
            "y",
            "earliest",
            "latest",
            "service_duration",
        ]
        data = [
            [
                n.id,
                n.alias,
                n.type.name,
                n.x,
                n.y,
                n.tw.earliest,
                n.tw.latest,
                n.service_duration,
            ]
            for n in self.nodes
        ]
        dtype = {"id": "int32", "earliest": "int32", "latest": "int32"}

        df = pd.DataFrame(data, columns=columns).astype(dtype)

        # Consider if setting and sorting by the index is necessary
        # If it is, keep this line
        return df.set_index("id").sort_index()

    def get_data(self):
        TOTAL_HORIZON = 1440
        data = dict(
            origin_depot=[n.id for n in self.origin_nodes],
            K=[v.id for v in self.vehicles],
            K_params = {
                v.id:dict(cost_per_min=v.cost_per_min,
                          cost_per_km=v.cost_per_km,
                          speed_km_h=v.speed_km_h,
                          revenue_per_load_unit=v.revenue_per_load_unit) for v in self.vehicles},
            Q={v.id: v.capacity for v in self.vehicles},
            P=[n.id for n in self.pickup_nodes],
            D=[n.id for n in self.dropoff_nodes],
            L={r.pickup_node.id: r.max_ride_time for r in self.requests},
            el={n.id: (n.tw.earliest, n.tw.latest) for n in self.nodes},
            d={n.id: n.service_duration for n in self.nodes},
            q={n.id: n.load for n in self.nodes},
            destination_depot=[n.id for n in self.destination_nodes],
            dist_matrix=self.dist_matrix_id,
            total_horizon=TOTAL_HORIZON,
        )

        return data

    def __del__(self):
        """Resets the id count for requests, vehicles, and nodes"""
        Request.cleanup()
        Vehicle.cleanup()
        Node.cleanup()

    def __str__(self) -> str:
        output = str(self.config)
        output += "\n\n### Nodes:\n" + "\n".join(map(str, self.nodes))
        output += "\n\n### Requests:\n"
        output += "\n".join(map(str, self.requests))
        output += "\n\n### Vehicles:\n"
        output += "\n".join(map(str, self.vehicles))
        return output
