class Solution:
    def __init__(
        self,
        cost,
        total_duration,
        total_waiting,
        total_transit,
        avg_waiting=None,
        avg_transit=None,
        vehicle_routes=None,
    ):
        self.cost = cost
        self.total_duration = total_duration
        self.total_waiting = total_waiting
        self.total_transit = total_transit
        self.vehicle_routes = vehicle_routes
        self.avg_transit = avg_transit
        self.avg_waiting = avg_waiting

    def __repr__(self):
        avg_w = (
            f"avg_waiting={self.avg_waiting:10.4f}, "
            if self.avg_waiting
            else ""
        )
        avg_t = (
            f", avg_transit={self.avg_transit:10.4f}"
            if self.avg_transit
            else ""
        )

        return (
            "Solution("
            f"total_cost={self.cost:10.4f}, "
            f"total_duration={self.total_duration:10.4f}, "
            f"total_waiting={self.total_waiting:10.4f}, {avg_w}"
            f"total_transit={self.total_transit:10.4f} {avg_t}"
            ")"
        )


class VehicleSolution:
    def __init__(self, id_vehicle, D, Q, W, T, visits):
        self.id = id_vehicle
        self.D = D
        self.Q = Q
        self.W = W
        self.T = T
        self.visits = visits

    def __repr__(self):
        visits = " ".join(map(str, self.visits))
        return (
            f"{self.id} "
            f"D:   {self.D:10.4f} "
            f"Q: {self.Q:>2} "
            f"W:    {self.W:10.4f} "
            f"T:    {self.T:10.4f} "
            f"{visits}"
        )


class NodeSolution:
    def __init__(self, id_node, w, b, t, q):
        self.id = id_node
        self.w = w
        self.b = b
        self.t = t
        self.q = q

    def __repr__(self):
        return (
            f"{self.id} ("
            "w: {self.w:6.2f}; "
            "b: {self.b:6.2f}; "
            "t: {self.t:6.2f}; "
            "q: {self.q:6.2f})"
        )
