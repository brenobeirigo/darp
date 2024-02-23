from ..model.node import PickupNode, DropoffNode
from ..model.TimeWindow import TimeWindow
import math
from ..model.node import NodeInfo


class Request:
    count = 1

    def __str__(self) -> str:
        return (
            f"{self.alias:>2}[{self.load}, ⧖{self.pu.service_duration}/{self.do.service_duration}]"
            f" {self.pickup_node}"
            f" → {self.dropoff_node}"
        )

    @staticmethod
    def cleanup():
        Request.count = 1

    def __repr__(self) -> str:
        return self.__str__()

    def __init__(
        self,
        pu_info: NodeInfo,
        do_info: NodeInfo,
        max_ride_time=math.inf,
        alias=None,
    ):
        self.id = Request.count
        self.alias = str(alias if alias else self.id)

        self.pu = pu_info
        self.do = do_info

        # Defined system wide for all requests
        self.max_ride_time = max_ride_time

        self.load = pu_info.load
        self.pickup_node = PickupNode(pu_info, self)
        self.dropoff_node = DropoffNode(do_info, self)

        Request.count += 1

    @property
    def pickup_tw(self):
        return self.pu.tw

    @property
    def dropoff_tw(self):
        return self.do.tw
