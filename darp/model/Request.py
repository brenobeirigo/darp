from model.Node import PickupNode, DropoffNode
from model.TimeWindow import TimeWindow
import math

class Request:
    
    count = 0
    def __str__(self) -> str:
        return (
            f"{self.alias:>2}[{self.load}, ⧖{self.pickup_delay}/{self.dropoff_delay}]"
            f" {self.pickup_node}"
            f" → {self.dropoff_node}"
        )
    
    @staticmethod
    def cleanup():
        Request.count = 0
        
    def __repr__(self) -> str:
        return self.__str__()
    
    def __init__(
        self,
        pickup_id,
        dropoff_id,
        earliest_pickup_time,
        latest_pickup_time,
        earliest_dropoff_time, 
        latest_dropoff_time,
        load=1,
        pickup_point=None,
        dropoff_point=None,
        pickup_delay=0,
        dropoff_delay=0,
        max_ride_time=math.inf,
        alias=None):

        self.id = Request.count
        self.max_ride_time = max_ride_time        
        self.load = load

        self.alias = str(alias if alias else self.id)
        
        self.pickup_tw = TimeWindow(
            earliest_pickup_time,
            latest_pickup_time)

        self.dropoff_tw = TimeWindow(
            earliest_dropoff_time,
            latest_dropoff_time)

        self.pickup_node = PickupNode(pickup_id, self, point=pickup_point)
        self.dropoff_node = DropoffNode(dropoff_id, self, point=dropoff_point)
        
        self.pickup_delay = pickup_delay
        self.dropoff_delay = dropoff_delay
        
        Request.count += 1
        
