from shapely.geometry import Point
from ..model.TimeWindow import TimeWindow

from enum import Enum
from dataclasses import dataclass
NodeType = Enum('NodeType', ['DEPOT_ORIGIN', 'PU', 'DO', "DEPOT_DESTINATION"])
@dataclass
class NodeInfo:
    id:int 
    x:float         
    y:float         
    service_duration:int         
    load:int         
    earliest:int         
    latest:int
    type:NodeType
    
    def __post_init__(self):
        self.point = Point(self.x, self.y)
        self.tw = TimeWindow(self.earliest, self.latest)

class Node:
    
    count = 0
    
    def __init__(self, pos, point=None, alias=None):
        self.id = Node.count
        self.alias = alias or self.id
        self.pos = pos
        self.point = (Point(point) if point is not None else None)
        self.arrival = None
        self.departure = None
        Node.count += 1

    @property
    def x(self):
        return self.point.x
    
    @property
    def xy_coord(self):
        return self.point.x, self.point.y
    
    @property
    def y(self):
        return self.point.y

    def __str__(self) -> str:
        return f"{self.alias:>3}"
    
    def __repr__(self) -> str: 
        return self.alias

    @staticmethod
    def cleanup():
        Node.count = 0
    
class PickupNode(Node):
    
    def __init__(self, pos, request, point=None):
        super().__init__(pos, point=point, alias=request.alias)
        self.request = request
    
    @property
    def tw(self):
        return self.request.pickup_tw
    
    @property
    def load(self):
        return self.request.load
    
    @property
    def el(self):
        return (
            self.tw.earliest,
            self.tw.latest)
    
    @property
    def service_delay(self):
        return self.request.pickup_delay
    
    def __str__(self) -> str:
        return super().__str__() + str(self.tw)
    
class DropoffNode(Node):
    
    def __init__(self, pos, request, point=None):
        super().__init__(pos, point=point, alias=f"{request.alias}'")
        self.request = request
    
    @property
    def tw(self):
        return self.request.dropoff_tw
    
    @property
    def load(self):
        return -self.request.load
    
    @property
    def el(self):
        return (
            self.tw.earliest,
            self.tw.latest)
    
    @property
    def service_delay(self):
        return self.request.dropoff_delay
    
    def __str__(self) -> str:
        return super().__str__() + str(self.tw)
        
class OriginNode(Node):
    
    def __init__(self, pos, vehicle, point=None):
        super().__init__(pos, point=point, alias=f"O({vehicle.alias})")
        self.vehicle = vehicle

    @property
    def tw(self):
        return self.vehicle.origin_tw
    
    @property
    def load(self):
        return 0
    
    @property
    def el(self):
        return (
            self.tw.earliest,
            self.tw.latest)
    

    @property
    def service_delay(self):
        return 0
    
    def __str__(self) -> str:
        return super().__str__() + str(self.tw)
    
class DestinationNode(Node):
    
    def __init__(self, pos, vehicle, point=None):
        super().__init__(pos, point=point, alias=f"D({vehicle.alias})")
        self.vehicle = vehicle

    @property
    def tw(self):
        return self.vehicle.destination_tw
    
    @property
    def load(self):
        return 0
    
    @property
    def el(self):
        return (
            self.tw.earliest,
            self.tw.latest)
    
    def __str__(self) -> str:
        return super().__str__() + str(self.tw)
    
    @property
    def service_delay(self):
        return 0