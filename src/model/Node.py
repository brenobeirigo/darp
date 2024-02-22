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
    alias:str = None
    
    def __post_init__(self):
        self.point = Point(self.x, self.y)
        self.tw = TimeWindow(self.earliest, self.latest)
        self.alias = self.alias or self.id

class Node:
    
    count = 0
    
    def __init__(self, info: NodeInfo, alias=None):
        self.id = info.id
        self.alias = alias or self.id
        self.pos = info.id
        self.point = info.point
        self.arrival = None
        self.departure = None
        self.service_duration = info.service_duration
        self.tw = info.tw
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
    
    def __init__(self, info, request):
        super().__init__(info, alias=request.alias)
        self.request = request
    
    @property
    def load(self):
        return self.request.load
    
    @property
    def el(self):
        return (self.tw.earliest, self.tw.latest)

    def __str__(self) -> str:
        return super().__str__() + str(self.tw)
    
class DropoffNode(Node):
    
    def __init__(self, info, request):
        super().__init__(info, alias=f"{request.alias}'")
        self.request = request
    
    @property
    def load(self):
        return -self.request.load
    
    @property
    def el(self):
        return (
            self.tw.earliest,
            self.tw.latest)
    
    def __str__(self) -> str:
        return super().__str__() + str(self.tw)
        
class OriginNode(Node):
    
    def __init__(self, info, vehicle):
        super().__init__(info, alias=f"O")
        self.vehicle = vehicle
    
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
    
class DestinationNode(Node):
    
    def __init__(self, info, vehicle):
        super().__init__(info, alias=f"D({vehicle.alias})")
        self.vehicle = vehicle
    
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
