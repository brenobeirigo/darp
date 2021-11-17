from shapely.geometry import Point

class Node:
    
    count = 0
    
    def __init__(self, pos, point=None, alias=None):
        Node.count += 1 
        self.id = Node.count
        self.alias = (alias if alias else self.id)
        self.pos = pos
        self.point = (Point(point) if point is not None else None)
        self.arrival = None
        self.departure = None

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
    def service_delay(self):
        return self.request.pickup_delay
    
    def __str__(self) -> str:
        return super().__str__() + str(str(self.tw))
    
class DropoffNode(Node):
    
    def __init__(self, pos, request, point=None):
        super().__init__(pos, point=point, alias=request.alias + "'")
        self.request = request
    
    @property
    def tw(self):
        return self.request.dropoff_tw
    
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
    
    def __str__(self) -> str:
        return super().__str__() + str(self.tw)
    
    @property
    def service_delay(self):
        return 0