from ..model.node import PickupNode, DropoffNode
from ..model.TimeWindow import TimeWindow
import math
from ..model.node import NodeInfo

class Request:
    
    count = 1
    def __str__(self) -> str:
        return (
            f"{self.alias:>2}[{self.load}, ⧖{self.pickup_delay}/{self.dropoff_delay}]"
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
            alias=None):

            self.id = Request.count
            self.alias = str(alias if alias else self.id)
            
            self.pu_info = pu_info
            self.do_info = do_info
            
            # Defined system wide for all requests
            self.max_ride_time = max_ride_time        


            self.load = pu_info.load
            self.pickup_node = PickupNode(pu_info.id, self, point=pu_info.point)
            self.dropoff_node = DropoffNode(do_info.id, self, point=do_info.point)
            
            Request.count += 1
       
    @property     
    def pickup_tw(self):
        return self.pu_info.tw
    
    @property
    def dropoff_tw(self):
        return self.do_info.tw
    
    @property
    def pickup_delay(self):
        return self.pu_info.service_duration
    
    @property
    def dropoff_delay(self):
        return self.do_info.service_duration