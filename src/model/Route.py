from ..solution.Solution import Solution

class Route:
    
    def __init__(self, origin_node):
        self.nodes = [origin_node]
    
    def __str__(self) -> str:
        return "[" + ", ".join(str(n) for n in self.nodes) + "]"
    
    def __repr__(self) -> str: 
        return "[" + ", ".join(str(n) for n in self.nodes) + "]"
    
    def append_node(self, node):
        self.nodes.append(node)
        
    def is_feasible(self, distance_matrix):
        current_node = self.nodes[0]
        earliest_arrival_current_node = 0
        earliest_departure_from_current_node = current_node.arrival + current_node.service_delay
        
        total_duration = 0
        total_waiting = 0
        total_transit = 0
        
        for next_node in self.nodes[1:]:
            arrival_current_node = earliest_arrival_current_node
            
            dist = distance_matrix[current_node.pos][next_node.pos]
            total_transit += dist
            
            latest_departure_from_current_node = next_node.tw.earliest - dist
            
            earliest_arrival_at_next_node = earliest_departure_from_current_node + dist
            arrival_next_node = max(earliest_arrival_at_next_node, next_node.tw.earliest)
            departure_next_node = arrival_next_node + next_node.service_delay
            
            departure_current_node = max(latest_departure_from_current_node, earliest_departure_from_current_node)
            
            total_waiting += arrival_next_node - next_node.tw.earliest
            
            print(
                f"{arrival_current_node:6.2f}",
                current_node,
                f"{departure_current_node:6.2f}",
                "→",
                f"{dist:6.2f}",
                "→",
                f"{arrival_next_node:6.2f}",
                next_node,
                f"{departure_next_node:6.2f}",
                f"W: {total_waiting:6.2f}")
            
            if earliest_arrival_at_next_node > next_node.tw.latest:
                return False
            
            
            
            current_node = next_node
            earliest_arrival_current_node = arrival_next_node
            earliest_departure_from_current_node = departure_next_node
        
        total_duration = earliest_departure_from_current_node
        
        s = Solution(0, total_duration, total_waiting, total_transit)
        print(s)
        
        
        return True