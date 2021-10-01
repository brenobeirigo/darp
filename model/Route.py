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
        departure_from_current_node = current_node.arrival + current_node.service_delay
        
        for next_node in self.nodes[1:]:
            dist = distance_matrix[current_node.pos][next_node.pos]
            arrival_at_next_node = departure_from_current_node + dist
            departure_from_next_node = arrival_at_next_node + next_node.service_delay
            
            print(
                current_node,
                current_node.arrival,
                departure_from_current_node,
                dist,
                next_node,
                arrival_at_next_node,
                departure_from_next_node)
            
            if arrival_at_next_node < next_node.tw.earliest:
                return False
            
            if arrival_at_next_node > next_node.tw.latest:
                return False
            
            current_node = next_node
            departure_from_current_node = departure_from_next_node
            
        return True