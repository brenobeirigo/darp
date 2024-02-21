# https://cdv.dei.uc.pt/wp-content/uploads/2014/03/ptmc02a.pdf
import os
import sys
sys.path.append(os.getcwd())
from config import FOLDER_DATA_RAW_INSTANCES
from src.instance import parser as instance_parser
from src.solution import parser as solution_parser
from pprint import pprint
import os
from src.model.node import DropoffNode
'''
1 D:     84.33 Q:      2.00 W:       8.37 T:     30.22  0 (b:188.54; t:0.00; q:0.00) 10 (w:0.00 a:191.99; t:0.00; q:1.00) 11 (w:0.00 a:202.58; t:0.00; q:2.00) 35 (w:0.00 a:215.00; t:2.42; q:1.00) 34 (w:33.49 a:260.00; t:58.01; q:0.00) 0 (w:0.00 a:272.87; t:0.00; q:0.00)
2 D:    351.65 Q:      3.00 W:       5.21 T:     40.15  0 (b:148.08; t:0.00; q:0.00) 14 (w:0.00 a:152.00; t:0.00; q:1.00) 22 (w:9.70 a:177.00; t:0.00; q:2.00) 3 (w:13.18 a:203.83; t:0.00; q:3.00) 27 (w:0.00 a:215.76; t:1.92; q:2.00) 46 (w:0.00 a:234.91; t:47.91; q:1.00) 38 (w:0.00 a:252.00; t:90.00; q:0.00) 12 (w:70.83 a:333.70; t:0.00; q:1.00) 24 (w:0.00 a:344.44; t:0.00; q:2.00) 48 (w:0.00 a:362.56; t:8.12; q:1.00) 6 (w:0.00 a:373.67; t:0.00; q:2.00) 36 (w:0.00 a:391.55; t:47.85; q:1.00) 15 (w:0.00 a:404.99; t:0.00; q:2.00) 18 (w:0.00 a:417.69; t:0.00; q:3.00) 30 (w:0.00 a:432.00; t:48.33; q:2.00) 21 (w:0.00 a:444.96; t:0.00; q:3.00) 39 (w:0.00 a:457.75; t:42.76; q:2.00) 42 (w:0.00 a:471.54; t:43.84; q:1.00) 45 (w:0.00 a:485.61; t:30.65; q:0.00) 0 (w:0.00 a:499.72; t:0.00; q:0.00)
3 D:    445.18 Q:      4.00 W:       3.23 T:     51.78  0 (b:83.13; t:0.00; q:0.00) 9 (w:0.00 a:85.38; t:0.00; q:1.00) 17 (w:0.00 a:104.63; t:0.00; q:2.00) 33 (w:0.00 a:123.00; t:27.62; q:1.00) 8 (w:30.28 a:166.05; t:0.00; q:2.00) 20 (w:0.00 a:178.53; t:0.00; q:3.00) 1 (w:0.00 a:190.74; t:0.00; q:4.00) 41 (w:0.00 a:204.63; t:90.00; q:3.00) 7 (w:2.56 a:218.82; t:0.00; q:4.00) 31 (w:0.00 a:231.04; t:2.22; q:3.00) 44 (w:0.00 a:241.96; t:53.43; q:2.00) 32 (w:0.00 a:252.00; t:75.95; q:1.00) 2 (w:10.42 a:274.43; t:0.00; q:2.00) 25 (w:0.00 a:287.00; t:86.26; q:1.00) 5 (w:17.88 a:320.42; t:0.00; q:2.00) 13 (w:0.00 a:333.65; t:0.00; q:3.00) 29 (w:0.00 a:346.94; t:16.52; q:2.00) 26 (w:0.00 a:361.00; t:76.57; q:1.00) 16 (w:20.78 a:393.70; t:0.00; q:2.00) 4 (w:0.00 a:407.07; t:0.00; q:3.00) 28 (w:0.00 a:418.76; t:1.69; q:2.00) 37 (w:0.00 a:433.65; t:90.00; q:1.00) 19 (w:2.02 a:454.00; t:0.00; q:2.00) 23 (w:0.00 a:473.81; t:0.00; q:3.00) 40 (w:0.00 a:488.62; t:84.92; q:2.00) 47 (w:0.00 a:501.95; t:18.14; q:1.00) 43 (w:0.00 a:513.86; t:49.86; q:0.00) 0 (w:0.00 a:528.31; t:0.00; q:0.00)

'''

## Loading the whole dataset


'''

files = os.listdir(instance_folder)

print(files)

for filename in files:
    print("#############", filename)
    instance_filepath = os.path.join(instance_folder, filename)

    instance = pi.parse_instance_from_filepath(
        instance_filepath, instance_parser=pi.PARSER_TYPE_CORDEAU)

    print(instance)
'''

root = FOLDER_DATA_RAW_INSTANCES
input_filepath = os.path.join(*"darp_ropke_2007/tabu/pr02".split("/"))
output_filepath = os.path.join(*"darpsrp_parragh_2015/DARP/pr02_result.txt".split("/"))

instance = instance_parser.parse_instance_from_filepath(
        os.path.join(root, input_filepath),
        instance_parser=instance_parser.PARSER_TYPE_ROPKE)

print(instance.config)

print(instance)

solution = solution_parser.parse_solution_from_filepath(os.path.join(root, output_filepath))

print("### ROUTES:")
total_cost = 0
total_duration = 0
total_transit = 0
for r in solution.vehicle_routes:
    print(r)
    total_transit += r.get_total_transit(instance)
    total_duration += r.get_total_duration()
    total_cost += r.get_total_cost(instance.dist_matrix)

assert round(total_cost,2) == round(solution.cost,2)
assert round(total_duration, 2) ==round(solution.total_duration,2), f'{total_duration} != {solution.total_duration}' 
assert round(total_transit, 2) == round(solution.total_transit,2), f'{total_transit} != {solution.total_transit}' 

print(r.is_route_feasible(instance))
print("TOTAL COST:", total_cost)
print("TOTAL TRANSIT:", total_transit)
print("TOTAL DURATION:", total_duration)
# TODO total waiting
# TODO minimal representation [[node_id, arrival]]
# 0 D:	455.309 Q:	3 W:	7.1095 T:	53.0065	0 
# 0 (w: 0; b: 85.2408; t: 0; q: 0)
# 44 (w: 0; b: 89; t: 0; q: 1)
# 20 (w: 29.7905; b: 131.026; t: 0; q: 2)
# 92 (w: 0; b: 142.205; t: 43.2055; q: 1)
# 38 (w: 0; b: 153.316; t: 0; q: 2)
# 27 (w: 0; b: 167; t: 0; q: 3)
# 68 (w: 15.4097; b: 195; t: 53.9744; q: 2)
# 86 (w: 0; b: 209.645; t: 46.3291; q: 1)
# 30 (w: 21.2879; b: 248; t: 0; q: 2)
# 75 (w: 0; b: 261.402; t: 84.4021; q: 1)
# 15 (w: 58.657; b: 334.401; t: 0; q: 2)
# 78 (w: 0; b: 348; t: 90; q: 1)
# 1 (w: 0; b: 359.092; t: 0; q: 2)
# 25 (w: 18.666; b: 390; t: 0; q: 3)
# 63 (w: 0; b: 405.759; t: 61.3586; q: 2)
# 4 (w: 0; b: 418.866; t: 0; q: 3)
# 52 (w: 0; b: 431.388; t: 2.52195; q: 2)
# 49 (w: 12.5978; b: 459; t: 89.9084; q: 1)
# 26 (w: 0; b: 473.387; t: 0; q: 2)
# 73 (w: 0; b: 488.611; t: 88.6115; q: 1)
# 74 (w: 0; b: 500.733; t: 17.3457; q: 0)
# 37 (w: 0; b: 512.889; t: 0; q: 1)
# 85 (w: 0; b: 528.304; t: 5.41463; q: 0)
# 0 (w: 0; b: 540.55; t: 0; q: 0)

'''

route = [44,20,92,38,27,68,86,30,75,15,78,1,25,63,4,52,49,26,73,74,37,85]

print(instance)
v1_node_ids = [44,20,92,38,27,68,86,30,75,15,78,1,25,63,4,52,49,26,73,74,37,85]
print(instance.vehicles)
print(instance.node_id_dict)

v0 = instance.vehicles[0]
for n in v1_node_ids:
    v0.route.append_node(instance.node_id_dict[n])

v0.route.append_node(v0.destination_node)

print("FEASIBLE:", v0.route.is_feasible(instance.dist_matrix))
print(v0.route)


'''

total_v_transit = 0        
for v_sol in solution.vehicle_routes:
    
    print("\n" + str(v_sol))
    
    
    print("All nodes:")
    b = 0
    vehicle_waiting = 0
    for n in v_sol.visits:
        
        node = instance.node_id_dict[n.id]
        node.arrival = n.b
        
        b += n.b-node.tw.earliest
        
        departure = node.arrival + node.service_delay
        B_i = max(node.arrival, node.tw.earliest)
        v_W = B_i - node.arrival
        vehicle_waiting += v_W
        print(n, node, f"{B_i:6.2f}", f"{v_W:6.2f}", f"{node.arrival:6.2f}", f"{node.tw.earliest:6.2f}")
    
    print("Dropoffs:")
    transit = 0
    total_ride = 0
    max_q = [n.q for n in v_sol.visits] 
    for n in v_sol.visits:    
        node = instance.node_id_dict[n.id]
        print(n.q)
        if type(node) == DropoffNode:
            d = n
            o = node.request.pickup_node
            departure_o = o.arrival + o.service_delay
            ride_delay = d.b - departure_o
            total_ride += ride_delay
            shortest_dist = instance.dist_matrix[o.pos][node.pos]
            
            t = ride_delay - shortest_dist
            transit += ride_delay
            print(o, node, o.pos, node.pos, f"ride: {round(ride_delay,2)}", round(shortest_dist,2),  f"total transit: {round(transit,2)}",f"      total ride: {round(total_ride,2)}", f"t: {round(t,2)}")
    
    total_v_transit+=transit
    print("TRANSIT:", total_v_transit)
    print(max(max_q))
    print("from to")
    cost = 0
    earliest_vehicle = 0
    arrival = 0
    total_distance = 0
    vehicle_waiting_node = 0
    vehicle_waiting = 0
    for o, d in zip(v_sol.visits[:-1], v_sol.visits[1:]):
        n_o = instance.node_id_dict[o.id]
        n_d = instance.node_id_dict[d.id]
        departure = o.b + n_o.service_delay
        node_waiting = o.b - n_o.tw.earliest
        dist = instance.dist_matrix[n_o.pos][n_d.pos]
        cost+=dist
        print(f"arrival({o.id:>6}) = {arrival:8.3f} / {o.b:7.3f} (w:{vehicle_waiting_node:>8.3f}) \t\t departure({o.id:>6}) = {departure:8.3f} \t\t dist({o.id:>6},{d.id:>6}) = {dist:7.3f} / ({total_distance:7.3f}) + d[{o.id:>6}] = {n_o.service_delay:7.3f} \t\t ({n_o.tw.earliest:7},{n_d.tw.latest:7}) \t\t node w: {node_waiting:7.3f} \t\t cost: {cost:7.3f}")
        arrival = departure + dist
        vehicle_waiting_node = d.b - arrival
        vehicle_waiting += vehicle_waiting_node
        total_distance += dist


    n_d = instance.node_id_dict[d.id]
    departure = d.b + n_d.service_delay
    node_waiting = d.b - n_d.tw.earliest
    print(f"arrival({d.id:>6}) = {arrival:8.3f} / {d.b:7.3f} (w:{vehicle_waiting_node:>8.3f}) \t\t departure({d.id:>6}) = {departure:8.3f} \t\t dist({d.id:>6},{0:>6}) = {0:7.3f} / ({total_distance:7.3f}) + d[{d.id:>6}] = {n_d.service_delay:7.3f} \t\t ({0:7},{n_d.tw.latest:7}) \t\t node w: {node_waiting:7.3f} \t\t cost: {cost:7.3f}")

    
            #print(n_o, n_o.service_delay, o, " - delay:", o.b-n_o.tw.earliest, "   →   ", n_d, n_d.service_delay, d, " - delay:", d.b-n_d.tw.earliest, " - DIST: ", dist, delay_node, earliest_vehicle)
print("cost:", cost)
print("b:", b)
print("total waiting time:", vehicle_waiting, vehicle_waiting/(len(v_sol.visits)-2))
print(solution)
print("\n".join(map(str, solution.vehicle_routes)))
