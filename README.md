# Benchmark

We download all benchmark instances listed in the paper:

- [Ho, Sin C., et al. “A Survey of Dial-a-Ride Problems: Literature Review and Recent Developments.” *Transportation Research Part B-Methodological*, vol. 111, 2018, pp. 395–421](https://doi.org/10.1016/j.trb.2018.02.001).

A summary of the algorithms' comparison and best known results can be found at <https://sites.google.com/site/darpsurvey/>.

## Downloading the instances

Execute the script at `data/download_benchmark.py` to download all instances.
The script will loop over the instance table `data/benchmark_instances.csv` and create a folder for each instance set inside the `data` folder.

## DARP Formulation

### Sets and Indices

- $i, j$: Indices for nodes, including pickup and drop-off points.
- $k$: Index for vehicles.
- $P$: Set of pickup nodes.
- $D$: Set of drop-off nodes.
- $V$: Set of vehicles.

### Parameters

- $n$: Number of users (requests).
- $Q$: Capacity of each vehicle.
- $T$: Maximum duration of any route.
- $L$: Maximum ride time for a user.
- $c_{ij}$: Travel cost from node $i$ to node $j$.
- $t_{ij}$: Travel time from node $i$ to node $j$.
- $e_i, l_i$: Earliest and latest service time for node $i$.
- $d_i$: Service duration at node $i$.
- $q_i$: Demand of user associated with node $i$ (often 1 for pickup and -1 for drop-off).

### Decision Variables

- $x_{ijk}$: Binary variable, 1 if vehicle $k$ travels from node $i$ to node $j$; 0 otherwise.
- $s_i$: Service start time at node $i$.
- $L_i$: Ride time for user $i$.
- $B_{ki}$: Arrival time of vehicle $k$ at node $i$.

### Objective Function

The objective is to minimize the total travel cost or time:
$$
\text{Minimize} \sum_{k \in V} \sum_{i \in P \cup D} \sum_{j \in P \cup D} c_{ij} x_{ijk}
$$

### Constraints

1. **Vehicle Route**:
   $$
   \sum_{j \in P \cup D} x_{ijk} = 1 \quad \forall i \in P, k \in V
   $$
   $$
   \sum_{i \in P \cup D} x_{ijk} = 1 \quad \forall j \in D, k \in V
   $$

2. **Vehicle Capacity**:
   $$
   \sum_{i \in P} q_i x_{ijk} \leq Q \quad \forall k \in V
   $$

3. **Time Windows**:
   $$
   e_i \leq s_i \leq l_i \quad \forall i \in P \cup D
   $$

4. **Service Time and Maximum Route Duration**:
   $$
   s_i + d_i + t_{ij} \leq s_j \quad \forall i, j \in P \cup D, i \neq j, k \in V
   B_{ki} + T \geq B_{kj} \quad \forall i, j \in P \cup D, k \in V
   $$

5. **Pickup and Delivery for the Same Vehicle**:
   $$
   \sum_{j \in P \cup D} x_{ijk} = \sum_{j \in P \cup D} x_{jik} \quad \forall i \in P, k \in V
   $$

6. **Arrival and Ride Time Constraints**:
   $$
   B_{ki} + t_{ij} \leq B_{kj} \quad \forall i, j \in P \cup D, k \in V
   $$
   $$
   L_i \leq L \quad \forall i \in P \cup D
   $$

## DARP Instances

### How Were Instances Generated?

In the scenario detailed by Cordeau et al. (2006), an instance involves $n$ users, with $n$ being an even number. Users are divided into two categories: the first half (users $1$ to $n/2$) are designated as outbound request users, and the second half (users $n/2+1$ to $n$) as inbound request users. The locations for pick-up and drop-off, defined as nodes, are selected randomly and independently within a square area, specifically within the coordinates $[-10,10] \times [−10,10]$. This selection follows a uniform distribution pattern. The depot, an essential point in the network, is centrally located in this square.

For every arc $(v_i,v_j)$ in the set $A$, there are two key parameters: the routing cost $c_{ij}$ and the travel time $t_{ij}$. Both of these are determined by the Euclidean distance between the nodes involved.

Each node is assigned a time window, denoted as $[e_i, l_i]$. For outbound users, the time window creation process begins with selecting a number $l_{n+i}$ from the interval $[0, T-60]$. Subsequently, the value $e_{n+i}$ is set to be $l_{n+i} − 15$. For inbound users, the process involves choosing $e_i$ from within the interval $[0, T − 60]$ and then setting $l_i$ to the value of $e_i + 15$. Here, $T$ represents the total length of the planning horizon.

#### Time Windows

For time windows associated with outbound and inbound user requests, specific methodologies are applied.

For an outbound user, the time window at their origin node is adjusted with the following formula: $e_i$ is set as $\max{\{0, e_{n+1} - L - d_i\}}$ and $l_{i}$ as $\min{\{l_{n+i} - t_{i, n+i} - d_i, T\}}$. For an inbound user, adjustments are made at the destination node by setting $e_{n+i}$ as $\max{\{0, e_{i} + d_i + t_{i, n+1}\}}$ and $l_{n+i}$ as $\min{\{l_{i} + d_i + L, T\}}$. Additionally, the time windows for nodes $0$ and $2n + 1$ are specifically tightened using the minimum and maximum calculations based on the parameters of $e_i$, $l_i$, $t_{0i}$, and $t_{i,2n+1}$.

In the first set of instances, each vehicle is designated with a capacity of $Q = 3$, and for every user, the demand is set at $q_i=1$ with a service duration of $d_i = 3$. The maximum allowed ride time is capped at 30 minutes. The second set differs in that the vehicle capacity is increased to $Q = 6$, and the $q_i$ values are randomly selected following a uniform distribution from the set $\{1, \dots, Q\}$. In this second scenario, the maximum ride time is extended to 45 minutes. Here, the service time, denoted as $d_i$, is considered to be proportional to the number of passengers. In both sets, the maximum route duration aligns with the planning horizon $T$. The first set of instances is representative of scenarios where cars are utilized for individual transportation, while the second set is indicative of scenarios involving mini-busses for transporting either individuals or groups.

### Instance

A typical DARP instance looks like this:

    2 16 480 3 30
    0   0.000   0.000   0   0    0  480
    1  -1.198  -5.164   3   1    0 1440
    2   5.573   7.114   3   1    0 1440
    3  -6.614   0.072   3   1    0 1440
    4  -7.374  -1.107   3   1    0 1440
    5  -9.251   8.321   3   1    0 1440
    6   6.498  -6.036   3   1    0 1440
    7   0.861   6.903   3   1    0 1440
    8   3.904  -5.261   3   1    0 1440
    9   7.976  -9.000   3   1  276  291
    10  -2.610   0.039   3   1   32   47
    11   4.487   7.142   3   1  115  130
    12   8.938  -4.388   3   1   14   29
    13  -4.172  -9.096   3   1  198  213
    14   7.835  -9.269   3   1  160  175
    15   2.792  -7.944   3   1  180  195
    16   5.212   9.271   3   1  366  381
    17   6.687   6.731   3  -1  402  417
    18  -2.192  -9.210   3  -1  322  337
    19  -1.061   8.752   3  -1  179  194
    20   6.883   0.882   3  -1  138  153
    21   5.586  -1.554   3  -1   82   97
    22  -9.865   1.398   3  -1   49   64
    23  -9.800   5.697   3  -1  400  415
    24   1.271   1.018   3  -1  298  313
    25   4.404  -1.952   3  -1    0 1440
    26   0.673   6.283   3  -1    0 1440
    27   7.032   2.808   3  -1    0 1440
    28  -0.694  -7.098   3  -1    0 1440
    29   3.763  -7.269   3  -1    0 1440
    30   6.634  -7.426   3  -1    0 1440
    31  -9.450   3.792   3  -1    0 1440
    32  -8.819  -4.749   3  -1    0 1440
    33   0.000   0.000   0   0    0  480

The first line `2 16 480 3 30` features, in turn:

- $|K|$: number of vehicles (`2`),
- $|P|$: number of customers (`16`),
- $T$: maximum route duration (`480`),
- $|Q|$: vehicle capacity (`3`),
- $L$: maximum ride time (`30`).

The subsequent lines comprise the columns:

- node id ($i \in N$),
- x coordinate,
- y coordinate,
- service duration at node ($d$)
- load (positive for $i \in P$, negative for $i \in D$, zero for $i=\text{depot}$),
- earliest arrival time ($e$),
- latest arrival time ($l$).

The second line `0   0.000   0.000   0   0    0  480` corresponds to the depot data.

The following $n=|P|=16$ lines feature the pickup node data:

    1  -1.198  -5.164   3   1    0 1440
    2   5.573   7.114   3   1    0 1440
    3  -6.614   0.072   3   1    0 1440
    4  -7.374  -1.107   3   1    0 1440
    5  -9.251   8.321   3   1    0 1440
    6   6.498  -6.036   3   1    0 1440
    7   0.861   6.903   3   1    0 1440
    8   3.904  -5.261   3   1    0 1440
    9   7.976  -9.000   3   1  276  291
    10  -2.610   0.039   3   1   32   47
    11   4.487   7.142   3   1  115  130
    12   8.938  -4.388   3   1   14   29
    13  -4.172  -9.096   3   1  198  213
    14   7.835  -9.269   3   1  160  175
    15   2.792  -7.944   3   1  180  195
    16   5.212   9.271   3   1  366  381

The following $n=|D|=16$ lines feature the destination node data:

    17   6.687   6.731   3  -1  402  417
    18  -2.192  -9.210   3  -1  322  337
    19  -1.061   8.752   3  -1  179  194
    20   6.883   0.882   3  -1  138  153
    21   5.586  -1.554   3  -1   82   97
    22  -9.865   1.398   3  -1   49   64
    23  -9.800   5.697   3  -1  400  415
    24   1.271   1.018   3  -1  298  313
    25   4.404  -1.952   3  -1    0 1440
    26   0.673   6.283   3  -1    0 1440
    27   7.032   2.808   3  -1    0 1440
    28  -0.694  -7.098   3  -1    0 1440
    29   3.763  -7.269   3  -1    0 1440
    30   6.634  -7.426   3  -1    0 1440
    31  -9.450   3.792   3  -1    0 1440
    32  -8.819  -4.749   3  -1    0 1440

The last line `33   0.000   0.000   0   0    0  480` corresponds to the destination depot data.
This node is replicated so all vehicles have to return to it.

Hence,  for example, passenger $i=1$:

- shall be picked up at node $1 \in P$ with load $q_1 = 1$,
- shall be picked up point $(x,y)=(-1.198, -5.164)$,
- shall be picked up within time window $(e_1, l_1) = (0, 1440)$,
- takes a pickup service duration $d_1=3$.

Later, passenger $1$:

- shall be delivered at node $n+i = 17$ with load $q_{17} = -1$ (i.e., $q_{17} = -q_1$),
- shall be delivered at point $(x,y)=(6.687, 6.731)$,
- shall be delivered within time window $(e_{17}, l_{17}) = (402, 417)$,
- takes a delivery service duration $d_{17}=3$.

## Best known results

Optimal results compiled by Ho, Sin C., et al. (2018) for the DARP instances by Cordeau (2006) and Ropke et al. (2007).
Researchers have used both artificially-generated and real-life data to evaluate the quality of the developed algorithms.
Set "a" considers small vehicle capacities and set "b" considers large vehicle capacities.

| Instance | Optimal | Instance | Optimal |
|:--------:|:-------:|:--------:|:-------:|
| a2-16    | 294.25  | b2-16    | 309.41  |
| a2-20    | 344.83  | b2-20    | 332.64  |
| a2-24    | 431.12  | b2-24    | 444.71  |
| a3-18    | 300.48  | b3-18    | 301.64  |
| a3-24    | 344.83  | b3-24    | 394.51  |
| a3-30    | 494.85  | b3-30    | 531.44  |
| a3-36    | 583.19  | b3-36    | 603.79  |
| a4-16    | 282.68  | b4-16    | 296.96  |
| a4-24    | 375.02  | b4-24    | 371.41  |
| a4-32    | 485.5   | b4-32    | 494.82  |
| a4-40    | 557.69  | b4-40    | 656.63  |
| a4-48    | 668.82  | b4-48    | 673.81  |
| a5-40    | 498.41  | b5-40    | 613.72  |
| a5-50    | 686.62  | b5-50    | 761.4   |
| a5-60    | 808.42  | b5-60    | 902.04  |
| a6-48    | 604.12  | b6-48    | 714.83  |
| a6-60    | 819.25  | b6-60    | 860.07  |
| a6-72    | 916.05  | b6-72    | 978.47  |
| a7-56    | 724.04  | b7-56    | 823.97  |
| a7-70    | 889.12  | b7-70    | 912.62  |
| a7-84    | 1033.37 | b7-84    | 1203.37 |
| a8-64    | 747.46  | b8-64    | 839.89  |
| a8-80    | 945.73  | b8-80    | 1036.34 |
| a8-96    | 1229.66 | b8-96    | 1185.55 |

## How to Read Vehicle Solutions

- Vehicle id
- `D`: Route total duration (e.g., for vehicle `4`: 466.708 - 98.245 = 368.463)
- `Q`: Max. occupancy
- `W`: Avg. total waiting at pickup and delivery nodes (vehicle arrived earlier than earliest time window)
- `T`: Avg. transit time (total ride time / number of requests)

- Node id
- `w`: Slack time (vehicle arrives at time `t`, waits `w` time units until `b`)
- `b`: Arrival time (t + w)
- `t`: Ride delay (only at dropoff nodes)
- `q`: Vehicle current capacity

Excerpt of instance `pr02` solution:

    0 D:	455.309 Q:	3 W:	7.1095 T:	53.0065	0 (w: 0; b: 85.2408; t: 0; q: 0) 44 (w: 0; b: 89; t: 0; q: 1)... 0 (w: 0; b: 540.55; t: 0; q: 0) 

    1 D:	373.95 Q:	3 W:	4.99494 T:	26.9578	0 (w: 0; b: 130.294; t: 0; q: 0) 32 (w: 0; b: 131.129; t: 0; q: 1) 3 ... (w: 0; b: 504.244; t: 0; q: 0) 

    2 D:	381.499 Q:	3 W:	24.5507 T:	37.0241	0 (w: 0; b: 54.962; t: 0; q: 0) 12 (w: 0; b: 60.639; t: 0; q: 1) ... 0 (w: 0; b: 436.461; t: 0; q: 0) 

    3 D:	408.096 Q:	4 W:	7.90828 T:	39.385	0 (w: 0; b: 79.9294; t: 0; q: 0) 42 (w: 0; b: 81.7573; t: 0; q: 1)... 0 (w: 0; b: 488.025; t: 0; q: 0) 

    4 D:	368.463 Q:	6 W:	2.75005 T:	44.4871	0 (w: 0; b: 98.2452; t: 0; q: 0) ... 89 (w: 0; b: 453.527; t: 1.94537; q: 0) 0 (w: 0; b: 466.708; t: 0; q: 0) 

    cost: 301.336
    total duration: 1987.32
    total waiting time: 725.982 average: 7.56231
    total transit time: 1965.46 average: 40.9472

Excerpt of instance solution `pr01.res`:

    190.02

    1 D:     84.33 Q:      2.00 W:       8.37 T:     30.22

    id   w     b      t     q             x        y     d  q   e   l     dist_previous
    0        188.54  0.00 0.00     0   -1.044    2.000  0  0    0 1440    0
    10  0.00  191.99  0.00 1.00    10    2.303    1.164 10  1    0 1440    3.4498268
    11  0.00  202.58  0.00 2.00    11    2.548    0.629 10  1    0 1440    0.58843019
    35  0.00  215.00  2.42 1.00    35    0.129    0.735 10 -1  178  215    2.42132144
    34 33.49  260.00 58.01 0.00    34    1.623    0.932 10 -1  260  276    1.50693234
    0  0.00  272.87  0.00 0.00

    215 (arrival at 35) + 10 (service duration) + 2.42 (travel time) + 227.42 + 33.49 = 260.

### Static vs. dynamic problems

A 'static' vehicle routing problem is the traditional academic problem where jobs and drivers are 100% known prior to the delivery period starting, and a single plan can therefore be generated at the start of the period.

## Solver SCIP

What is SCIP? <https://www.scipopt.org/>

A similar technique is used for solving both *Integer Programs* and *Constraint Programs*: the problem is successively divided into smaller subproblems (branching) that are solved recursively.

On the other hand, Integer Programming and Constraint Programming have different strengths:

- *Integer Programming* uses LP relaxations and cutting planes to provide strong dual bounds, while
- *Constraint Programming* can handle arbitrary (non-linear) constraints and uses propagation to tighten domains of variables.

SCIP is a framework for Constraint Integer Programming oriented towards the needs of mathematical programming experts who want to have total control of the solution process and access detailed information down to the guts of the solver. SCIP can also be used as a pure MIP and MINLP solver or as a framework for branch-cut-and-price.

SCIP is implemented as C callable library and provides C++ wrapper classes for user plugins. It can also be used as a standalone program to solve mixed integer linear and nonlinear programs given in various formats such as MPS, LP, flatzinc, CNF, OPB, WBO, PIP, etc. Furthermore, SCIP can directly read ZIMPL models.

## References

- [Jean-François Cordeau, (2006). A Branch-and-Cut Algorithm for the Dial-a-Ride Problem. Operations Research 54(3):573-586](http://dx.doi.org/10.1287/opre.1060.0283)
- [Jean-François Cordeau, Gilbert Laporte, (2003). A tabu search heuristic for the static multi-vehicle dial-a-ride problem.](https://www.sciencedirect.com/science/article/abs/pii/S0191261502000450?via%3Dihub)