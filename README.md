# Benchmark

We download all benchmark instances listed in the paper:

- [Ho, Sin C., et al. “A Survey of Dial-a-Ride Problems: Literature Review and Recent Developments.” *Transportation Research Part B-Methodological*, vol. 111, 2018, pp. 395–421](https://doi.org/10.1016/j.trb.2018.02.001).

A summary of the algorithms' comparison and best known results can be found at <https://sites.google.com/site/darpsurvey/>.

## Downloading the instances

Execute the script at `data/download_benchmark.py` to download all instances.
The script will loop over the instance table `data/benchmark_instances.csv` and create a folder for each instance set inside the `data` folder.

## Solver SCIP

What is SCIP? <https://www.scipopt.org/>

A similar technique is used for solving both *Integer Programs* and *Constraint Programs*: the problem is successively divided into smaller subproblems (branching) that are solved recursively.

On the other hand, Integer Programming and Constraint Programming have different strengths:

- *Integer Programming* uses LP relaxations and cutting planes to provide strong dual bounds, while
- *Constraint Programming* can handle arbitrary (non-linear) constraints and uses propagation to tighten domains of variables.

SCIP is a framework for Constraint Integer Programming oriented towards the needs of mathematical programming experts who want to have total control of the solution process and access detailed information down to the guts of the solver. SCIP can also be used as a pure MIP and MINLP solver or as a framework for branch-cut-and-price.

SCIP is implemented as C callable library and provides C++ wrapper classes for user plugins. It can also be used as a standalone program to solve mixed integer linear and nonlinear programs given in various formats such as MPS, LP, flatzinc, CNF, OPB, WBO, PIP, etc. Furthermore, SCIP can directly read ZIMPL models.
