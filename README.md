# Benchmark

We download all benchmark instances listed in the paper:

- [Ho, Sin C., et al. “A Survey of Dial-a-Ride Problems: Literature Review and Recent Developments.” *Transportation Research Part B-Methodological*, vol. 111, 2018, pp. 395–421](https://doi.org/10.1016/j.trb.2018.02.001).

A summary of the algorithms' comparison and best known results can be found at https://sites.google.com/site/darpsurvey/.


## Downloading the instances

Execute the script at `data/download_benchmark.py` to download all instances.
The script will loop over the instance table `data/benchmark_instances.csv` and create a folder for each instance set inside the `data` folder.

