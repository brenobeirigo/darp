
# The Problem Instances

```{python}
import os
import sys
from pathlib import Path
from pprint import pprint
import pandas as pd
import re
from src.data import parser as instance_parser

# Append the project's root directory to the system path
sys.path.append("../")

# Importing necessary modules from the project
from src.instance import Instance
```

```{python}
FOLDER_DATA_RAW_INSTANCES = "../data/raw/darp_instances"
FOLDER_RESULTS = "../data/results"
FILEPATH_DATA_SOURCE = "../src/data/benchmark_instances.csv"
```

## List of Instances

```{python}
instances = pd.read_csv(FILEPATH_DATA_SOURCE)
instances
```


## List of Downloaded Instances

```{python}
print(f"Instance folders at '{FOLDER_DATA_RAW_INSTANCES}':")
pprint(os.listdir(FOLDER_DATA_RAW_INSTANCES))
```
### Load and Process Instances

In this example, we focus on instances from "darp_cordeau_2006":
```{python}
instance_folder = Path(FOLDER_DATA_RAW_INSTANCES) / "darp_cordeau_2006"
instance_filenames = os.listdir(instance_folder)
print(instance_filenames)
```