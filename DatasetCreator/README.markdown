# Preparing Datasets

Get an academic gurobi licence at https://www.gurobi.com/.
This is a neccesary step to run prepare_datasets.py, which uses gurobi to obtain optimal results for CO problem instances.

## Create Datasets

run prepare_datasets.py

e.g.
```setup
python prepare_datasets.py --dataset RB_iid_100 --problem MVC
```

All possible datasets and CO problems are listed within prepare_datasets.py.
