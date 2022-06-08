Copyright 2022 Brandon C. Alston, Hamidreza Validi

# Optimal Binary Classification Trees

Code for the paper ["Mixed Integer Linear Optimization Formulations for Learning Optimal Binary Classification Trees"](hyperlink) by Brandon Alston, Hamidreza Validi, and Illya V. Hicks.

This code uses [python3.x](https://www.python.org/downloads/) (version 3.6 and higher) and requires the [Gurobi9.x](https://www.gurobi.com/) solver. Required python packages are outlined in `requirements.txt`.

*** 
***

## Summary of Repository
- `Code/` contains the code of the formulations **MCF1**, **MCF2**, **CUT1**, and **CUT2**
  - `OBCT.py` contains the formulations of each model for solving using Gurobi9.x
  - `TREE.py` creates the necessary tree information including path, child, and other information
  - `SPEED_UP.py` contains the code for the callbacks used in user fractional separation procedures
  - `UTILS.py` contains the code for viewing model decision variable results and generating the .csv results files among other utility functions
  - `model_runs.py` contains the code necessary to create, solve, and report results in a `.csv` file of each instance called by the user
  - `pareto_runs.py` contains the code necessary to create a pareto frontier `.png` file of instances called by the user

- `results_files/` folder stores the generated `.csv` files with model metrics
- `pareto_figures/` folder stores the generated `.png` files for experimental results
- `log_files/` folder stores model `.lp` files and Gurobi `.txt` log files
- `Datasets/` folder contains the datasets used for generating experimental results
  - Note: `Datasets/` should also be used as the folder where user dataset `.csv` files are stored

***
***

## Running Code

- Ensure the latest versions of the packages in `requirements.txt` are installed
- For an instance of `OBCT` run the `main` function in `model_runs.py` with the following arguments
    - d : `str list`, name(s) of dataset file(s)
    - h : `int list`, maximum depth(s) of trained tree(s)
    - t : `float`, gurobi model time limit in s
    - m : `str list`, list of model(s) to use
    - r : `int list`, list of random seed(s) to use
      - `rand_seed = [k,...,k]`  for repeat use of randome state `k`
    - e : `str list`, model extra(s), if applicable
    - c : `str`, tuning parameter
    - f : `str`, results output file `.csv`
    - l : `boolean`, log console to `.txt` file and write model to `.lp` file, both saved to the `\log_files` folder for each model called by user

You can call the `model_runs.py` main function within a python file as follows,

```python
import model_runs
data_names = ['soybean-small_enc','monk3_enc','balance-scale_enc','car_evaluation_enc']
heights = [3,4,5]
models = ['MCF1', 'MCF2', 'CUT1-ALL', 'CUT2-FRAC-3']
time_limit = 3600
extras = ['max_features-25']
rand_seed = [13, 58, 94, None]
tuning = None
file = 'test_results.csv'
consol_log = False
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit,
    "-e", extras, "-r", rand_seed, "-c", tuning, "-f", file, "-l", consol_log])
```

To run from terminal do the following,
```bash
python3 model_runs.py -d ['soybean-small_enc','monk3_enc','balance-scale_enc','car_evaluation_enc'] -h [3,4,5] -m ['MCF1','MCF2','CUT1-ALL','CUT2-FRAC-3'] -t 3600 -e ['max_features-25'] -r [13, 58, 94, None] -c None -f 'test_results.csv' -l False
```
Note:
- We assume the target column is labeled `'target'`. Change the hard code in `model_runs.py` to change the according target column (line 84)
- If results output file `-f file` is `None` the `models_run.py` automatically generates a `.csv` results file with the parameters of the function call as the file name saved to the `\results_files` folder
- `-e model_extras`, `-c tuning`, and `-f file` may be `None` input arguments, all others must hold a valid value

***
## Pareto Frontier
To generate the Pareto frontier call the `main` function in `pareto_runs.py` with the below parameters:
  - d : str list, names of dataset files
  - h : int, maximum depth of trained trees
  - t : float, gurobi model time limit in s
  - m : str list, models to use
  - f : str, results output file .csv

A `.png` file for each dataset called by the user is generated and stored in `\results_figures\` folder

We assume `-f file` is located in the `\results_files` folder
- If results output file `-f file` is `None` the `pareto_runs.py` automatically generates a `.csv` results file with the parameters of the function call as the file name saved to the `\results_files` folder

You can generate pareto frontiers from within a python file as follows,
```python
import pareto_runs
height = 4
models = ['FOCT', 'MCF1', 'MCF2', 'CUT1', 'CUT2']
rand_states = [15, 78, 0]
data_names = ['hayes-roth_enc', 'house-votes-84_enc']
file = 'pareto_test.csv'
pareto_runs.main(["-d", data_names, "-h", height, "-m", models, "-t", 3600, "-r", rand_states, "-f", file])
```

To run from terminal do the following 
```bash
python3 pareto_runs.py -d ['hayes-roth_enc', 'house-votes-84_enc'] -h 4 -m ['FOCT', 'MCF1', 'MCF2', 'CUT1', 'CUT2'] -t 3600 -r [15, 78, 0] -f 'pareto_test.csv'
```
- Note: `FOCT` must be the model name to generate the pareto frontier of FlowOCT
***

## Models Functionality
For understanding model functionality associated with integer and fractional separation procedures in **CUT1** and **CUT2** models, `-e model_extras` and `-c tuning` functionality please refer to the `USAGE.md` file. 

***
## Paper Results
To recreate the results outlined in the [paper](hyperlink) run `paper_results.py`

***

## Acknowledgments
The code found in `BendersOCT.py`, `FlowOCT.py`, `FlowOCTTree.py,` and `FlowOCTutils.py` are taken directly from the [Strong Tree](https://github.com/pashew94/StrongTree/) GitHub public repository.
All rights and ownership are to the original owners. 

***
***

![Screenshot](CAAM_logo.png)
