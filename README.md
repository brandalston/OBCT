Copyright 2023 Brandon C. Alston, Hamidreza Validi, Illya V. Hicks

# Optimal Binary Classification Trees

Code for the paper ["Mixed Integer Linear Optimization Formulations for Learning Optimal Binary Classification Trees"](http://arxiv.org/abs/2206.04857) by Brandon Alston, Hamidreza Validi, and Illya V. Hicks.

This code uses [python3.x](https://www.python.org/downloads/) (version 3.6 and higher) and requires the [Gurobi](https://www.gurobi.com/) solver. Required python3.x packages are outlined in `requirements.txt`.

*** 
***

## Summary of Repository
- `OBCT.py` contains the formulations of each model for solving using Gurobi9.x
- `TREE.py` creates the necessary tree information including path, child, and other information
- `SPEED_UP.py` contains the code for the callbacks used in user fractional separation procedures
- `UTILS.py` contains the code for viewing model decision variable results and generating the .csv results files among other utility functions
- `model_runs.py` contains the code necessary to create, solve, and report results in a `.csv` file of each instance called by the user
- `results_files/` folder stores the generated `.csv` files with model metrics
- `figures/` folder stores the generated `.png` files for experimental results
- `log_files/` folder stores model `.lp` files and Gurobi `.txt` log files
- `Datasets/` folder contains the datasets used for generating experimental results
  - Note: `Datasets/` should also be used as the folder where user dataset files are stored
  - Note that ``POKE`` is equivalent to ``CUT_W`` in the paper. We invoke a different name in implementation due to the similarity in parsing the strings ``CUT, CUT_W``.
***
***

## Running Code

- Ensure the latest versions of the packages in `requirements.txt` are installed
- For an instance of `OBCT` run with one of the functions in `model_runs.py` we use following arguments (not all functional call parameters apply to all functions)
    - d : `str list`, name(s) of dataset file(s)
    - h : `int list`, maximum depth(s) of trained tree(s)
    - t : `float`, gurobi model time limit in s
    - m : `str list`, list of model(s) to use
    - r : `int list`, list of random seed(s) to use
      - `rand_seed = [k,...,k]`  for repeat use of randome state `k`
    - f : `str`, results output file `.csv`
    - c : `boolean`, calibration of a weighted objective function where we calibrate the hyperparameter using a 15% validation set and 5-fold cross validation
    - p : `str`, objective priority parameter used in bi-objective modeling
    - e : `str list`, model extra(s), if applicable
    - l : `boolean`, log console to `.txt` file and write model to `.lp` file, both saved to the `\log_files` folder for each model called by user
Note:
- We assume the target column is labeled `'target'`. Change the hard code in `model_runs.py` to change the according target column
- `-e model_extras`, `-p priorities`, and `-f file` may be `None` input arguments, all others must hold a valid value
- If results output file `-f file` is `None` the `models_run.py` calls automatically generates a `.csv` results file with the parameters of the function call as the file name saved to the `\results_files` folder

***
Call the `model_runs.py` `main` function within a python file as follows to generate a model ignorning our second objective,

```python
import model_runs

data_names = ['soybean_small', 'monk3', 'car', 'iris', 'climate']
heights = [3, 4, 5]
models = ['SCF', 'MCF', 'POKE-ALL', 'CUT-FRAC-3']
time_limit = 3600
extras = ['max_features-25']
rand_seed = [13, 58, 94, None]
tuning = False
file = 'example_code_output.csv'
consol_log = False
model_runs.main(
  ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-c", tuning,
   "-e", extras, "-r", rand_seed, "-f", file, "-l", consol_log])
```
To run from terminal do the following,
```bash
python3 import model_runs; model_runs.main -d ['soybean-small','monk3','car','iris','climate'] -h [3,4,5] -m ['SCF','MCF','POKE-ALL','CUT-FRAC-3'] -t 3600 -e ['max_features-25'] -r [13, 58, 94, None] -c False -f 'test_results.csv' -l False
```

***
## Bi-objective Modeling
Call the `multiobj` function within a python file as follows to generate a model using the heirarchical modeling capabilities of Gurobi

```python
import model_runs

data_names = ['ionosphere', 'monk2', 'breat_cancer', 'climate']
height = 5
models = ['SCF', 'MCF', 'POKE-ALL', 'CUT-FRAC-3']
time_limit = 3600
rand_seed = [13, 58, 94, None]
priorities = ['data','equal']
file = 'biobj_example.csv'
consol_log = False
model_runs.multiobj(
  ["-d", data_names, "-h", height, "-m", models, "-t", time_limit,
   "-p", priorities, "-r", rand_seed, "-f", file, "-l", consol_log])
```
To run from terminal do the following,
```bash
python3 import model_runs; model_runs.multiobj -d ['ionosphere', 'monk2', 'breat_cancer', 'climate'] -h 5 -m ['SCF','MCF','POKE-ALL','CUT-FRAC-3'] -t 3600 -p ['data','equal'] -r [13, 58, 94, None] -f 'biobj_example.csv' -l False
```

***
## Pareto Frontier
To generate the Pareto frontier call the `main` function in `pareto_runs.py` with the below parameters:
  - d : str list, names of dataset files
  - h : int, maximum depth of trained trees
  - t : float, gurobi model time limit in s
  - m : str list, models to use
  - r : str list, random state(s) to use
  - f : str, results output file .csv

A `.png` file for each dataset called by the user is generated and stored in `\results_figures\` folder

We assume `-f file` is located in the `\results_files` folder
- If results output file `-f file` is `None` the `pareto` function automatically generates a `.csv` results file with the parameters of the function call as the file name saved to the `\results_files` folder

You can generate pareto frontiers from within a python file as follows,

```python
import model_runs

height = 4
models = ['FlowOCT', 'SCF', 'MCF', 'POKE', 'CUT']
rand_states = [15, 78, 0]
data_names = ['hayes_roth', 'house_votes_84']
file = 'pareto_example.csv'
model_runs.pareto(["-d", data_names, "-h", height, "-m", models, "-t", 3600, "-r", rand_states, "-f", file])
```

To run from terminal do the following 
```bash
python3 import model_runs; model_runs.pareto -d ['hayes_roth', 'house_votes_84'] -h 4 -m ['FOCT', 'SCF', 'MCF', 'POKE', 'CUT'] -t 3600 -r [15, 78, 0] -f 'pareto_example.csv'
```
- Note: `FlowOCT` must be the model name to generate the pareto frontier of FlowOCT
***

## Models Functionality
For understanding model functionality associated with integer and fractional separation procedures in **POKE** and **CUT** models, `-e model_extras` and `-c tuning` functionality please refer to the `USAGE.md` file. 


`example_code.py` contains additional instances of the above and how to call `OBCT` directly without using `model_runs.py`
***

## Acknowledgments
The code found in `BendersOCT.py`, `FlowOCT.py`, `FlowOCTTree.py,` and `FlowOCTutils.py` are taken directly from the [Strong Tree](https://github.com/pashew94/StrongTree/) GitHub public repository.
The code found in `Quant-BnB-2D.jl`, `Quant-BnB-3D.jl`, `Algorithms.jl`, `lowerbound_middle.jl` are taken directly from the [Quant-BnB](https://github.com/mengxianglgal/Quant-BnB) GitHub public repository.

All rights and ownership are to the original owners. 

***
***

![Screenshot](cmor_logo.png)
