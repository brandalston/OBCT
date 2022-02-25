# Optimal Binary Classification Trees

Code for the paper ["Mixed Integer Linear Optimization Formulations for Learning Optimal Binary Classification Trees"](hyperlink) by Brandon Alston, Hamidreza Validi, and Illya V. Hicks.

This code uses python3 (version 3.6 and higher) and requires the [Gurobi9.x](https://www.gurobi.com/downloads/gurobi-optimizer-eula/) solver. Required python packages are outlined in `requirements.txt`.

*** 
***

## Summary of Repository
- `Code` contains the code of the formulations **MCF1**, **MCF2**, **CUT1**, **CUT2** and **AGHA**
  - `OBCT.py` contains the formulations of each model for solving using Gurobi9.x
  - `model_runs.py` contains the code necessary to create, solve, and report results of each instance called by the user
  - `TREE.py` creates the necessary tree information including path, child, and other information
  - `SPEED_UP.py` contains the code for the callbacks used in user fractional separation procedures
  - `RESULTS.py` contains the code for viewing model decision variable results and generating the .csv results files among other model results
- `results_files/` folder stores the generated .csv files with model metrics
- `results_figures/` folder stores the generated .png files for experimental results
- `Datasets/` folder contains the datasets used for generating experimental results. It should also be used as the folder where user datasets are stored

***
***

## Running Code

- Ensure the latest versions of the packages in `requirements.txt` are installed
- For  running an instance of OBCT run the `main` function in `model_runs.py` with the following arguments
    - d : str list, names of dataset files
    - h : int list, maximum depth of trained trees
    - t : float, gurobi model time limit in s
    - m : str list, models to use
    - e : str list, model extras, if applicable
    - r : int, number of repeat trees to generate for each model
    - c : str, tuning parameter
    - f : str, results output file .csv

You can call the `model_runs.py` main function within a python file as follows,

```python
import model_runs
data_names = ['monk1_enc','breast-cancer_enc']
heights = [2,3,4,5]
models = ['MCF1','MCF2','CUT1','CUT2']
time_limit = 3600
extras = ['fixing','max_features-15']
repeats = 5
tuning = None
file = 'results.csv'
model_runs.main(["-d",data_names,"-h",heights,"-m",models,"-t",time_limit,"-e",extras,"-r",repeats,"-c", tuning,"-f",file])
```

To run from terminal do the following,
```bash
python3 model_runs.py -d ['monk1_enc','breast-cancer_enc'] -h [2,3,4,5] -m ['MCF1','MCF2','CUT1','CUT2'] -t 3600 -e ['fixing','max_features_15'] -r 5 -c tuning -f 'results.csv'
```
Note:
- We assume the target column is labeled `'target'`. Change the code in `model_runs.py` to change the label column
- If results output file `-f file` is `None` the `models_run.py` automatically generates a `.csv` results file with the parameters of the function call as the file name
- `-e model_extras`, `-c tuning`, and `-f file` may be `None` input arguments, all others must hold a valid value

***
## Pareto Frontier
To generate the Pareto frontier call the pareto_runs.main() function with the below parameters:
  - d : str list, names of dataset files
  - h : int, maximum depth of trained trees
  - t : float, gurobi model time limit in s
  - m : str list, models to use
  - f : str, results output file .csv

You can generate pareto frontiers from within a python file as follows,
```python
import pareto_runs
height = 5
models = ['AGHA','MCF1','MCF2','CUT1','CUT2']
repeats = 5
data_names = ['house-votes-84_enc']
file = 'pareto.csv'
pareto_runs.main(["-d",data_names,"-h",height,"-m",models,"-t",3600,"-r",repeats,"-f",file])
```

To run from terminal do the following 
```bash
python3 pareto_runs.py -d ['monk1_enc','soybean-small_enc'] -h 5 -m ['MCF1','MCF2','CUT1','CUT2'] -t 3600 -r 5 -c tuning -f 'pareto.csv'
```
A `.png` file with the pareto_runs functional parameter call is generated and stored in `\results_figures\` folder
***

## Models Functionality
For understanding model functionality associated with integer and fractional separation procedures in CUT1 and CUT2 models, `-e model_extras` and `-c tuning` functionality please refer to the `USAGE.md` file. 

***

## Acknowledgments
The code found in `BendersOCT.py FlowOCT.py FlowOCTTree.py FlowOCTutils.py` are taken directly from the [Strong Tree](https://github.com/pashew94/StrongTree/) GitHub public repository.
All rights and ownership are to the original owners. 

***
***

![Screenshot](CAAM_logo.png)
