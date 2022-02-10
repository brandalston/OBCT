# Optimal Binary Classification Trees

Code for the paper [Mixed Integer Linear Optimization Formulations for Learning Optimal Binary Classification Trees"](hyperlink)

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

- `results_figures` contains the generated .png files for experimental results
- `results_files` stores the generated .csv files with model metrics
- `Datasets` contains the datasets used for generating experimental results. It should also be used as the folder where user datasets are stored

***
***

## Running Code

- Ensure the latest versions of the packages in `requirements.txt` are installed
- For  running an instance of OBCT run the `main` function in `model_runs.py` with the following arguments
    - d : list of names of dataset .csv files
    - h : list of maximum depth of trained trees
    - t : gurobi model time limit in s
    - m : list of models to use
    - e : list of model extras, if applicable
    - r : number of repeat trees to generate for each model
    - f : string of results output file .csv

You can call the `model_runs.py` main function within a python file as follows,

```python
import model_runs
data_names = ['data.csv']
heights = [2,3,4,5]
models = ['MCF1','MCF2', 'CUT1','CUT2']
time_limit = 3600
extras = ['fixing','max_features-15']
repeats = 5
file = 'results.csv'
model_runs.main(["-d",data_names,"-h",heights,"-m",models,"-t",time_limit,"-e",extras,"-r",repeats,"-f",file])
```

To run from terminal do the following,
```bash
python3 model_runs.py -d ['data.csv'] -h [2,3,4,5] -m ['MCF1','MCF2','CUT1','CUT2'] -t 3600 -e ['fixing','max_features-15'] -r 5 -f 'results.csv'
```
Note:
- We assume the target column is labeled 'target'
- Change the code in `model_runs.py` to change the label column
- If results output file `-f file` is `None` the `models_run.py` automatically generates a .csv results file with the parameters of the function call as the file name

***

### CUT Models Functionality

We offer the functionality to control both integer and fractional separation using the three types of violating rules in models CUT1 and CUT2.

By default, models CUT1 and CUT2 invoke gurobi lazy parameters = 3 for separation constraints. There are a number of ways to invoke the fractional separation procedures outlined in the paper.

For fractional procedures use the following syntax where `#` specifies the user type of fractional cut (1,2,3)
- `CUT1-FRAC-#`
  - `CUT1-FRAC-#-ROOT` for only adding user cuts at the root node of the branch and bound tree
- `1` = all violating cuts in 1,v path
- `2` = first found violating cut in 1,v path
- `3` = most violating cut closest to root in 1,v path
- The fractional separation procedure is independent of the CUT model specified (i.e. can mix and match)
- ex. `CUT1-FRAC-1, CUT2-FRAC-3-ROOT`

To invoke such functionality replace CUT models in `models` with the following
- ex. `CUT1-BOTH-I2-F2`, `CUT2-BOTH-I1-F3`
- `#` specifies the violating rules type
- `BOTH` must be in the model name
- Must specify a type for both integral and fractional
  - The integral and fractional separation procedures are independent of the CUT model specified (i.e. can mix and match)
- Cannot use `FRAC-#`, `INT-#` syntax must use `-BOTH-I#-F#`

***
***

![Screenshot](/Users/brandonalston/Desktop/CAAM_logo.png)
