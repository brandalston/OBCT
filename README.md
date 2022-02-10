# Optimal Binary Classification Trees

Code for the paper "Mixed Integer Linear Optimization Formulations for Learning Optimal Binary Classification Trees".

This code uses python3 (version 3.6 and higher) and requires the Gurobi9.x solver. Required packages are outlined in `requirements.txt`.

*** 
***
# Summary of Repository
- `results_figures` contains the generated .png files for experimental results
- `results_data_files` contains the generated .csv files for experimental results
- `Datasets` contains the datasets used for generating experimental results. It should also be used as the folder where user datasets are stored

***
***

## Running Code

- Ensure the latest versions of the packages in `requirements.txt` are installed
- For  running an instance of OBCT run the main function in `model_runs.py` with the following arguments
    - d : list of name of data files
    - h : list of maximum depth of the tree
    - t : time limit in s
    - m : list of models to use
    - e : list of model extras, if applicable
    - r : number of repeats
    - f : results output file

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

By default, models CUT1 and CUT2 invoke gurobi lazy parameters = 3 for separation constraints. There are a number of ways to invoke the fractional separation procedures outlined in the paper.

For fractional procedures use the following syntax where # specifies the user type of fractional cut (1,2,3)
- CUT1-FRAC-#
- CUT1-FRAC-#-ROOT for only adding user cuts at the root node of the branch and bound tree
- 1 = all violating cuts in 1,v path
- 2 = first found violating cut in 1,v path
- 3 = most violating cut closest to root in 1,v path
- The fractional separation procedure is independent of the CUT model specified (i.e. can mix and match)

Replace `models` in `TEST_CODE.py` with the following for fractional separation in CUT models
```python
models = ['CUT1-FRAC1','CUT2-FRAC-3-ROOT']
```

# Strong Optimal Classification Trees

![Screenshot](logo.png)

Code for the paper ''[Strong Optimal Classification Trees](https://sites.google.com/view/sina-aghaei/home)''.

The code will run in python3 ( version 6 and higher) and require [Gurobi9.x](https://www.gurobi.com/downloads/gurobi-optimizer-eula/) solver. The required packages are listed in `requirements.txt`.

***

## Overview

The content of this repository is as follows:

- `Code` contains the implementation of approaches **FlowOCT** and **BendersOCT**:

    - `FlowOCT.py` and `BendersOCT.py` include the formulation of each approach in gurobipy.

    - `FlowOCTReplication.py` and `BendersOCTReplication.py` contains the replication code for each approach which is responsible for creating and solving the problem and producing result

    - We have some utility files as follows
      - `Tree.py` which creates the binary tree
      - `logger.py` logs the output of console into a text file
- `Results` contains the experiments raw results.

- `plots` contains the R script for generating figures and tables from the tabular results.

- `DataSets` contains all datasets used for the experiments in the paper


***

## How to Run the Code

- First install the required packages in the `requirements.txt`
- For  running an instance of FlowOCT (BendersOCT) run the main function in `FlowOCTReplication.py` (`BendersOCTReplication.py`) with the following arguments
    - f : name of the dataset
    - d : maximum depth of the tree
    - t : time limit
    - l : value of _lambda
    - i : sample (for shuffling and splitting data)
    - c : 1 if we do calibration; in this case we train our model on 50% of the data otherwise we train on 75% of the data

You can call the main function within a python file as follows (This is what we do in `run_exp.py`

```python
import FlowOCTReplication
FlowOCTReplication.main(["-f", 'monk1_enc.csv', "-d", 2, "-t", 3600, "-l", 0, "-i", 1, "-c", 1])
```
or you could run it from terminal    

```bash
python3 FlowOCTReplication.py -f monk1_enc.csv -d 2 -t 3600 -l 0 -i 1 -c 1
```

The result of each instance would get stored in a csv file in `./Results/`.

Remember that in the replication files we assume that the name of the class label column is `target`. If you want to use a dataset of your own choice,
change the hard-coded label name to the desired name.
