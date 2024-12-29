# MODEL FUNCTIONALITY
***
***
We offer the user the ability to control both integer and fractional separation using the three types of violating rules in models **POKE** and **CUT**. Note that **POKE** is equivalent to **CUT_W** in the paper. We invoke a different name in implementation due to the similarity in parsing the strings ``CUT, CUT_W``.

By default, models **POKE** and **CUT** invoke gurobi [lazy](https://www.gurobi.com/documentation/9.5/refman/lazy.html) parameters = 3, constraints that cut off the relaxation solution at the root node are pulled in for separation constraints.

***
### Fractional Separation
There are a number of ways to invoke the fractional separation procedures outlined in the paper. 
For fractional procedures use the following syntax where `#` specifies the user type of fractional cut (1,2,3)
- ex. `POKE-FRAC-1, CUT-FRAC-3-ROOT`
  - `#-ROOT` for only adding user fractional cuts at the root node of the branch and bound tree
- `1` = all violating cuts in 1,v path
- `2` = first found violating cut in 1,v path
- `3` = most violating cut closest to root of DT in 1,v path
- The separation procedure is independent of the CUT model specified (i.e. can mix and match)

***
### Integer Separation
For user controlled integral separation procedures use the following syntax where `#` specifies the user type of integer cut (1,2,3)
- ex. `POKE-INT-1, CUT-INT-3`
- `1` = all violating cuts in 1,v path
- `2` = first found violating cut in 1,v path
- `3` = most violating cut closest to root in 1,v path
- The separation procedure is independent of the CUT model specified (i.e. can mix and match)

***
### Integer & Fractional Separation
We also control integer and fractional separation using gurobi callback. To invoke such functionality use the following
- ex. `POKE-BOTH-I2-F2`, `CUT-BOTH-I1-F3`
- `#` specifies the violating rules type
- `BOTH` must be in the model name
- Must specify a rule type for both integral and fractional
  - The integral and fractional separation procedures are independent of the CUT model specified (i.e. can mix and match)
- Cannot use `FRAC-#`, `INT-#`, syntax must use `-BOTH-I#-F#`

***
## Model Extras
To invoke the `-e model_extras` parameter use the following guide. Each choice used should be a `str` placed in a `list` named `model_extras`
- ex. `model_extras = ['max_features-23','single use','num_features-12]`
- No more than `k` features used in the DT
  - `'max_features-k'`
- Exactly `k` features used in the DT
  - `'num_features-k'`
- Each future used once no more than k times in DT
  - `'repeat_use-k'`
- Require k datapoints in a classification node to apply regularization
  - `'regularization-k`

Note: `model_extras` are not applicable to models: **FOCT**, **FlowOCT**, **BendersOCT**
***
## Multiobjective Modeling
The -p priority parameter used in our multiobjective modeling. A choice of priority between ``data, equal, tree_size`` must be used
- We hard code a priority values of 10 and 5, giving the former to the priority choice given at the `model_runs.multiobj` function call
  - ``Equal``, priority value of 5 is given to both
- Change the hard code of `OBCT.py` to change the priority values


***
***

![Screenshot](cmor_logo.png)
