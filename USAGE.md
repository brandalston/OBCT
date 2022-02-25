# MODEL FUNCTIONALITY
***
***
We offer the user the ability to control both integer and fractional separation using the three types of violating rules in models CUT1 and CUT2.

By default, models CUT1 and CUT2 invoke gurobi lazy parameters = 3 for separation constraints. There are a number of ways to invoke the fractional separation procedures outlined in the paper.

For fractional procedures use the following syntax where `#` specifies the user type of fractional cut (1,2,3)
- ex. `CUT1-FRAC-1, CUT2-FRAC-3-ROOT`
- `CUT1-FRAC-#`
  - `-#-ROOT` for only adding user cuts at the root node of the branch and bound tree
- `1` = all violating cuts in 1,v path
- `2` = first found violating cut in 1,v path
- `3` = most violating cut closest to root in 1,v path
- The fractional separation procedure is independent of the CUT model specified (i.e. can mix and match)

We also control integer and fractional separation using gurobi callback. To invoke such functionality use the following
- ex. `CUT1-BOTH-I2-F2`, `CUT2-BOTH-I1-F3`, `CUT2-BOTH-I3-F1-ROOT`
- `#` specifies the violating rules type
- `BOTH` must be in the model name
- Must specify a rule type for both integral and fractional
  - The integral and fractional separation procedures are independent of the CUT model specified (i.e. can mix and match)
- Cannot use `FRAC-#`, `INT-#`, syntax must use `-BOTH-I#-F#`

***
### Model Extras
To invoke the `-e model_extras` parameter use the following guide. Each choice used should be a `str` placed in a `list` named `model_extras`
- ex. `model_extras = ['fixing','max_features_23','single use']`
- Fixing DV based on datapoint unreachable nodes
    - `'fixing'`
- No more than `k` features used in the DT
  - `'max_features_k`
- Conflict contraints
  - `'conflict_constraints'`
- Each future used once in DT
  - `'single_use'`
- Super feature relationship in parent, child branching vertices
  - `'super_feature'`
***
### Tuning and Warm Starting
- The `tuning` parameter is meant for warm starting the model for solution time and out-of-sample accuracy
  - ex.`-c 'calibration'` or `-c 'warm_start'`
    - Only one can be specified bewteen `calibration` and `warm_start`
- `warm_start`
  - Warm start models using randomly assigned tree
- `calibration`
  - Calibration which uses the second objective function as a constraint and generates the pareto frontier
  - First generate a 25% calibration training set for finding calibrated `max_features_k*` parameter
  - Each `max_features_k-1` DT is used as a warm for the `max_features_k` DT
  - Use `k`  in [ 1 , B ]
  - The best in-sample-accuracy `max_features_k*` is used as the calibrated decision tree
    - Note: `max_features_k*` replaces any user specified `max_features`
  - Train full model on 25% calibration model + 50% training model
  - Warm start full model using calibration tree assignments

***
***

![Screenshot](CAAM_logo.png)
