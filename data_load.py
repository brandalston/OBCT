import pandas as pd

cols_dict = {
        'auto-mpg': ['target', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model-year',
                     'origin', 'car_name'],
        'balance-scale': ['target', 'left-weight', 'left-distance', 'right-weight', 'right-distance'],
        'banknote_authentication': ['variance-of-wavelet', 'skewness-of-wavelet', 'curtosis-of-wavelet', 'entropy',
                                    'target'],
        'blood_transfusion': ['R', 'F', 'M', 'T', 'target'],
        'breast-cancer': ['target', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig',
                          'breast', 'breast-quad', 'irradiat'],
        'car': ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'target'],
        'climate': ['Study', 'Run', 'vconst_corr', 'vconst_2', 'vconst_3', 'vconst_4', 'vconst_5', 'vconst_7',
                    'ah_corr', 'ah_bolus', 'slm_corr', 'efficiency_factor', 'tidal_mix_max', 'vertical_decay_scale',
                    'convect_corr', 'bckgrnd_vdc1', 'bckgrnd_vdc_ban', 'bckgrnd_vdc_eq', 'bckgrnd_vdc_psim',
                    'Prandtl', 'target'],
        'flare1': ['class', 'largest-spot-size', 'spot-distribution', 'activity', 'evolution',
                   'previous-24hr-activity', 'historically-complex', 'become-h-c', 'area', 'area-largest-spot',
                   'c-target', 'm-target', 'x-target'],
        'flare2': ['class', 'largest-spot-size', 'spot-distribution', 'activity', 'evolution',
                   'previous-24hr-activity', 'historically-complex', 'become-h-c', 'area', 'area-largest-spot',
                   'c-target', 'm-target', 'x-target'],
        'glass': ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'target'],
        'hayes-roth': ['name', 'hobby', 'age', 'educational-level', 'marital-status', 'target'],
        'house-votes-84': ['target', 'handicapped-infants', 'water-project-cost-sharing',
                           'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid',
                           'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
                           'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending',
                           'superfund-right-to-sue', 'crime', 'duty-free-exports',
                           'export-administration-act-south-africa'],
        'image_segmentation': ['target', 'region-centroid-col', 'region-centroid-row', 'region-pixel-count',
                               'short-line-density-5', 'short-line-density-2', 'vedge-mean', 'vegde-sd',
                               'hedge-mean', 'hedge-sd', 'intensity-mean', 'rawred-mean', 'rawblue-mean',
                               'rawgreen-mean', 'exred-mean', 'exblue-mean', 'exgreen-mean', 'value-mean',
                               'saturatoin-mean', 'hue-mean'],
        'ionosphere': list(range(1, 35)) + ['target'],
        'iris': ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'target'],
        'kr-vs-kp': ['bkblk', 'bknwy', 'bkon8', 'bkona', 'bkspr', 'bkxbq', 'bkxcr', 'bkxwp', 'blxwp', 'bxqsq',
                     'cntxt', 'dsopp', 'dwipd', 'hdchk', 'katri', 'mulch', 'qxmsq', 'r2ar8', 'reskd', 'reskr',
                     'rimmx', 'rkxwp', 'rxmsq', 'simpl', 'skach', 'skewr', 'skrxp', 'spcop', 'stlmt', 'thrsk',
                     'wkcti', 'wkna8', 'wknck', 'wkovl', 'wkpos', 'wtoeg', 'target'],
        'monk1': ['target', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'],
        'monk2': ['target', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'],
        'monk3': ['target', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'],
        'parkinsons': ['name', 'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                       'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                       'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'target', 'RPDE',
                       'DFA', 'spread1', 'spread2', 'D2', 'PPE'],
        'soybean-small': list(range(1, 36)) + ['target'],
        'tic-tac-toe': ['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square',
                        'middle-middle-square', 'middle-right-square', 'bottom-left-square', 'bottom-middle-square',
                        'bottom-right-square', 'target'],
        'wine-red': ['fixed-acidity', 'volatile-acidity', 'citric-acid', 'residual-sugar', 'chlorides',
                     'free-sulfur dioxide', 'total-sulfur-dioxide', 'density', 'pH', 'sulphates', 'alcohol',
                     'target'],
        'wine-white': ['fixed-acidity', 'volatile-acidity', 'citric-acid', 'residual-sugar', 'chlorides',
                       'free-sulfur dioxide', 'total-sulfur-dioxide', 'density', 'pH', 'sulphates', 'alcohol',
                       'target']
    }


def load_acute_inflammations(decision_number):
    """ Load the Acute Inflammations dataset.

    Contains a mix of numerical and categorical attributes. Decided to not use
    this dataset in the paper for this reason.

    https://archive.ics.uci.edu/ml/datasets/Acute+Inflammations
    """
    if decision_number not in [1, 2]:
        raise ValueError("problem_number must be 1 or 2")
    df = pd.read_csv("Datasets/diagnosis.data",
                     names=["a1", "a2", "a3", "a4", "a5", "a6", "d1", "d2"], decimal=',',
                     encoding='utf-16', delim_whitespace=True)
    y = df["d{}".format(decision_number)]
    X = df.drop(columns=["d1", "d2"])
    return X, y


def load_acute_inflammations_1():
    """ Load the Acute Inflammations dataset with decision 1 as the label.

    Contains a mix of numerical and categorical attributes.
    """
    return load_acute_inflammations(1)


def load_acute_inflammations_2():
    """ Load the Acute Inflammations dataset with decision 1 as the label.

    Contains a mix of numerical and categorical attributes.
    """
    return load_acute_inflammations(2)


def load_balance_scale():
    """ Load the Balance Scale dataset.

    Contains only categorical attributes.

    https://archive.ics.uci.edu/ml/datasets/Balance+Scale
    """
    names = ["target", "left weight", "left distance", "right weight", "right distance"]
    df = pd.read_csv("Datasets/balance-scale.data", names=names)
    y = df["target"]
    X = df.drop(columns="target")
    return X, y


def load_banknote_authentication():
    """ Load the Banknote Authentication dataset.

    Contains only numerical attributes.

    https://archive.ics.uci.edu/ml/datasets/banknote+authentication
    """
    names = ["variance", "skewness", "curtosis", "entropy", "target"]
    df = pd.read_csv("Datasets/banknote_authentication.data", names=names)
    y = df["target"]
    X = df.drop(columns="target")
    return X, y


def load_blood_transfusion():
    """ Load the Blood Transfusion dataset.

    Contains only numerical attributes.

    https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
    """
    names = ["recency", "frequency", "monetary", "time", "target"]
    df = pd.read_csv("Datasets/blood_transfusion.data", header=0, names=names)
    y = df["target"]
    X = df.drop(columns="target")
    return X, y


def load_breast_cancer():
    """ Load the Breast-cancer dataset.

       Contains only categorical attributes. Dataset already partitions examples
       into train and test sets.

       https://archive.ics.uci.edu/ml/datasets/breast+cancer
       """
    df = pd.read_csv("Datasets/breast-cancer.data",
                     names=["target", "age", "menopause", "tumor size",
                            "inv-nodes", "node-caps", "deg-malig",
                            "breast", "breast-quad","irradiat"])
    y = df["target"]
    X = df.drop(columns="target")
    return X, y


def load_car_evaluation():
    """ Load the Car Evaluation dataset.

    Contains only categorical attributes.

    https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
    """
    names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "target"]
    df = pd.read_csv("Datasets/car.data", names=names)
    y = df["target"]
    X = df.drop(columns="target")
    return X, y


def load_chess():
    """ Load the Chess (King-Rook vs. King-Pawn) dataset.

    Contains only categorical attributes.

    https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King-Pawn%29
    """
    names = list("a{}".format(j + 1) for j in range(36)) + ["target"]
    df = pd.read_csv("Datasets/kr-vs-kp.data", names=names)
    y = df["target"]
    X = df.drop(columns="target")
    return X, y


def load_climate_model_crashes():
    """ Load the Climate Model Crashes dataset.

    Contains only numerical attributes.

    https://archive.ics.uci.edu/ml/datasets/climate+model+simulation+crashes
    """
    names = ['Study', 'Run', 'vconst_corr', 'vconst_2', 'vconst_3', 'vconst_4', 'vconst_5', 'vconst_7',
             'ah_corr', 'ah_bolus', 'slm_corr', 'efficiency_factor', 'tidal_mix_max', 'vertical_decay_scale',
             'convect_corr', 'bckgrnd_vdc1', 'bckgrnd_vdc_ban', 'bckgrnd_vdc_eq', 'bckgrnd_vdc_psim',
             'Prandtl', 'target']
    df = pd.read_csv("Datasets/climate.data", names=names)
    y = df["target"]
    X = df.drop(columns=["Study", "Run", "target"])
    return X, y


def load_congressional_voting_records():
    """ Load the Congressional Voting Records dataset.

    Contains only categorical attributes.

    https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
    """
    names = list("a{}".format(j + 1) for j in range(17))
    df = pd.read_csv("Datasets/house_votes_84.data", names=names)
    y = df["a1"]
    X = df.drop(columns="a1")
    return X, y


def load_fico_binary():
    df = pd.read_csv("Datasets/fico_binary_enc.csv")
    y = df["target"]
    X = df.drop(columns="target")
    return X, y


def load_glass_identification():
    """ Load the Glass Identification dataset.

    Contains only numerical attributes.

    https://archive.ics.uci.edu/ml/datasets/glass+identification
    """
    df = pd.read_csv("Datasets/glass.data",
                     names=["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "target"])
    y = df["target"]
    X = df.drop(columns="target")
    return X, y


def load_hayes_roth():
    """ Load the Hayes-Roth dataset.

    Contains only categorical attributes. Dataset already partitions examples
    into train and test sets.

    https://archive.ics.uci.edu/ml/datasets/Hayes-Roth
    """
    df_train = pd.read_csv("Datasets/hayes_roth.data",
                     names=["name", "hobby", "age", "educational level", "marital status", "target"],
                     index_col="name")
    df_test = pd.read_csv("Datasets/hayes_roth.test",
                     names=["hobby", "age", "educational level", "marital status", "target"])
    df = pd.concat([df_test, df_train], ignore_index=True)
    y = df["target"]
    X = df.drop(columns="target")
    return X, y


def load_image_segmentation():
    """ Load the Image Segmentation dataset.

    Contains only numerical attributes. Dataset already partitions examples
    into train and test sets.

    http://archive.ics.uci.edu/ml/datasets/image+segmentation
    """
    names = ["target"] + list("a{}".format(j + 1) for j in range(19))
    df = pd.read_csv("Datasets/image_segmentation.data", skiprows=5, names=names)
    y = df["target"]
    X = df.drop(columns="target")
    return X, y


def load_ionosphere():
    """ Load the Ionosphere dataset.

    Contains only numerical attributes.

    https://archive.ics.uci.edu/ml/datasets/ionosphere
    """
    names = list("a{}".format(j + 1) for j in range(34)) + ["target"]
    df = pd.read_csv("Datasets/ionosphere.data", names=names)
    y = df["target"]
    X = df.drop(columns="target")
    return X, y


def load_iris():
    names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "target"]
    df = pd.read_csv("Datasets/iris.data", names=names)
    y = df["target"]
    X = df.drop(columns="target")
    return X, y


def load_monk(problem_number):
    """ Load the MONK's Problems dataset.

    Contains only categorical attributes. Test set is a "full" set of examples
    and the training set is simply a subset of the test set.

    https://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems
    """
    if problem_number not in {1, 2, 3}:
        raise ValueError("problem_number must be 1, 2, or 3")
    train_file = "Datasets/monks-{}.train".format(problem_number)
    df_train = pd.read_csv(train_file,
                     names=["target", "a1", "a2", "a3", "a4", "a5", "a6", "Id"],
                     index_col="Id",
                     delim_whitespace=True)
    df_train = df_train.reset_index(drop=True)
    test_file = "Datasets/monks-{}.test".format(problem_number)
    df_test = pd.read_csv(test_file,
                           names=["target", "a1", "a2", "a3", "a4", "a5", "a6", "Id"],
                           index_col="Id",
                           delim_whitespace=True)
    df_test = df_test.reset_index(drop=True)
    df = pd.concat([df_test, df_train], ignore_index=True)
    y = df_train["target"]
    X = df_train.drop(columns="target")
    return X, y


def load_monk1():
    """ Load the MONK's problem 1 dataset.

    Contains only categorical attributes. Test set is a "full" set of examples
    and the training set is simply a subset of the test set.
    """
    return load_monk(1)


def load_monk2():
    """ Load the MONK's problem 2 dataset.

    Contains only categorical attributes. Test set is a "full" set of examples
    and the training set is simply a subset of the test set.
    """
    return load_monk(2)


def load_monk3():
    """ Load the MONK's problem 3 dataset.

    Contains only categorical attributes. Test set is a "full" set of examples
    and the training set is simply a subset of the test set.
    """
    return load_monk(3)


def load_parkinsons():
    """ Load the Parkinsons dataset.

    Contains only numerical attributes.

    https://archive.ics.uci.edu/ml/datasets/parkinsons
    """
    names = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
             'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
             'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'target', 'RPDE',
             'DFA', 'spread1', 'spread2', 'D2', 'PPE']
    df = pd.read_csv("Datasets/parkinsons.data", names=names)
    y = df["target"]
    X = df.drop(columns="target")
    return X, y


def load_soybean_small():
    """ Load the Soybean (Small) dataset.

    Contains only categorical attributes.

    https://archive.ics.uci.edu/ml/datasets/Soybean+%28Small%29
    """
    names = list("a{}".format(j + 1) for j in range(35)) + ["target"]
    df = pd.read_csv("Datasets/soybean-small.csv", names=names, header=0)
    y = df["target"]
    X = df.drop(columns="target")
    return X, y


def load_spect():
    names = ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12",
             "F13","F14","F15","F16","F17","F18","F19","F20","F21","F22","target"]

    df = pd.read_csv("Datasets/spect.data", names=names)
    y = df["target"]
    X = df.drop(columns="target")
    return X, y


def load_tictactoe_endgame():
    """ Load the Tic-Tac-Toe Endgame dataset.

    Contains only categorical attributes.

    https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame
    """
    names = list("a{}".format(j + 1) for j in range(9)) + ["target"]
    df = pd.read_csv("Datasets/tic-tac-toe.data", names=names)
    y = df["target"]
    X = df.drop(columns="target")
    return X, y


def load_wine_red():
    names = ['fixed-acidity', 'volatile-acidity', 'citric-acid', 'residual-sugar', 'chlorides',
             'free-sulfur dioxide', 'total-sulfur-dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'target']
    df = pd.read_csv("Datasets/wine-red.data", names=names)
    y = df["target"]
    X = df.drop(columns="target")
    return X, y


def load_wine_white():
    names = ['fixed-acidity', 'volatile-acidity', 'citric-acid', 'residual-sugar', 'chlorides',
             'free-sulfur dioxide', 'total-sulfur-dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'target']
    df = pd.read_csv("Datasets/wine-white.data", names=names)
    y = df["target"]
    X = df.drop(columns="target")
    return X, y
