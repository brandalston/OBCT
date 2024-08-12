import model_runs
from Benchmark_Methods import FB_OCT, DL85#, GOSDTg

"""
model_runs.main(["-d", ['iris','soybean_small'], "-h", [2], "-m", ['CUT1','MCF1','MCF2','CUT2'],
                 "-t", 3600, "-r", [15], "-f", 'test_runs.csv', "-l", False, "-e", None])

FB_OCT.main(["-d", ['iris','soybean_small'], "-h", [2], "-m", ['Flow', 'Benders'],
             "-t", 3600, "-r", [15], "-f", 'test_runs.csv', "-l", False])

DL85.main(["-d", ['iris','soybean_small'], "-h", [2],
            "-t", 3600, "-r", [15], "-f", 'test_runs.csv'])

GOSDTg.main(["-d", ['iris','soybean_small'], "-h", [2],
            "-t", 3600, "-r", [15], "-f", 'test_runs.csv'])
"""

models = ['MCF1', 'MCF2', 'CUT1', 'CUT2']
priorities = ['data', 'equal']
model_runs.biobjective(["-d", ['soybean_small'], "-h", 5, "-m", models, "-t", 3600,
                     "-p", priorities, "-r", [15,48,100,42], "-f", 'biobj_example.csv', "-l", False])


#model_runs.pareto(["-d", ['soybean_small'], "-h", 5, "-m", ['CUT1','CUT2','MCF1','MCF2'],
#                   "-t", 3600, "-r", [15,42,138,], "-f", 'pareto_test.csv'])