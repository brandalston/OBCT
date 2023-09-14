include("Quant-BnB-main/QuantBnB-2D.jl")
include("Quant-BnB-main/QuantBnB-3D.jl")
include("Quant-BnB-main/gen_data.jl")
include("Quant-BnB-main/lowerbound_middle.jl")
include("Quant-BnB-main/Algorithms.jl")
using LinearAlgebra
using Printf
using CSV, Tables, DataFrames, Dates

cols = ["Data","H","|I|","Out-Acc","In-Acc","Sol-Time","Gap","ObjVal","ObjBound","Model","# CB",
        "User Cuts","Cuts/CB", "CB-Time", "INT-CB-time","FRAC-CB-TIME","CB-Eps","Time Limit",
        "Rand. State","Warm Start","Single Feature Use","Max Features"]
outfile = pwd()*"/results_files/paper_results.csv"
timelimit = 3600
rand_states = [138, 15, 89, 42, 0]
heights = [5]
datasets = ["ionosphere"]

summary = DataFrame([name => [] for name in cols])
for file in datasets
    for h in heights
        for seed in rand_states
            println("\n\nQuantBnB run\nDataset: ", file, ". H: ", h , ". Rand State: ", seed, ". Run Start: (", Dates.format(now(), "HH:MM"),")")
            X_train, X_test, Y_train, Y_test = generate_realdata(file,0.75,seed)
            n_train, m = size(Y_train)
            n_test, _ = size(Y_test)
            if h == 3
                gre_train, gre_tree = greedy_tree(X_train, Y_train, 3, "C")
                opt_train, opt_tree, opt_time = QuantBnB_3D(X_train, Y_train, 3, 3, gre_train*(1+1e-6), 0, 0, nothing, "C", timelimit, true)
            else
                gre_train, gre_tree = greedy_tree(X_train, Y_train, 2, "C")
                opt_train, opt_tree, opt_time = QuantBnB_2D(X_train, Y_train, 3, gre_train*(1+1e-6), 2, 0.2, nothing, "C", timelimit, true)
            end
                opt_test = sum((Y_test - tree_eval(opt_tree, X_test, 3, m)).>0)
            if opt_time < timelimit
                    println("Optimal solution found in ", round(opt_time;digits=4),"s")
            else
                    print("Time limit reached. ","(", Dates.format(now(), "HH:MM"),")")
            end
            push!(summary, [file, h, size(X_train)[1], 1-opt_test/size(X_test)[1], 1-opt_train/size(X_train)[1], opt_time,
            "N/A", "N/A", "N/A", "QuantBnB", timelimit, seed, "N/A", "N/A"])
            CSV.write(outfile, last(summary,1),append=true)
        end
    end
end