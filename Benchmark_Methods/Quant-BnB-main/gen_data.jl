using LinearAlgebra
using JSON
using StatsBase
using Random

function tree_eval(tree, X, D, m)
    n, p = size(X)
    y0 = zeros(n, m)
    if length(tree) != 4 || typeof(tree[3]) == typeof(1.0)
        return y0 .+ reshape(tree, (1,m))
    end
    f = tree[1]
    b = tree[2]
    idx1, idx2 = treesplit(x -> x<b, X[:,f])
    y0[idx1,:] = tree_eval(tree[3], X[idx1,:], D-1, m)
    y0[idx2,:] = tree_eval(tree[4], X[idx2,:], D-1, m)
    return y0    
end

function generate_realdata(name, train_size=0.75, seed=0)
    Random.seed!(seed)

    if occursin("quant", name)
        realdata = JSON.parsefile(string("./Quant-BnB-main/dataset/class/",replace(name,"quant-" => "" ),".json"))
        fullX = reduce(vcat, [realdata["Xtrain"], realdata["Xtest"]])
        fullY = reduce(vcat, [realdata["Ytrain"], realdata["Ytest"]])
        full_index = 1:size(fullXdata)[1]
        train_index = shuffle(full_index)[1:floor(Int, train_size*size(fullXdata)[1])]
        test_index = filter(e->!(e in train_index),full_index)
    
        X_train = zeros(size(train_index)[1], realdata["F"])
        X_test = zeros(size(test_index)[1], realdata["F"])
        Y_train = zeros(size(train_index)[1], realdata["C"])
        Y_test = zeros(size(test_index)[1], realdata["C"])

        counter = 1
        for i in train_index
            X_train[counter,:] = fullX[i]
            Y_train[counter, Int(fullY[i]) + 1] = 1
            counter += 1
        end
        
        counter = 1
        for i in test_index
            X_test[counter,:] = fullX[i]
            Y_test[counter, Int(fullY[i]) + 1] = 1
            counter += 1
        end
    
        n, p = size(X_train)
        goodfeature = Vector{Int64}()
        for i = 1:p
            if length(unique(X_train[:,i])) >= 2
                append!(goodfeature, i)
            end
        end
    
        X_train = X_train[:,goodfeature]
        X_test = X_test[:,goodfeature]
    else
        data = DataFrame(CSV.File(pwd()*"/Datasets/"*name*"_enc.csv"))
        full_index = 1:size(data)[1]
        train_index = shuffle(full_index)[1:floor(Int, train_size*size(data)[1])]
        test_index = filter(e->!(e in train_index),full_index)
        features = filter(e->!(e in ["target"]), names(data))
        fullX, fullY = Matrix(data[:,features]), data[:,:target]

        X_train = zeros(size(train_index)[1], size(features)[1])
        X_test = zeros(size(test_index)[1], size(features)[1])
        Y_train = zeros(size(train_index)[1], size(unique!(data[!,:target]))[1])
        Y_test = zeros(size(test_index)[1], size(unique!(data[!,:target]))[1])
    

        counter = 1
        for i in train_index
            X_train[counter,:] = fullX[i,:]
            Y_train[counter, Int(fullY[i])+1] = 1
            counter += 1
        end
        
        counter = 1
        for i in test_index
            X_test[counter,:] = fullX[i,:]
            Y_test[counter, Int(fullY[i])+1] = 1
            counter += 1
        end
    
        n, p = size(X_train)
        goodfeature = Vector{Int64}()
        for i = 1:p
            if length(unique(X_train[:,i])) >= 2
                append!(goodfeature, i)
            end
        end
    
        X_train = X_train[:,goodfeature]
        X_test = X_test[:,goodfeature]

    end
    return X_train, X_test, Y_train, Y_test
end
