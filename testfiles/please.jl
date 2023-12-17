using CSV
using Distributions
using QuadGK
using DataFrames
using Ipopt
using JuMP
using StatsBase
using LinearAlgebra
using Random

#Type to hold model outputs
struct FittedModel
    beta::Union{Vector{Float64},Nothing}
    errorModel::UnivariateDistribution
    eval::Function
    errors::Vector{Float64}
    u::Vector{Float64}
end


#general t sum ll function
function general_t_ll(mu,s,nu,x)
    td = TDist(nu)*s + mu
    sum(log.(pdf.(td,x)))
end

#fit regression model with T errors
function fit_regression_t(y,x)
    n = size(x,1)

    global __x, __y
    __x = hcat(fill(1.0,n),x)
    __y = y

    nB = size(__x,2)

    mle = Model(Ipopt.Optimizer)
    set_silent(mle)

    #approximate values based on moments and OLS
    b_start = inv(__x'*__x)*__x'*__y
    e = __y - __x*b_start
    start_m = mean(e)
    start_nu = 6.0/kurtosis(e) + 4
    start_s = sqrt(var(e)*(start_nu-2)/start_nu)

    @variable(mle, m, start=start_m)
    @variable(mle, s>=1e-6, start=1)
    @variable(mle, nu>=2.0001, start=start_s)
    @variable(mle, B[i=1:nB],start=b_start[i])

    #Inner function to abstract away the X value
    function _gtl(mu,s,nu,B...)
        beta = collect(B)
        xm = __y - __x*beta
        general_t_ll(mu,s,nu,xm)
    end

    register(mle,:tLL,nB+3,_gtl;autodiff=true)
    @NLobjective(
        mle,
        Max,
        tLL(m, s, nu, B...)
    )
    optimize!(mle)

    m = value(m) #Should be 0 or very near it.
    s = value(s)
    nu = value(nu)
    beta = value.(B)

    #Define the fitted error model
    errorModel = TDist(nu)*s+m

    #function to evaluate the model for a given x and u
    function eval_model(x,u)
        n = size(x,1)
        _temp = hcat(fill(1.0,n),x)
        return _temp*beta .+ quantile(errorModel,u)
    end

    #Calculate the regression errors and their U values
    errors = y - eval_model(x,fill(0.5,size(x,1)))
    u = cdf(errorModel,errors)

    return FittedModel(beta, errorModel, eval_model, errors, u)
end

#MLE for a Generalize T
function fit_general_t(x)
    global __x
    __x = x
    mle = Model(Ipopt.Optimizer)
    set_silent(mle)

    #approximate values based on moments
    start_m = mean(x)
    start_nu = 6.0/kurtosis(x) + 4
    start_s = sqrt(var(x)*(start_nu-2)/start_nu)

    @variable(mle, m, start=start_m)
    @variable(mle, s>=1e-6, start=1)
    @variable(mle, nu>=2.0001, start=start_s)

    #Inner function to abstract away the X value
    function _gtl(mu,s,nu)
        general_t_ll(mu,s,nu,__x)
    end

    register(mle,:tLL,3,_gtl;autodiff=true)
    @NLobjective(
        mle,
        Max,
        tLL(m, s, nu)
    )
    optimize!(mle)

    m = value(m)
    s = value(s)
    nu = value(nu)

    #create the error model
    errorModel = TDist(nu)*s + m
    #calculate the errors and U
    errors = x .- m
    u = cdf(errorModel,x)

    eval(u) = quantile(errorModel,u)

    return FittedModel(nothing, errorModel, eval, errors, u)

    #return the parameters as well as the Distribution Object
    return (m, s, nu, TDist(nu)*s+m)
end


function fit_normal(x)
    #Mean and Std values
    m = mean(x)
    s = std(x)
    
    #create the error model
    errorModel = Normal(m,s)
    #calculate the errors and U
    errors = x .- m
    u = cdf(errorModel,x)

    eval(u) = quantile(errorModel,u)

    return FittedModel(nothing, errorModel, eval, errors, u)

end

function simulate_pca(a, nsim; pctExp=1, mean=[],seed=1234)
    n = size(a,1)

    #If the mean is missing then set to 0, otherwise use provided mean
    _mean = fill(0.0,n)
    m = size(mean,1)
    if !isempty(mean)
        copy!(_mean,mean)
    end

    #Eigenvalue decomposition
    vals, vecs = eigen(a)
    vals = real.(vals)
    vecs = real.(vecs)
    #julia returns values lowest to highest, flip them and the vectors
    flip = [i for i in size(vals,1):-1:1]
    vals = vals[flip]
    vecs = vecs[:,flip]
    
    tv = sum(vals)

    posv = findall(x->x>=1e-8,vals)
    if pctExp < 1
        nval = 0
        pct = 0.0
        #figure out how many factors we need for the requested percent explained
        for i in 1:size(posv,1)
            pct += vals[i]/tv
            nval += 1
            if pct >= pctExp 
                break
            end
        end
        if nval < size(posv,1)
            posv = posv[1:nval]
        end
    end
    vals = vals[posv]

    vecs = vecs[:,posv]

    # println("Simulating with $(size(posv,1)) PC Factors: $(sum(vals)/tv*100)% total variance explained")
    B = vecs*diagm(sqrt.(vals))

    Random.seed!(seed)
    m = size(vals,1)
    r = randn(m,nsim)

    out = (B*r)'
    #Loop over itereations and add the mean
    for i in 1:n
        out[:,i] = out[:,i] .+ _mean[i]
    end
    return out
end

# 9.1
# loading in data
cin = CSV.read("testfiles/data/test9_1_returns.csv",DataFrame)
prices = Dict{String,Float64}()
# getting prices
prices["A"] = 20.0
prices["B"] = 30

# dictionary for models
# fit distributions based on what we are provided with the types of distributions of the data
models = Dict{String,FittedModel}()
models["A"] = fit_normal(cin.A)
models["B"] = fit_general_t(cin.B)

# getting close number through lots of simulations
nSim = 100000

# U values from the distributions (matrix of the u values from fitted models)
U = [models["A"].u models["B"].u]
# take correlation of the u values, use spearman bc its a rank correlation, if we use pearson we would need to transform into z scores
spcor = corspearman(U)
# if std dev are 1, then corr is covar matrix
uSim = simulate_pca(spcor,nSim)
# back to 0-1 values, are the quantiles pretty much
uSim = cdf.(Normal(),uSim)

# using fitted distribution to get p value to get something on that distribution
simRet = DataFrame(:A=>models["A"].eval(uSim[:,1]), :B=>models["B"].eval(uSim[:,2]))

portfolio = DataFrame(:Stock=>["A","B"], :currentValue=>[2000.0, 3000.0])
# however many simulations
iteration = [i for i in 1:nSim]
# full simulation, need to apply returns to each of the holdings
values = crossjoin(portfolio, DataFrame(:iteration=>iteration))

nv = size(values,1)
# empty vectors of profit and loss, and the simulated vlaues
pnl = Vector{Float64}(undef,nv)
simulatedValue = copy(pnl)
for i in 1:nv
    simulatedValue[i] = values.currentValue[i] * (1 + simRet[values.iteration[i],values.Stock[i]])
    pnl[i] = simulatedValue[i] - values.currentValue[i]
end

values[!,:pnl] = pnl
values[!,:simulatedValue] = simulatedValue

risk = select(aggRisk(values,[:Stock]),[:Stock, :VaR95, :ES95, :VaR95_Pct, :ES95_Pct])

print(risk)