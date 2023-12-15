import Pkg; Pkg.add("CSV")

using CSV
using Distributions
using Plots
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

    beta = collect(B)
    xm = __y - __x*beta
    print(size(xm))

    # #Inner function to abstract away the X value
    # function _gtl(mu,s,nu,B...)
    #     beta = collect(B)
    #     xm = __y - __x*beta
    #     print(xm)
    #     general_t_ll(mu,s,nu,xm)
    # end

    # register(mle,:tLL,nB+3,_gtl;autodiff=true)
    # @NLobjective(
    #     mle,
    #     Max,
    #     tLL(m, s, nu, B...)
    # )
    # optimize!(mle)

    # m = value(m) #Should be 0 or very near it.
    # s = value(s)
    # nu = value(nu)
    # beta = value.(B)

    # #Define the fitted error model
    # errorModel = TDist(nu)*s+m

    # #function to evaluate the model for a given x and u
    # function eval_model(x,u)
    #     n = size(x,1)
    #     _temp = hcat(fill(1.0,n),x)
    #     return _temp*beta .+ quantile(errorModel,u)
    # end

    # #Calculate the regression errors and their U values
    # errors = y - eval_model(x,fill(0.5,size(x,1)))
    # u = cdf(errorModel,errors)

    # return FittedModel(beta, errorModel, eval_model, errors, u)
end

# 7.3 Fit T Regression
cin = CSV.read("testfiles/data/test7_3.csv",DataFrame)
fd = fit_regression_t(cin.y,Matrix(select(cin,Not(:y))))