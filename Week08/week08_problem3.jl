using BenchmarkTools
using Distributions
using Random
using StatsBase
using DataFrames
import Pkg; Pkg.add("Query")
using Query
# using Plots
using LinearAlgebra
using JuMP
using Ipopt
using Dates
# using ForwardDiff
# using FiniteDiff
using CSV
# using LoopVectorization
# # using GLM

# include("../library/RiskStats.jl")
# include("../library/simulate.jl")
# include("../library/return_calculate.jl")
# include("../library/fitted_model.jl")

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

function VaR(a; alpha=0.05)
    x = sort(a)
    nup = convert(Int64,ceil(size(a,1)*alpha))
    ndn = convert(Int64,floor(size(a,1)*alpha))
    v = 0.5*(x[nup]+x[ndn])

    return -v
end

function VaR(d::UnivariateDistribution; alpha=0.05)
    -quantile(d,alpha)
end

function ES(a; alpha=0.05)
    x = sort(a)
    nup = convert(Int64,ceil(size(a,1)*alpha))
    ndn = convert(Int64,floor(size(a,1)*alpha))
    v = 0.5*(x[nup]+x[ndn])
    
    es = mean(x[x.<=v])
    return -es
end

function ES(d::UnivariateDistribution; alpha=0.05)
    v = VaR(d;alpha=alpha)
    f(x) = x*pdf(d,x)
    st = quantile(d,1e-12)
    return -quadgk(f,st,-v)[1]/alpha
end

ff3 = CSV.read("Week08/F-F_Research_Data_Factors_daily.CSV", DataFrame)
mom = CSV.read("Week08/F-F_Momentum_Factor_daily.CSV",DataFrame)
returns = CSV.read("Week08/DailyReturn.csv",DataFrame)

# Join the FF3 data with the Momentum Data
ffData = innerjoin(ff3,mom,on=:Date)
rename!(ffData, names(ffData)[size(ffData,2)] => :Mom)
rename!(ffData, Symbol("Mkt-RF")=>:Mkt_RF)
ffData[!,names(ffData)[2:size(ffData,2)]] = Matrix(ffData[!,names(ffData)[2:size(ffData,2)]]) ./ 100
ffData[!,:Date] = Date.(string.(ffData.Date),dateformat"yyyymmdd")

returns[!,:Date] = Date.(returns.Date,dateformat"mm/dd/yyyy")

#join the FF3+1 to Stock data - filter to stocks we want
stocks = [:AAPL, :MSFT, Symbol("BRK-B"), :CSCO, :JNJ]
to_reg = innerjoin(returns[!,vcat(:Date, :SPY, stocks)], ffData, on=:Date)


xnames = [:Mkt_RF, :SMB, :HML, :Mom]

#OLS Regression for all Stocks
X = hcat(fill(1.0,size(to_reg,1)),Matrix(to_reg[!,xnames]))

Y = Matrix(to_reg[!,stocks])
Betas = (inv(X'*X)*X'*Y)'
resid = Y - X*Betas

Betas = Betas[:,2:size(xnames,1)+1]

max_dt = max(to_reg.Date...)
min_dt = max_dt - Year(10)
to_mean = ffData |>  @filter(_.Date >= min_dt && _.Date <= max_dt) |> DataFrame

#historic daily factor returns
exp_Factor_Return = mean.(eachcol(to_mean[!,xnames]))
expFactorReturns = DataFrame(:Factor=>xnames, :Er=>exp_Factor_Return)


#scale returns and covariance to geometric yearly numbers
stockMeans =log.(1 .+ Betas*exp_Factor_Return)*255 
covar = cov(log.(1.0 .+ Y))*255

# Function for Portfolio Volatility
function pvol(w...)
    x = collect(w)
    return(sqrt(x'*covar*x))
end

# Function for Component Standard Deviation
function pCSD(w...)
    x = collect(w)
    pVol = pvol(w...)
    csd = x.*(covar*x)./pVol
    return (csd)
end

# Sum Square Error of cSD
function sseCSD(w...)
    csd = pCSD(w...)
    mCSD = sum(csd)/n
    dCsd = csd .- mCSD
    se = dCsd .*dCsd
    return(1.0e5*sum(se)) # Add a large multiplier for better convergence
end

n = length(stocks)

m = Model(Ipopt.Optimizer)

# Weights with boundry at 0
@variable(m, w[i=1:n] >= 0,start=1/n)
register(m,:distSSE,n,sseCSD; autodiff = true)
@NLobjective(m,Min, distSSE(w...))
@constraint(m, sum(w)==1.0)
optimize!(m)

w = value.(w)

RPWeights = DataFrame(:Stock=>String.(stocks), :Weight => w, :cEr => stockMeans .* w, :CSD=>pCSD(w...))
println(RPWeights)


# RP on Simulated ES
#remove the mean
m = mean.(eachcol(Y))
Y = Y .- m'

#Fit T Models to the returns
n = size(Y,2)
m = size(Y,1)
models = Vector{FittedModel}(undef,n)
U = Array{Float64,2}(undef,m,n)
for i in 1:n
    models[i] = fit_general_t(Y[:,i])
    U[:,i] = models[i].u
end

nSim = 5000

# Gaussian Copula -- Technically we should do 255 days ahead...
corsp = corspearman(U)
_simU = cdf.(Normal(),rand(MvNormal(fill(0.0,n),corsp),nSim))'
simReturn = similar(_simU)

for i in 1:n
    simReturn[:,i] = models[i].eval(_simU[:,i])
end

# internal ES function
function _ES(w...)
    x = collect(w)
    r = simReturn*x
    es = ES(r)
    print(es)
    prt(asdf)
    ES(r)
end

# Function for the component ES
function CES(w...)
    x = collect(w)
    n = size(x,1)
    ces = Vector{Any}(undef,n)
    es = _ES(x...)
    e = 1e-6
    for i in 1:n
        old = x[i]
        x[i] = x[i]+e
        ces[i] = old*(_ES(x...) - es)/e
        x[i] = old
    end
    ces
end

# SSE of the Component ES
function SSE_CES(w...)
    ces = CES(w...)
    ces = ces .- mean(ces)
    1e3*(ces'*ces)
end
    

#Optimize to find RP based on Expected Shortfall
n = length(stocks)

m = Model(Ipopt.Optimizer)

# Weights with boundry at 0
@variable(m, w[i=1:n] >= 0,start=1/n)
register(m,:distSSE,n,SSE_CES; autodiff = true)
@NLobjective(m,Min, distSSE(w...))
@constraint(m, sum(w)==1.0)
optimize!(m)

w = value.(w)

ES_RPWeights = DataFrame(:Stock=>String.(stocks), :Weight => w, :cEr => stockMeans .* w, :CES=>CES(w...))
println(ES_RPWeights)
println(RPWeights)