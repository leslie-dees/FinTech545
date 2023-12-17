"""
HW Week 2:

    1: -1 Need to test hypothesis of Skewness, only kurtosis provided
    2: -1.5
        - Need to report OLS beta value
        - No need to perform normality hypothesis on MLE fitting. You are given the normality assumption and T distribution assumption.
          All you should do is to get the optimal distribution parameters under the assumption.
          Need to report MLE for T distribution result
          Need to report MLE for normality result
        - Need to calculate AIC and BIC. No calculation provided in both code and report.
   
    3: -0.25 ACF and PCF should differ for the MA processes.
"""

"""
HW Week 3:

    1. -0.5 Reverse weights in EWA, data is in oldest to newest.
            Covariance matrix calculation is incorrect.
    2. -0.1 Need to calculate the norm on the output matrices to the input, not between the two outputs.
            Looking to understand how far the "fix" changed the matrix
    3. Full Credit.  Issue in EW Covariance is throwing off norm values.
"""

"""
HW Week 4:

    1. -1 Only need simulation, not based on the dataset. You are doing multiple time forward, only single time needed. Returns are calculated correctly.
    2. -0.5
        -- VaR is the left tail of the distribution, and we express it as a positive number.  Unless stated below, values are correct but the sign should by positive
        - -0.3 Perform the ARIMA regression first to get parameters, then based on the parameters, do the simulation to calculate VaR.
        - -0.2 Only need 1 day forward sample simulation for the META return  The value should not be so different from other values.  You can use those values
              To sanity check your numbers
    3. -0.35
        - -0.25 Standard deviation calculation is wrong for delta normal method.  You are not accounting for the correlations – you need to do the matrix multiplication against the full covariance matrix.
        - -0.1  Changing to a different lambda is the same method.  This can be a valid change, but you need to explain why you would do this.

"""

"""
HW Week 5:
    1. -0.5
        note: mu for the T fit is incorrect, should be about -9.4e-5
        -0.5 ES for the Normal is incorrect, check your function.
    2. Full Credit
    3. -2  
        VaR and ES need to be calculated using the fitted T distribution, the Gaussian copula to
        account for correlation, and a simulation method.

"""

"""
HW Week 6
    1. -1.5 your option calculation function is incorrect.  No need to do the integral when you have a closed form solution.
    2. Call graph is incorrect likely because of the issue above.
       +1 for discussion
    3. -1.5 Graphs are incorrect, often times inverse from expected.  You might have a sign error.  Needed to calculate VaR/ES for each portfolio.
"""

"""
HW Week 7

    1. -1 No option price or implied volatility is given.  This question was impossible to answer.
        No one caught this.
    2. -1.5 VaR and ES values wrong as discussed in class.  AR is OK.  Work with TAs if needed.
    3. -0.5 You did not need to fit all 10 years for the factors.  Just use the data provided.
       Needed to annualize covariance matrix based on the problem.
       Present the portfolio weights in a readable format.
       Looks like you did the optimization correctly.

"""