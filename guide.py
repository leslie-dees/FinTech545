# Covariance:
# # Missing data:   test 1.1
# # Pairwise:   test 1.3
# # Exponentially Weighted: test 2.1
"calculate_ewma_covariance_matrix"

# Correlation:
# # Missing data:   test 1.2
# # Pairwise:   test 1.4
# # Exponentiall Weighted:  test 2.2
"calculate_ewma_correlation_matrix"

# Near PSD:
# # Standard:   test 3.1
# # Higham:     test 3.3
# # Cholesky PSD: test 4.1
"chol_psd"
"is_psd"
"near_psd"
"higham_nearestPSD"

# Simulations:
# # Normal:     test 5.1
# # PCA:        test 5.5
"simulate_pca" # for use in Gaussian copula
"multivariate_normal_simulation"

# Returns:
# # Arithmetic: test 6.1
# # Log:        test 6.2
"return calculate"

# Fitting:
# # Normal distribution:    test 7.1
# # T distribution:         test 7.2
# # T regression:           test 7.3

# Value at Risk (VaR):
# # From Normal:    test 8.1
# # From T:         test 8.2
# # From simulation:    test 8.3
# # Portfolio:
"VaR_error_model"
"VaR_simulation"

# Expected Shortfall (ES):
# # From Normal:    test 8.4
# # From T:         test 8.5
# # from Simulation:    test 8.6
"ES_error_model"
"ES_simulation"

# Gaussian Copula:
# # test 9.1

# OLS:
# # On T:    test 7.3
"fit_regression_t"

# Skewness, Kurtosis:
"first4Moments"

# Maximum Likelihood Estimation (MLE):
"mle_normal_distribution_one_input"
"mle_t_distribution_one_input"

# Prices:
"calculate_prices"

# Options:
# # BSM:
# # # Integral:
"integral_bsm_with_coupons" # not for closed form
# # # Closed form price:
"options_price"
# # Option price error:
"option_price_error"

# Implied volatility:
"calculate_implied_volatility"

# Greeks:
# # Functional
"greeks"
# # Finite Difference:
"greeks_df"
# # With dividends:
"greeks_with_dividends"

# Binomial tree
# # European
"binomial_tree_pricing_european"
# # American
"binomial_tree_pricing_american"