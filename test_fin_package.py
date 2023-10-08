import pytest
import math
from fin_package import (
    first4Moments, #done
    calc_estimated_kurtosis, #don't need to test
    perform_ols,
    mle_normal_distribution,
    mle_t_distribution,
    mle_normal_distribution_one_input,
    mle_t_distribution_one_input,
    simulate_MA,
    simulate_AR,
    plot_acf_pacf,
    calc_var_normal,
    calc_var_t_dist,
    calc_expected_shortfall_normal,
    calc_expected_shortfall_t,
    calculate_ewma_covariance_matrix,
    chol_psd,
    near_psd,
    is_psd,
    higham_near_psd,
    multivariate_normal_simulation,
    simulate_and_print_norms,
    calculate_portfolio_var,
    return_calculate,
    portfolio_es
)

def test_first4Moments():
    mean = 0
    std_dev = 1
    known_kurtosis = 3
    known_skew = 0
    num_samples = 5000
    sample_size = 1000
    avg_est_kurt, _, avg_est_skew, _ = calc_estimated_kurtosis(num_samples, sample_size, mean, std_dev)
    assert math.isclose(avg_est_kurt, known_kurtosis, rel_tol = 0.01)

