from scipy.integrate import quad
import scipy.stats as stats
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def integral_bsm_with_coupons(call, underlying, days, rf, ivol, tradingDayYear, couponRate):
    # time to maturity
    ttm = days / tradingDayYear

    # daily volatility with continuously compounded implied volatility
    dailyVol = ivol / np.sqrt(tradingDayYear)

    # std dev and mean for log normal distribution
    sigma = np.sqrt(days) * dailyVol
    mu = np.log(underlying) + ttm * rf - 0.5 * sigma**2

    # log normal distribution
    d = stats.lognorm(scale=np.exp(mu), s=sigma)

    # calculate the present value of coupons
    couponPV = 0.0
    for day in range(int(ttm * tradingDayYear)):
        # present value of the coupon payment for each day, 
         couponPV += couponRate * np.exp(-rf * (day / tradingDayYear))

    if call:
        # option value for call
        def f(x):
            return (max(0, x - strike) + couponPV) * d.pdf(x)
        val, _ = quad(f, 0, underlying * 2)
    else:
        # option value for put
        def g(x):
            return (max(0, strike - x) + couponPV) * d.pdf(x)
        val, _ = quad(g, 0, underlying * 2)

    return val * np.exp(-rf * ttm)

# provided values
underlying_price = 165
strike = 170
start_date = datetime(2023, 3, 3)
options_expiration_date = datetime(2023, 3, 17)
days_to_maturity = (options_expiration_date - start_date).days
trading_days_in_year = 365
risk_free_rate = 0.0525
continuously_compounding_coupon_rate = 0.0053

implied_volatilities = np.linspace(0.1, 0.8, 100)

call_option_values = []
put_option_values = []

# calculate option values for each implied volatility
for ivol in implied_volatilities:
    # calculate the European call option value
    call_option_value = integral_bsm_with_coupons(True, underlying_price, days_to_maturity, risk_free_rate, ivol, trading_days_in_year, continuously_compounding_coupon_rate)
    call_option_values.append(call_option_value)

    # calculate the European put option value
    put_option_value = integral_bsm_with_coupons(False, underlying_price, days_to_maturity, risk_free_rate, ivol, trading_days_in_year, continuously_compounding_coupon_rate)
    put_option_values.append(put_option_value)

plt.figure(figsize=(12, 6))

plt.plot(implied_volatilities, call_option_values, label='Call Option')
plt.plot(implied_volatilities, put_option_values, label='Put Option', color='red')

plt.title('European Call and Put Option Values with Continuously Compounding Coupon')
plt.xlabel('Implied Volatility')
plt.ylabel('Option Value')
plt.legend()

plt.tight_layout()
plt.show()