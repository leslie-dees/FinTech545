import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin

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
    call_option_value = fin.integral_bsm_with_coupons(True, underlying_price, strike, days_to_maturity, risk_free_rate, ivol, trading_days_in_year, continuously_compounding_coupon_rate)
    call_option_values.append(call_option_value)

    # calculate the European put option value
    put_option_value = fin.integral_bsm_with_coupons(False, underlying_price, strike, days_to_maturity, risk_free_rate, ivol, trading_days_in_year, continuously_compounding_coupon_rate)
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