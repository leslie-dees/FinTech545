import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np
from scipy.stats import spearmanr, norm
from itertools import product

test_portfolio = pd.read_csv("testfiles/data/test9_1_portfolio.csv")
test_returns = pd.read_csv("testfiles/data/test9_1_returns.csv")
testout_data = pd.read_csv("testfiles/data/testout9_1.csv")

prices = {
    'A': 20,
    'B': 30
}

# fit distributions based on what are provided with for the typing
model_a = fin.fit_normal(test_returns.A)
mu, sigma, nu, model_b = fin.fit_general_t(test_returns.B)

models = {
    'A': model_a,
    'B': model_b
}

nSim = 100000

# U values from the distributions (cdfs)
# Example U values (replace with your actual U values)
U = np.array([models["A"].u, models["B"].u])

# Calculate Spearman rank-order correlation coefficient for each pair of columns
spcor_matrix, _ = spearmanr(U, axis=1)

# Calculate Spearman rank-order correlation coefficient for each pair of values
def elementwise_spearmanr(x, y):
    rho, _ = spearmanr(x, y)
    return rho

# Create an empty matrix to store the results
spcor_matrix = np.zeros((len(U), len(U)))

# Calculate element-wise Spearman rank-order correlation coefficients
for i in range(len(U)):
    for j in range(len(U)):
        spcor_matrix[i, j] = elementwise_spearmanr(U[i], U[j])



uSim = fin.simulate_pca(spcor_matrix, 100000)

# Back to 0-1 values
uSim = norm.cdf(uSim)

simRet = {
    'A': models['A'].evaluate(uSim[:, 0]),
    'B': models['B'].evaluate(uSim[:, 1])
}

portfolio = pd.DataFrame({
    'Stock': ["A", "B"],
    'currentValue': [2000.0, 3000.0]
})

# However many simulations
nSim = 10000
iteration = list(range(1, nSim + 1))

# Full simulation, need to apply returns to each of the holdings
values = pd.DataFrame(list(product(portfolio['Stock'], portfolio['currentValue'], iteration)),
                      columns=['Stock', 'currentValue', 'iteration'])

# Calculate simulated values and P&L
nv = len(values)
simulated_value = np.zeros(nv)
pnl = np.zeros(nv)

for i in range(nv):
    stock = values['Stock'][i]
    iteration = values['iteration'][i]
    
    simulated_value[i] = values['currentValue'][i] * (1 + simRet[stock][iteration - 1])
    pnl[i] = simulated_value[i] - values['currentValue'][i]

values['pnl'] = pnl
values['simulatedValue'] = simulated_value

a = values[values['Stock'] == 'A']['pnl']

def aggRisk(df, stock_column):
    unique_stocks = df[stock_column].unique()

    dfs = []

    for stock in unique_stocks:
        stock_data = df[df[stock_column] == stock]

        simulated_var = fin.VaR_simulation(stock_data['pnl'])
        simulated_es = fin.ES_simulation(stock_data['pnl'])

        var_95_pct = simulated_var / stock_data['currentValue'].iloc[0]
        es_95_pct = simulated_es / stock_data['currentValue'].iloc[0]

        stock_df = pd.DataFrame({
            'Stock': [stock],
            'VaR95': [simulated_var],
            'ES95': [simulated_es],
            'VaR95_Pct': [var_95_pct],
            'ES95_Pct': [es_95_pct]
        })

        # Append the DataFrame to the list
        dfs.append(stock_df)

    # Concatenate the list of DataFrames into the final result_df
    result_df = pd.concat(dfs, ignore_index=True)

    # Calculate total VaR and total ES on the entire DataFrame
    total_var = fin.VaR_simulation(df['pnl'])
    total_es = fin.ES_simulation(df['pnl'])

    # Calculate VaR and ES as percentages of the total currentValue
    total_var_pct = total_var / df['currentValue'].sum()
    total_es_pct = total_es / df['currentValue'].sum()

    # Add a row for the total across all stocks
    result_df = result_df.append({
        'Stock': 'Total',
        'VaR95': total_var,
        'ES95': total_es,
        'VaR95_Pct': total_var_pct,
        'ES95_Pct': total_es_pct
    }, ignore_index=True)

    return result_df


result = aggRisk(values, 'Stock')
print(result)
print(testout_data)