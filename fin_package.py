import scipy.stats as stats


def kurtosis_and_bias(data):
    # Calculate kurtosis from scipy kurtosis function
    kurtosis_value = stats.kurtosis(data)

    print("Kurtosis: ", kurtosis_value)

    return kurtosis_value

