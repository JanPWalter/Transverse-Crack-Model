### Weibull fit for non-interactive regime based on multiple csv-datasets ###

#Non-linear regression for Weibull distribution minimizing the sum of squares of residuals
# Takes datasets from the \data folder and fits Weibull parameters to each dataset
# Plots the data and fitted Weibull curves for each dataset

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd #For reading csv files
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  # Set default figure size

def crack_density_model(stress, sigma0, m, l0=0.1):
    return (1/l0) * (1 - np.exp(-(stress/sigma0)**m)) #Defines non-linear crack density function
# sigma0: characteristic stress, m: Weibull modulus, l0: crack spacing scale

def fit_weibull(stress, crack_density, l0=0.1): #Fitting non-linear model to data
    sigma0_guess = np.percentile(stress, 75)  # avoids extreme guesses (uses 75th percentile of stress data)
    m_guess = 2.0  # typical for brittle materials
    popt, _ = curve_fit(lambda x, sigma0, m: crack_density_model(x, sigma0, m, l0),
                        stress, crack_density,
                        p0=[sigma0_guess, m_guess], #provides an initial guess for the optimization
                        bounds=([1e-3, 0.1], [np.inf, 20])) # avoids non-physical values (negative or too large values)
    #curve_fit minimizes the sum of squares of residuals between the observed and predicted values
    # using the Levenberg-Marquardt algorithm
    return popt  # sigm0, m

# List of filepaths
datasets = {
    "Example 1": "data\example1_crack_data.csv",
    "Example 2": "data\example2_crack_data.csv",
    "Example 3": "data\example3_crack_data.csv",
    "Example 4": "data\example4_crack_data.csv",
    "Beispiel": "data\Mappe1.csv"
}

weibull_results = {} # To store Weibull parameters for each dataset

for label, filepath in datasets.items():
    # Load data from CSV
    data = pd.read_csv(filepath, delimiter=';')
    stress_data = data['stress'].to_numpy()
    crack_density_data = data['crack_density'].to_numpy()

    # Fit Weibull model
    sigma0_fit, m_fit = fit_weibull(stress_data, crack_density_data, l0=0.1)
    fitted_density = crack_density_model(stress_data, sigma0_fit, m_fit)

    # Store results
    weibull_results[label] = {
        'sigma0': sigma0_fit,
        'm': m_fit
    }
    plt.plot(stress_data, crack_density_data, 'o', label=f"{label} data")
    plt.plot(stress_data, fitted_density, '-', label=f"{label} fit with\nm={m_fit:.2f} and σ₀={sigma0_fit:.2f} MPa")
    print(f"Stress data for {label}: {stress_data}")
    print(f"Crack density data for {label}: {crack_density_data}")

# Plotting

plt.xlabel("Stress [MPa]")
plt.ylabel("Crack Density [1/mm]")
plt.legend(fontsize=10, loc='upper left')
plt.title("Weibull Fit Across Multiple Layups")
plt.grid(True)
plt.tight_layout()
plt.show()

# Printing Weibull parameters for each dataset
for label, results in weibull_results.items():
    print(f"{label}: shape parameter m = {results['m']:.4f}, characteristic strength σ₀ = {results['sigma0']:.4f}")