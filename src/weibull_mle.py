### Weibull fit for non-interactive regime based on multiple csv-datasets ###

# Non-linear regression for Weibull distribution using maximum likelihood estimation (MLE)
# Takes datasets from the \data folder and fits Weibull parameters to each dataset
import numpy as np
import pandas as pd #For reading csv files
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12, 6)  # Set default figure size
from matplotlib import rcParams
rcParams['axes.prop_cycle'] = plt.cycler(color=['#0800ff', '#ff0000', '#00ff04', '#fff700', '#1f77b4', '#ff7f0e', '#2ca02c', '#7066e4'])
from scipy.stats import weibull_min

### Load data from CSV
all_data = pd.read_csv("data\in_situ_all.csv", delimiter=';', decimal=",")
# all_data is a pandas DataFrame containing the data from the CSV file
#print(all_data.head())
#print(all_data.columns)

### Find all unique specimens
specimens = all_data['specimen'].unique()
#print("Specimens found:", specimens)

group_colors = { # Dictionary to map specimen groups to colors
    "S_600_140": "tab:blue",
    "S_600_95": "tab:orange",
    "S_600_65": "tab:green",
    "S_450_140": "tab:red",
    "S_450_95": "tab:purple",
    "S_450_65": "tab:brown",
    "S_300_140": "tab:pink",
    "S_300_95": "tab:gray",
    "S_300_65": "tab:olive",
    "S_200_95": "tab:cyan",
    "S_200_65": "#1f99ff",
    "S_140_140": "#e2b007",
    "S_140_65": "#66c2a5",
    "S_95_95": "#a50f15",
    "S_65_65": "#4d4d4d",
    "H_600_140": "#2b8cbe",
    "H_140_140": "#843c39",
    "H_65_65": "#fdbf6f"
}

def get_group(specimen): # Function to extract the group from the specimen name
    # Example: group is everything before the second underscore
    # e.g., "S_600_140_1" -> "S_600_140"
    return "_".join(specimen.split("_")[:3])
    # Splits the specimen name by "_" and joins the first three parts

plotted_groups = set() # Set to keep track of groups that have already been plotted
# This is used to avoid labeling the same group multiple times in the legend

weibull_results = []

for s in specimens:
    try: # Tries to process the specimen or print an error message
        print(f"Processing specimen: {s}")

        group = get_group(s)
        color = group_colors.get(group, "black") # Default color if group not found

        ### Filter data for the current specimen and sort by strain
        data = all_data[all_data['specimen'] == s].sort_values(by='strain')
        strain_data_raw = pd.to_numeric(data['strain'], errors='coerce').to_numpy()
        crack_density_data_raw = pd.to_numeric(data['crack_density'], errors='coerce').to_numpy()
        # to_numeric converts every value in the column to a numeric type (float)
        # errors='coerce' replaces all non-convertible values with NaN
        # Converts the column to a numpy array

        ### Filter out NaN values
        mask = np.isfinite(strain_data_raw) & np.isfinite(crack_density_data_raw)
        # mask is a boolean array where True indicates finite values in both strain and crack density data
        strain_data = strain_data_raw[mask] # Filtered strain data
        crack_density_data = crack_density_data_raw[mask] # Filtered crack density data

        ### Filter out negative strain values (not physically meaningful)
        if np.any(strain_data < 0):
            print("⚠️  Negative strain value detected — set to 0.") # Print warning
            strain_data[strain_data < 0] = 0 # Set negative strain values to 0

        ### Convert crack density to micro-failure events (integer cumulative crack counts)
        l_total = 100 # Total length of the sample in mm
        cracks = np.round(crack_density_data * l_total).astype(int) # Round mathematically and then convert to integer counts
        # cracks is a numpy array as the datasets are made of numpy arrays
        # regular numpy arrays consist of floats, but diff, insert, repeat require integer values
        print("cracks:", cracks)

        ### Compute per-interval increments (number of micro-failures per strain level)
        crack_increments = np.diff(np.insert(cracks, 0, 0)).astype(int)
        # insert adds a 0 at the beginning of the cracks-array to represent the initial state (no cracks)
        # diff computes the difference between consecutive elements, giving the number of new cracks at each strain level

        # Remove negative cracking increments — no physical meaning for reduction of cracks
        crack_increments = np.clip(crack_increments, 0, None) # Clip values between 0 and None (no upper limit)

        # Reconstruct local failure strains: repeat each strain value by the number of cracks at that level
        local_strains = np.repeat(strain_data, crack_increments)
        # local_strains contains the strain value for each individual crack (failure strains)
        # Each strain value is repeated according to the number of new cracks at that strain level
        # repeat creates an array by repeating each element of strain_data according to the corresponding value in crack_increments
        # For example, if strain_data = [10, 20, 30] and crack_increments = [0, 2, 1], then local_strains = [20, 20, 30]

        #print(f"crack_increments: {crack_increments}")
        #print(f"local strains: {local_strains}")

        ### Give warning if no cracks are detected
        if len(local_strains) == 0:
            print(f"⚠️ No micro-failures reconstructed for specimen {s}. Skipping.")
            continue

        ### Give warning if non-finite values are detected
        if not np.all(np.isfinite(local_strains)):
            print(f"⚠️ Non-finite values in local_strains for specimen {s}. Skipping.")
            continue

        ### Give warning if no variation in local strains
        if np.std(local_strains) < 1e-6:
            print(f"⚠️ Barely any variation in specimen {s}. Skipping.")
            continue

        ### Give warning if too few new cracks are detected
        if np.sum(crack_increments) < 5:
            print(f"⚠️ Too few new cracks for a robust distribution for specimen {s}. Skipping.")
            continue

        ### Print total number of cracks
        print(f"Total reconstructed micro-failures: {len(local_strains)}")
        #Length of local_strengths gives the total number of cracks across all strain levels

        ### Two-parameter Weibull with fixed location parameter
        shape, loc, scale = weibull_min.fit(local_strains, floc=0)
        # fit uses maximum likelihood estimation (MLE) to fit the Weibull distribution to the data
        # floc=0 fixes the location parameter to 0, as only the scale and shape parameters are of interest
        # weibull_min instead of weibull_max because it considers the earliest failure/ the weakest element

        weibull_results.append([s, shape, scale])

        print(f"Estimated distribution shape constant (m): {shape:.4f}")
        print(f"Estimated characteristic strain (epsilon₀): {scale:.4f} MPa\n")
        #print(f"Estimated location parameter (l): {loc:.4f} MPa\n")

        ### Adding Weibull parameters of each specimen to the plot
        x = np.linspace(min(local_strains), max(local_strains), 200)
        # Generates 200 evenly spaced values between the minimum and maximum of local_strains as the x-axis
        y = weibull_min.pdf(x, shape, loc=loc, scale=scale)
        # Computes Weibull PDF for each strain value in x

        label = group if group not in plotted_groups else None
        # If the group has not been plotted yet, use it as the label; otherwise, set label to None

        plt.hist(local_strains, bins=10, density=True, alpha=0.5, label=label, color=color)
        # Plots a histogram of the probability density (y) for the failure strains (x)
        # Each bar represents the frequency of crack intiation at a given strain level
        # This is an empirically estimated PDF to be compared with the analytically estimated PDF
        # density=True normalizes the histogram to make it comparable in scale to a probability density function (PDF)
        # bins splits the data into intervals/bars
        # alpha=0.6 sets the transparency of the histogram bars
        plt.plot(x, y, color=color)
        # Plots the PDF based on x and y

        if label:
            plotted_groups.add(group) # Add the group to the set of plotted groups

    except Exception as e:
        print(f"Error processing specimen {s}: {e}")

### Plotting
plt.xlabel("Strain")
plt.ylabel("Probability Density")
plt.legend(fontsize=10, loc='upper right', ncol=1)
plt.title("Weibull Fit Across Multiple Layups")
plt.grid(True)
plt.tight_layout()
plt.show()

# Average the results for some specimens to compare results for different layups
# Continue with Monte Carlo simulation for Weibull distribution