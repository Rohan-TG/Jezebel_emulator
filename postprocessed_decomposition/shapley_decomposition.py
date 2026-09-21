import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import shapiq
from sklearn.linear_model import LinearRegression

nominal_directory = input("Nominal XS directory: ")

feature_names = [
	"Pu9 elastic",
	"Pu9 inelastic",
	"Pu9 (n,2n)",
	"Pu9 fission",
	"Pu9 capture",

	"Pu0 elastic",
	"Pu0 inelastic",
	"Pu0 (n,2n)",
	"Pu0 fission",
	"Pu0 capture",

	"Pu1 elastic",
	"Pu1 inelastic",
	"Pu1 (n,2n)",
	"Pu1 fission",
	"Pu1 capture",
]

directory = '/home/rnt26/uncertaintyanalysis/ml/mldata/pchip-data/0-15999'
files = os.listdir(directory)

# for getting the right columns etc.
exampledf = pd.read_parquet(os.path.join(directory, files[0]))

cols = exampledf.columns
cols = cols[1:-2] # remove unnecessary columns e.g. erg etc.

nonrealmatrix = [[] for i in range(0,15)] # list per channel

keff_values = [] # stores SCONE k_eff values
# main dataset loading
for f in tqdm(files, total=len(files)):
	df = pd.read_parquet(os.path.join(directory, f))
	df = df[df.ERG >= 2500]
	keff_values.append(df['keff'].values[0])

	# flat_array = []
	for ci, channel in enumerate(cols):
		nonrealmatrix[ci].append(df[channel].values)
		# flat_array += list(df[channel].values)

##############################################################################################################
# Nominal data loading and processing
keff_nominal = 0.99980
f_nominal = f'{nominal_directory}/endfbviii0_baseline_data_Pu-239_-1_Pu-240_-1_Pu-241_-1.parquet'
df_nominal = pd.read_parquet(f_nominal)
df_nominal = df_nominal[df_nominal.ERG >= 2500]

nominal_nonrealmatrix = [[] for i in range(0,15)]
for nominal_ci, nominal_channel in enumerate(cols):
	nominal_nonrealmatrix[nominal_ci].append(df_nominal[nominal_channel].values)



nominal_pcamatrix = []
nominal_mode_number = []

##############################################################################################################
# pca decomposition of dataset
pcamatrix = []
for channel, nominal_channel in zip(nonrealmatrix, nominal_nonrealmatrix):
	pca = PCA(n_components=0.999, svd_solver='full') # keep modes explaining 99.9% of variance
	X_pca = pca.fit_transform(channel)
	X_nominal_pca = pca.transform(nominal_channel)

	pcamatrix.append(X_pca)
	nominal_pcamatrix.append(X_nominal_pca)

for i in nominal_pcamatrix:
	nominal_mode_number.append(len(i[0]))


# Convert into the right shape (16000, n_total_modes)
flattened_pca_matrix = [[] for i in range(0, len(nonrealmatrix[0]))]
flattened_nominal_pca_matrix = [[] for i in range(0, len(nominal_nonrealmatrix[0]))]


for pca_channel in tqdm(pcamatrix, total=len(pcamatrix)):
	for sample_index, pca_sample in enumerate(pca_channel):
		flattened_pca_matrix[sample_index] += list(pca_sample)

# flatten nominal pca
for npc in nominal_pcamatrix:
	for sidx, pcasnominal in enumerate(npc):
		flattened_nominal_pca_matrix[sidx] += list(pcasnominal)

mode_number = []
for i in pcamatrix:
	mode_number.append(len(i[0]))

fpm = np.array(flattened_pca_matrix)
fnpm = np.array(flattened_nominal_pca_matrix)

X_delta = fpm - fnpm
###################################################################################################################

group_cols = {}
start = 0
groups = []
for idx, (name, mode_n) in enumerate(zip(feature_names, mode_number)):
	group_cols[name] = list(range(start, start + mode_n))
	groups.append(list(range(start, start + mode_n)))
	start += mode_n

n_groups = len(group_cols)

fpm = np.array(flattened_pca_matrix)
# Begin r^2 cooperative game
cache = {}

delta_k = np.array(keff_values) - keff_nominal # ∆keff

nominal_ss = np.sum(delta_k**2)


def r2_game(coalitions):
	"""coalitions are one of the (2C1)^15 possible combinations of groups where the possible 2 choices are
	including or excluding a channel from the coalition"""
	coalitions = np.asarray(coalitions, dtype=bool)
	if coalitions.ndim == 1:
		coalitions = coalitions.reshape(1, -1)

	values = np.zeros(coalitions.shape[0])

	for row, coalition in tqdm(enumerate(coalitions), total=len(coalitions)):
		key = tuple(coalition.tolist())

		if key in cache:
			values[row] = cache[key]
			continue

		selected_groups = np.flatnonzero(coalition) # remove unused groups

		if len(selected_groups) == 0:
			value = 0.0
			cache[key] = value
			values[row] = value
			continue

		cols = np.concatenate([groups[g] for g in selected_groups])

		X_delta_subset = X_delta[:, cols]

		model = LinearRegression(fit_intercept=False)
		model.fit(X_delta_subset, delta_k) # fit linear regression to PCA modes

		delta_k_pred = model.predict(X_delta_subset)

		residual_ss = np.sum((delta_k - delta_k_pred) ** 2) # prediction error
		value = 1.0 - residual_ss / nominal_ss
		cache[key] = value
		values[row] = value

	return values



computer = shapiq.ExactComputer(r2_game, n_players=15)
sv = computer(index="SV", order=1)

full_coalition = np.ones((1, 15), dtype=bool)
full_value = r2_game(full_coalition)[0]
print("Full explained fraction:", full_value)








