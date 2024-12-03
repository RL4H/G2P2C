import os
import sys
from decouple import config

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)
import pandas as pd
import numpy as np
from idtxl.bivariate_mi import BivariateMI
from idtxl.multivariate_mi import MultivariateMI
from idtxl.bivariate_te import BivariateTE

from idtxl.data import Data
from idtxl.visualise_graph import plot_network
import matplotlib.pyplot as plt

cohorts = ['adult', 'adolescent', 'child']
seeds = ['1', '2', '3']
# seeds = ['3']

subjects = ['6']  # , '1', '2', '3', '4', '5', '6', '7', '8', '9']
# subjects = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


cohort = 'adolescent'

n_trials = 5  # 500


def combine_data(cohort, sub):
    print("=" * 100)
    print("Starting for patient {}".format(sub))

    df_combined = pd.DataFrame()

    d = []
    columns = ['x12', 'x11', 'x10', 'x9', 'x8', 'x7', 'x6', 'x5', 'x4', 'x3', 'x2', 'x1', 'i12', 'i11', 'i10', 'i9',
               'i8', 'i7', 'i6', 'i5', 'i4', 'i3', 'i2', 'i1', 'y']

    for trial in range(0, n_trials):
        worker_id = int(trial + 6000)
        for seed in seeds:
            # PATH1 = MAIN_PATH + '/results/' + cohort + '/PPO/P' + sub + '_' + seed + '/testing/data/logs_worker_' + str(worker_id) + '.csv'
            # PATH1 = MAIN_PATH + '/results/Best_Models/Best_DPG' + '/DPG' + sub + '_' + seed + '/testing/data/logs_worker_' + str(worker_id) + '.csv'
            # PATH1 = MAIN_PATH + '/results/Best_Models/Best_DDPG' + '/DDPG' + sub + '_' + seed + '/testing/data/logs_worker_' + str(worker_id) + '.csv'
            # PATH1 = MAIN_PATH + '/results/Best_Models/Best_TD3' + '/TD3' + sub + '_' + seed + '/testing/data/logs_worker_' + str(worker_id) + '.csv' # Random buffer TD3 only
            PATH1 = MAIN_PATH + '/results/Best_Models/Best_Final_TD3' + '/TD3' + sub + '_' + seed + '/testing/data/logs_worker_' + str(worker_id) + '.csv'


            data = pd.read_csv(PATH1)
            for i in range(12, data.shape[0]):
                d.append([data.iloc[i - 12]['cgm'], data.iloc[i - 11]['cgm'], data.iloc[i - 10]['cgm'],
                          data.iloc[i - 9]['cgm'], data.iloc[i - 8]['cgm'], data.iloc[i - 7]['cgm'],
                          data.iloc[i - 6]['cgm'], data.iloc[i - 5]['cgm'], data.iloc[i - 4]['cgm'],
                          data.iloc[i - 3]['cgm'], data.iloc[i - 2]['cgm'], data.iloc[i - 1]['cgm'],
                          data.iloc[i - 12]['ins'], data.iloc[i - 11]['ins'], data.iloc[i - 10]['ins'],
                          data.iloc[i - 9]['ins'], data.iloc[i - 8]['ins'], data.iloc[i - 7]['ins'],
                          data.iloc[i - 6]['ins'], data.iloc[i - 5]['ins'], data.iloc[i - 4]['ins'],
                          data.iloc[i - 3]['ins'], data.iloc[i - 2]['ins'], data.iloc[i - 1]['ins'],
                          data.iloc[i]['ins']])

            df2 = pd.DataFrame(np.array(d), columns=columns)
            df_combined = pd.concat([df_combined, df2], ignore_index=True)

    return df_combined


# ==== Combine all validations into single table

df_combined = pd.DataFrame()
for sub in subjects:
    df = combine_data(cohort, sub)
    df_combined = pd.concat([df_combined, df], ignore_index=True)

# Convert DataFrame to consistent with IDTxl
df_combined = df_combined[['y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12',
                           'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'i11', 'i12']]
data_array = df_combined.to_numpy().T.reshape((25, 1, -1))

print(df_combined)

# Pass to idxtl data class
data = Data(data_array, dim_order='psr')

# === Run IDTxl Analysis
# Initialise Analaysis
network_analysis = MultivariateMI()
settings = {'cmi_estimator': 'JidtGaussianCMI',
            'max_lag_sources': 0,
            'min_lag_sources': 0}

# Undertake analysis
target_idx = 0
source_idx = [i for i in range(1, 25)]

results = network_analysis.analyse_single_target(settings=settings, data=data, target=target_idx, sources=source_idx)

# Plot results
results.print_edge_list(weights='max_te_lag', fdr=False)
plot_network(results=results, weights='max_te_lag', fdr=False)
plt.show()

# min_rows = min(df.shape[0] for df in dataframes)
# print(min_rows)
# print('%'*100)
#
# #Make all dataframes consistent length
# truncated_arrays = [df.iloc[:min_rows].to_numpy() for df in dataframes]
#
# # Convert dataframes into 3d array
# combined_array = np.stack(truncated_arrays, axis=2)
#
# # Step 4: Reorder the dimensions to (n_variables, n_events, n_timesteps)
# final_array = np.transpose(combined_array, (1, 2, 0))
#
# #Pass to idxtl data class
# data = Data(final_array, dim_order='prs')
#
# #==== IDTXL Analsyis
# # Initialise Analaysis
# network_analysis = BivariateTE()
# settings = {'cmi_estimator': 'JidtGaussianCMI',
#             'max_lag_sources': 12,
#             'min_lag_sources': 1}
#
# # Undertake analysis
# target_idx = 1
# source_idx = [0]
#
# results = network_analysis.analyse_single_target(settings=settings, data=data, target=target_idx, sources=source_idx)
#
# # Plot results
# results.print_edge_list(weights='max_te_lag', fdr=False)
# plot_network(results=results, weights='max_te_lag', fdr=False)
# plt.show()
