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

from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import normalized_mutual_info_score

cohorts = ['adult', 'adolescent', 'child']
seeds = ['1', '2', '3']
subjects = ['6']  # , '1', '2', '3', '4', '5', '6', '7', '8', '9']
# subjects = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


cohort = 'adolescent'

n_trials = 1  # 500


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
            # PATH1=MAIN_PATH + '/results/'+cohort+'/PPO/P'+sub+'_'+seed+'/testing/data/logs_worker_'+str(worker_id)+'.csv'
            # PATH1 = MAIN_PATH + '/results/Best_Models/Best_DPG' + '/DPG' + sub + '_' + seed + '/testing/data/logs_worker_' + str(worker_id) + '.csv'
            # PATH1 = MAIN_PATH + '/results/Best_Models/Best_DDPG' + '/DDPG' + sub + '_' + seed + '/testing/data/logs_worker_' + str(worker_id) + '.csv'
            PATH1 = MAIN_PATH + '/results/Best_Models/Best_TD3' + '/TD3' + sub + '_' + seed + '/testing/data/logs_worker_' + str(
                worker_id) + '.csv'

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

# Convert DataFrame into
X = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12']]
y = df['y']

# Compute the mutual information scores
mi_scores = mutual_info_regression(X, y, random_state=0)
# mi_scores = normalized_mutual_info_score(X, y)

# Create a DataFrame to hold the results
mi_df = pd.DataFrame({
    'Variable': X.columns,
    'Mutual Information': mi_scores
})

# Sort the DataFrame by the mutual information scores in descending order
mi_df = mi_df.sort_values(by='Mutual Information', ascending=False).reset_index(drop=True)

# Printout of the mutual information scores
print("Mutual Information Scores:")
print(mi_df)

# Ranked list
print("\nRanked List of Variables by Mutual Information:")
for i, row in mi_df.iterrows():
    print(f"Rank {i + 1}: {row['Variable']} - MI Score: {row['Mutual Information']}")

# Bar plot
plt.figure(figsize=(10, 6))
plt.bar(mi_df['Variable'], mi_df['Mutual Information'], color='blue')
plt.xlabel('Variables')
plt.ylabel('Mutual Information Score')
plt.title('Mutual Information Scores for Each Variable')
plt.xticks(rotation=45)
plt.show()
