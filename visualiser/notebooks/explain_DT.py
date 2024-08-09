import os
import sys
from decouple import config

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cohorts = ['adult', 'adolescent', 'child']
seeds = ['1', '2', '3']
subjects = ['0', '2', '6']  # , '1', '2', '3', '4', '5', '6', '7', '8', '9']

cohort = 'adolescent'

n_trials = 500  # 500


def get_corr(cohort, sub):
    print("="*100)
    print("Starting for patient {}".format(sub))

    d = []
    columns = ['x12', 'x11', 'x10', 'x9', 'x8', 'x7', 'x6', 'x5', 'x4', 'x3', 'x2', 'x1', 'i12', 'i11', 'i10', 'i9',
               'i8', 'i7', 'i6', 'i5', 'i4', 'i3', 'i2', 'i1', 'y']
    df_combined = pd.DataFrame()

    for trial in range(0, n_trials):
        worker_id = int(trial + 6000)
        for seed in seeds:
            # PATH1=MAIN_PATH + '/results/'+cohort+'/PPO/P'+sub+'_'+seed+'/testing/data/logs_worker_'+str(worker_id)+'.csv'
            PATH1 = MAIN_PATH + '/results/EU59 Experiments/PenaltyTermSensitivity/TD3/NoCutOff/coefficient1e-2' + '/TD3' + sub + '_' + seed + '/testing/data/logs_worker_' + str(worker_id) + '.csv'

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
            df2['cgm_mean'] = (df2['x12'] + df2['x11'] + df2['x10'] + df2['x9'] + df2['x8'] + df2['x7'] + df2['x6'] +
                               df2['x5'] + df2['x4'] + df2['x3'] + df2['x2'] + df2['x1']) / 12
            corr = df2.corr()
            corr = corr.loc[['y']]
            corr['trial_id'] = trial
            corr['seed'] = seed
            corr['subject'] = sub

            df_combined = pd.concat([df_combined, corr], ignore_index=True)
            if trial % 10 == 0 and trial > 0 and seed == '1':
                print('Completed {}/{} trials for patient {}'.format(trial, n_trials, sub))

    print("Finished for patient {}".format(sub))

    return df_combined

def decompose_data(df, data_retain, data_decompose):
    df_decompose = pd.DataFrame(columns=['type', 'value'] + columns_retain)

    for index, row in df.iterrows():
        for column in columns_decompose:
            df_entry = row[columns_retain]
            df_entry['type'] = column
            df_entry['value'] = row[column]

            df_decompose = df_decompose.append(df_entry, ignore_index=True)
    return df_decompose

#==== Combine all validations into single table
df_combined = pd.DataFrame()
for sub in subjects:
    df = get_corr(cohort, sub)
    df_combined = pd.concat([df_combined, df], ignore_index=True)

df_combined.to_csv("td3_combined.csv")

#==== Decompose table for easier plotting
columns_retain = ["trial_id", "seed", "subject"]
columns_drop = ['Unnamed: 0', 'y']
columns_decompose = [header for header in df_combined.columns
                     if header not in columns_retain
                     and header not in columns_drop]

df_decompose = decompose_data(df_combined, columns_retain, columns_decompose)
df_decompose.to_csv("td3_decomposed.csv")

