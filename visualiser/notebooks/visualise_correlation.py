import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def decompose_data(df, data_retain, data_decompose):
    df_decompose = pd.DataFrame(columns=['type', 'value'] + columns_retain)

    for index, row in df.iterrows():
        for column in columns_decompose:
            df_entry = row[columns_retain]
            df_entry['type'] = column
            df_entry['value'] = row[column]

            df_decompose = df_decompose.append(df_entry, ignore_index=True)
    return df_decompose


# ===Plot comparing PPO and TD3
df_ppo = pd.read_csv('ppo_decomposed.csv')
df_ppo['algorithm'] = 'PPO'

df_td3 = pd.read_csv('td3_decomposed.csv')
df_td3['algorithm'] = 'TD3'

df = pd.concat([df_ppo, df_td3], ignore_index=True)
for patient in [0, 2, 6]:
    sns.boxplot(
        data=df[df['subject'] == patient], x="value", y="type", hue='algorithm',
        fill=False, palette="deep", showfliers=False)
    plt.title(f'Patient {patient}')
    plt.xlabel('Correlation')
    plt.ylabel('variable')
    plt.show()

for algo in ['PPO', 'TD3']:
    sns.boxplot(
        data=df[df['algorithm'] == algo], x="value", y="type", hue='subject',
        fill=False, palette="deep", showfliers=False)
    plt.title(f'{algo}')
    plt.xlabel('Correlation')
    plt.ylabel('variable')
    plt.show()
