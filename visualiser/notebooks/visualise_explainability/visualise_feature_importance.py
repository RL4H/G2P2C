import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# algorithms = ['ppo', 'td3', 'dpg', 'ddpg']
algorithms = ['ppo', 'td3']

# patients = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
patients = [0,2,6]

# ===Plot comparing PPO and TD3
# df = pd.read_excel('feature_importance.xlsx')
df = pd.read_excel('feature_importance_allAlgorithms_rev2.xlsx')
df['alg'] = pd.Categorical(df['alg'], categories=algorithms, ordered=True)
df = df.sort_values(by=['alg'], ascending=False)



df = df[df['subject'].isin(patients)]

df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.sort_values(by=['subject', 'Feature'], ascending=True, inplace=True)
df_ppo = df[df['alg'] == 'ppo']
df_dpg = df[df['alg'] == 'dpg']
df_ddpg = df[df['alg'] == 'ddpg']
df_td3 = df[df['alg'] == 'td3']


new_labels = ["t-" + str(i) for i in range(1,13)]
insulin_labels = ["i" + str(i) for i in range(1, 13)]
glucose_labels = ["x" + str(i) for i in range(1, 13)]

df_insulin = df[df['Feature'].isin(insulin_labels)]
df_glucose = df[df['Feature'].isin(glucose_labels)]

# Plot glucose
for patient in patients:
    sns.boxplot(
        data=df_glucose[df_glucose['subject'] == patient], x="Feature", y="Gain Importance", hue='alg',
        fill=False, palette="deep", showfliers=False, order=glucose_labels)
    plt.title(f'Patient {patient}')
    plt.xlabel('Glucose')
    plt.ylabel('Feature Importance')
    # plt.yscale('log')
    # plt.ylim(0.01, 100)
    plt.xticks(ticks=plt.gca().get_xticks(), labels=new_labels)  # Change x-axis labels
    # plt.gca().invert_xaxis()  # Reverse the x-axis

    plt.show()

for algo in algorithms:
    sns.boxplot(
        data=df_glucose[df_glucose['alg'] == algo], x="Feature", y="Gain Importance", hue='subject',
        fill=False, palette="deep", showfliers=False, order=glucose_labels)
    plt.title(f'{algo}')
    plt.xlabel('Glucose')
    plt.ylabel('Feature Importance')
    # plt.yscale('log')
    # plt.ylim(0.01, 100)
    plt.xticks(ticks=plt.gca().get_xticks(), labels=new_labels)  # Change x-axis labels
    # plt.gca().invert_xaxis()  # Reverse the x-axis

    plt.show()

# Plot Insulin
for patient in patients:
    sns.boxplot(
        data=df_insulin[df_insulin['subject'] == patient], x="Feature", y="Gain Importance", hue='alg',
        fill=False, palette="deep", showfliers=False, order=insulin_labels)
    plt.title(f'Patient {patient}')
    plt.xlabel('Insulin')
    plt.ylabel('Feature Importance')
    # plt.yscale('log')
    # plt.ylim(0.01, 100)
    plt.xticks(ticks=plt.gca().get_xticks(), labels=new_labels)  # Change x-axis labels
    # plt.gca().invert_xaxis()  # Reverse the x-axis

    plt.show()

for algo in algorithms:
    sns.boxplot(
        data=df_insulin[df_insulin['alg'] == algo], x="Feature", y="Gain Importance", hue='subject',
        fill=False, palette="deep", showfliers=False, order=insulin_labels)
    plt.title(f'{algo}')
    plt.xlabel('Insulin')
    plt.ylabel('Feature Importance')
    # plt.yscale('log')
    # plt.ylim(0.01, 100)
    plt.xticks(ticks=plt.gca().get_xticks(), labels=new_labels)  # Change x-axis labels
    # plt.gca().invert_xaxis()  # Reverse the x-axis

    plt.show()
