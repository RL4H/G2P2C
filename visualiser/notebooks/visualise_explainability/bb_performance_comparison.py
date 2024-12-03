import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load data
file_path = "data/Result Record - Patient Characteristics.xlsx"
df = pd.read_excel(file_path, sheet_name="Results")

# Filter for the algorithms you're interested in
df = df[df['Algorithm'].isin(['PPO', 'SAC', 'TD3', 'BBI', 'BBHE'])]

# Create separate dataframes for adolescents and adults
df_adolescent = df[df['Patient'] < 15]
df_adult = df[df['Patient'] >= 15]
df_adult['Patient'] = df_adult['Patient'] - 20

# Define the hue order and custom palette, skipping the color for 'SAC'
hue_order = ['PPO', 'TD3', 'BBI', 'BBHE']
palette = sns.color_palette(['#1f77b4', '#2ca02c', '#d62728', '#9467bd'])  # Blue, Green, Red, Purple

# Plot for adolescents
fig, ax = plt.subplots()
sns.barplot(x='Patient', y='Reward', hue='Algorithm', data=df_adolescent[df_adolescent['Algorithm'] != 'SAC'], hue_order=hue_order, palette=palette, ax=ax)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)  # Place legend outside the graph
plt.tight_layout()
plt.xlabel('Adolescent Patient')
plt.show()

fig, ax = plt.subplots()
sns.barplot(x='Patient', y='TIR', hue='Algorithm', data=df_adolescent[df_adolescent['Algorithm'] != 'SAC'], hue_order=hue_order, palette=palette, ax=ax)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)  # Place legend outside the graph
plt.tight_layout()
plt.xlabel('Adolescent Patient')
plt.show()

fig, ax = plt.subplots()
sns.barplot(x='Patient', y='Failure', hue='Algorithm', data=df_adolescent[df_adolescent['Algorithm'] != 'SAC'], hue_order=hue_order, palette=palette, ax=ax)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)  # Place legend outside the graph
plt.tight_layout()
plt.xlabel('Adolescent Patient')
plt.show()

# Plot for adults
fig, ax = plt.subplots()
sns.barplot(x='Patient', y='Reward', hue='Algorithm', data=df_adult[df_adult['Algorithm'] != 'SAC'], hue_order=hue_order, palette=palette, ax=ax)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)  # Place legend outside the graph
plt.tight_layout()
plt.xlabel('Adult Patient')
plt.show()

fig, ax = plt.subplots()
sns.barplot(x='Patient', y='TIR', hue='Algorithm', data=df_adult[df_adult['Algorithm'] != 'SAC'], hue_order=hue_order, palette=palette, ax=ax)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)  # Place legend outside the graph
plt.tight_layout()
plt.xlabel('Adult Patient')
plt.show()

fig, ax = plt.subplots()
sns.barplot(x='Patient', y='Failure', hue='Algorithm', data=df_adult[df_adult['Algorithm'] != 'SAC'], hue_order=hue_order, palette=palette, ax=ax)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)  # Place legend outside the graph
plt.tight_layout()
plt.xlabel('Adult Patient')
plt.show()