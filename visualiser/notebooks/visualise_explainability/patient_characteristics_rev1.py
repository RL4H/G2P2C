import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load data
file_path = "data/Result Record - Patient Characteristics.xlsx"
df = pd.read_excel(file_path, sheet_name="Results")

# Filter for PPO and TD3 algorithms
df = df[df['Algorithm'].isin(['PPO', 'TD3'])]

# Define colors for each algorithm
colors = {'PPO': 'blue', 'TD3': 'orange'}

# Plot and fit linear regression
def plot_and_fit(df, x, y='Reward', hue='Failure', style='Algorithm', palette="coolwarm"):
    plt.figure(figsize=(10, 6))
    # Scatter plot with linear fit
    sns.scatterplot(data=df, x=x, y=y, hue=hue, style=style, palette=palette)

    # Linear regression and fitting
    for algo in df[style].unique():
        sub_df = df[df[style] == algo].sort_values(by=x)
        X = sub_df[x].values.reshape(-1, 1)
        Y = sub_df[y].values

        # Fit unweighted regression
        linear_regressor_unweighted = LinearRegression()
        linear_regressor_unweighted.fit(X, Y)
        y_pred_unweighted = linear_regressor_unweighted.predict(X)

        # Plot the linear fit (unweighted) with dashed line
        plt.plot(X, y_pred_unweighted, label=f'{algo} Unweighted Fit', linestyle='--', color=colors[algo])

        # Calculate and display the equation and R-squared for unweighted
        slope_unweighted = linear_regressor_unweighted.coef_[0]
        intercept_unweighted = linear_regressor_unweighted.intercept_
        r2_unweighted = r2_score(Y, y_pred_unweighted)

        print(f'{algo} - {x} (Unweighted): y = {slope_unweighted:.2f}x + {intercept_unweighted:.2f}, R^2 = {r2_unweighted:.2f}')

        # Compute weights for weighted regression
        weights = 1 / np.abs(np.log((sub_df[hue] + 1e-5)))

        # Fit weighted regression
        linear_regressor_weighted = LinearRegression()
        linear_regressor_weighted.fit(X, Y, sample_weight=weights)
        y_pred_weighted = linear_regressor_weighted.predict(X)

        # Plot the linear fit (weighted) with solid line
        plt.plot(X, y_pred_weighted, label=f'{algo} Weighted Fit', linestyle='-', color=colors[algo])

        # Calculate and display the equation and R-squared for weighted
        slope_weighted = linear_regressor_weighted.coef_[0]
        intercept_weighted = linear_regressor_weighted.intercept_
        r2_weighted = r2_score(Y, y_pred_weighted, sample_weight=weights)

        print(f'{algo} - {x} (Weighted): y = {slope_weighted:.2f}x + {intercept_weighted:.2f}, R^2 = {r2_weighted:.2f}')

    plt.title(f'Reward vs {x}')
    plt.legend()
    plt.show()

# Run for each variable
variables = ['CR', 'CF', 'Age', 'Body Weight (Kg)', 'TDI']
for var in variables:
    plot_and_fit(df, x=var)
