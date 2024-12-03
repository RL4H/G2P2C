import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('td3_combined_allpatients_rev2.csv')

# Drop the unnamed index column and columns 'y' and 'cgm_mean'
df = df.drop(columns=['y', 'cgm_mean'])
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove any unnamed columns that might be index columns

# Reshape the DataFrame
# Columns to keep as-is
columns_to_keep = ['trial_id', 'seed', 'subject']

# Columns to melt (X12 to X1 and I12 to I1)
columns_to_melt = [col for col in df.columns if col not in columns_to_keep]

# Melt the DataFrame
melted_df = pd.melt(df, id_vars=columns_to_keep, value_vars=columns_to_melt,
                    var_name='type', value_name='value')

# Save the reshaped DataFrame to a new CSV file
melted_df.to_csv('reshaped_data.csv', index=False)
