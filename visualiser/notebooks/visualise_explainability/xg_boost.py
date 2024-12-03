import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost.plotting as plot
from xgboost import XGBRegressor
import optuna

# ==== Input parameters
# subject = 0
seeds = [1, 2, 3]
algorithms = ['ppo', 'dpg', 'ddpg', 'td3']
subjects = {'ppo': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'dpg': [0, 2, 6],
            'ddpg': [0, 2, 6],
            'td3': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}

file_location = {'ppo': 'data/ppo_combined_allpatients_rev1.csv',
            'dpg': 'data/dpg_combined_allpatients_rev1.csv',
            'ddpg': 'data/ddpg_combined_allpatients_rev1.csv',
            'td3': 'data/td3_combined_allpatients_rev3A.csv'}


df_all = pd.DataFrame()

for alg in algorithms:
    filename = file_location[alg]
    df = pd.read_csv(filename)

    for subject in subjects[alg]:
        print('=' * 100)
        print('Starting ' + alg + ' for Subject ' + str(subject))

        # df = pd.read_csv('data/ppo_combined_allpatients_rev1.csv')
        # df = pd.read_csv('data/dpg_combined_allpatients_rev1.csv')
        # df = pd.read_csv('data/ddpg_combined_allpatients_rev1.csv')
        # filename = 'data/' + alg + '_combined_allpatients_rev1.csv'
        # df = pd.read_csv(filename)

        # ==== Load Data
        df_sub = df[df['subject'] == subject]
        trial_list = sorted(df_sub['trial_id'].unique().tolist())

        for seed in seeds:
            df_filt = df_sub[df_sub['seed'] == seed]

            y = df_filt['y']
            X = df_filt[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9',
                         'x10', 'x11', 'x12', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8',
                         'i9', 'i10', 'i11', 'i12']]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


            # ====Setup parameter study
            # Define the objective function
            def objective(trial):
                # Define the hyperparameter search space
                param = {
                    'verbosity': 0,
                    'objective': 'reg:squarederror',  # Use squared error for regression tasks
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
                }

                # Create and train the model
                model = XGBRegressor(**param)
                model.fit(X_train, y_train)

                # Predict on the test set
                preds = model.predict(X_test)

                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(y_test, preds))

                # Optuna minimizes the objective, so we return the RMSE
                return rmse


            # ====Run Study
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=100, timeout=600)

            # Output the best result
            print("Best trial:")
            trial = study.best_trial

            print("  Value: {}".format(trial.value))
            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))

            # ==== Best performing model
            # Get the best hyperparameters
            best_params = study.best_params
            print(f"Best parameters: {best_params}")

            # Visualize the optimization history
            optuna.visualization.plot_optimization_history(study)

            # Visualize the hyperparameter importance
            optuna.visualization.plot_param_importances(study)

            # ==== Feature Importance
            best_params = study.best_params

            # Train the model on the entire dataset
            optimal_model = XGBRegressor(**best_params)
            optimal_model.fit(X, y)

            importance_gain = optimal_model.get_booster().get_score(importance_type='gain')
            importance_weight = optimal_model.get_booster().get_score(importance_type='weight')
            importance_cover = optimal_model.get_booster().get_score(importance_type='cover')

            # plot.plot_importance(optimal_model, importance_type='gain')

            # ==== Output Results
            # Assuming you have a list of feature names
            feature_names = X.columns  # Adjust according to your dataset

            # Create a DataFrame for feature importance based on gain
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Gain Importance': [importance_gain.get(f, 0) for f in feature_names],
                'Weight Importance': [importance_weight.get(f, 0) for f in feature_names],
                'Cover Importance': [importance_cover.get(f, 0) for f in feature_names],
            })

            # Sort by gain importance
            importance_df = importance_df.sort_values(by='Gain Importance', ascending=False)

            # Display the DataFrame
            # print(importance_df)
            importance_df['alg'] = alg
            importance_df['subject'] = subject

            df_all = pd.concat([df_all, importance_df], ignore_index=True)

df_all.to_excel('feature_importance_allAlgorithms_rev2.xlsx')
