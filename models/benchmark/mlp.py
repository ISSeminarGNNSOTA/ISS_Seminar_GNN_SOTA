# -*- coding: utf-8 -*-
"""mlp

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1R1BCEew-dUDVJKj0gMjHEdyH9Xqfiny9
"""

import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
class MLPModelOptimized:
    def __init__(self, ratings):
        self.ratings = ratings
        self.X, self.y = self.prepare_data()
        self.X = self.scale_data(self.X)

    def prepare_data(self):
        X = np.column_stack((
        self.ratings['rating_count_per_user'],
        self.ratings['rating_count_per_movie'],
        self.ratings['avg_rating_per_person'],
        self.ratings['avg_rating_per_movie'],
        self.ratings['ReleaseAge'],
        # Add cluster features
        self.ratings['Cluster_0'], self.ratings['Cluster_1'], 
        self.ratings['Cluster_2'], self.ratings['Cluster_3'], 
        self.ratings['Cluster_4'],
        # Add user embedding features
          
        self.ratings['user_embedding_0'], self.ratings['user_embedding_1'],
        self.ratings['user_embedding_2'], self.ratings['user_embedding_3'],
        self.ratings['user_embedding_4'], self.ratings['user_embedding_5'],
        self.ratings['user_embedding_6'], self.ratings['user_embedding_7'],
        self.ratings['user_embedding_8'], self.ratings['user_embedding_9'],
        self.ratings['user_embedding_10'], self.ratings['user_embedding_11'],
        self.ratings['user_embedding_12'], self.ratings['user_embedding_13'],
        self.ratings['user_embedding_14'], self.ratings['user_embedding_15'],
        self.ratings['user_embedding_16'], self.ratings['user_embedding_17'],
        self.ratings['user_embedding_18'], self.ratings['user_embedding_19'],
        # Add movie embedding features
        self.ratings['movie_embedding_0'], self.ratings['movie_embedding_1'],
        self.ratings['movie_embedding_2'], self.ratings['movie_embedding_3'],
        self.ratings['movie_embedding_4'], self.ratings['movie_embedding_5'],
        self.ratings['movie_embedding_6'], self.ratings['movie_embedding_7'],
        self.ratings['movie_embedding_8'], self.ratings['movie_embedding_9'],
        self.ratings['movie_embedding_10'], self.ratings['movie_embedding_11'],
        self.ratings['movie_embedding_12'], self.ratings['movie_embedding_13'],
        self.ratings['movie_embedding_14'], self.ratings['movie_embedding_15'],
        self.ratings['movie_embedding_16'], self.ratings['movie_embedding_17'],
        self.ratings['movie_embedding_18'], self.ratings['movie_embedding_19']
        ))
        y = np.array(self.ratings['Rating'])
        return X, y


    def scale_data(self, X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def objective(self, trial):
        hidden_layer_sizes_option = trial.suggest_categorical('hidden_layer_sizes_option', ['50', '100', '50_50', '100_100'])
        hidden_layer_sizes = self.format_hidden_layers(hidden_layer_sizes_option)

        activation = trial.suggest_categorical('activation', ['tanh', 'relu'])
        solver = trial.suggest_categorical('solver', ['sgd', 'adam'])
        alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-5, 1e-1, log=True)

        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
                             alpha=alpha, learning_rate_init=learning_rate_init, max_iter=1000)

        # Split the data into training and validation sets for tuning
        X_train, X_valid, y_train, y_valid = train_test_split(self.X, self.y, test_size=0.2, random_state=888)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        rmse = mean_squared_error(y_valid, y_pred, squared=False)

        return rmse

    def format_hidden_layers(self, option):
        if '_' in option:
            return tuple(map(int, option.split('_')))
        return (int(option),)

    def tune_hyperparameters(self, n_trials=10):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params

    def evaluate_with_cross_validation(self, best_params):
        start_time = time.time()  # Start timing for cross-validation
        hidden_layer_sizes = self.format_hidden_layers(best_params['hidden_layer_sizes_option'])
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                             activation=best_params['activation'],
                             solver=best_params['solver'],
                             alpha=best_params['alpha'],
                             learning_rate_init=best_params['learning_rate_init'],
                             max_iter=1000)

        kf = KFold(n_splits=5, shuffle=True, random_state=888)
        rmse_scores, mae_scores, mse_scores = [], [], []

        for fold, (train_index, test_index) in enumerate(kf.split(self.X), start=1):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5
            mae = mean_absolute_error(y_test, y_pred)

            mse_scores.append(mse)
            rmse_scores.append(rmse)
            mae_scores.append(mae)

            print(f"Fold {fold}: MSE = {mse:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")

        end_time = time.time()  # End timing for cross-validation
        total_time = end_time - start_time

        print(f"Average MSE: {np.mean(mse_scores):.4f}, Average RMSE: {np.mean(rmse_scores):.4f}, Average MAE: {np.mean(mae_scores):.4f}")
        print(f"Total time spent: {total_time:.2f} seconds")
