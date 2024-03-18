# -*- coding: utf-8 -*-
"""LR-Ridge

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1R1BCEew-dUDVJKj0gMjHEdyH9Xqfiny9
"""

import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge

class LRModel:
    def __init__(self, alpha=2.0):
        self.model = Ridge(alpha=alpha)

    def extract_features(self, ratings):
        X = np.column_stack((ratings['rating_count_per_user'], ratings['rating_count_per_movie'],
                             ratings['avg_rating_per_person'], ratings['avg_rating_per_movie'], ratings['ReleaseAge']))
        y = np.array(ratings['Rating'])
        return X, y

    def extract_features(self, ratings):
        X = np.column_stack((
        ratings['rating_count_per_user'],
        ratings['rating_count_per_movie'],
        ratings['avg_rating_per_person'],
        ratings['avg_rating_per_movie'],
        ratings['ReleaseAge'],
        # Add cluster features
        ratings['Cluster_0'], ratings['Cluster_1'], 
        ratings['Cluster_2'], ratings['Cluster_3'], 
        ratings['Cluster_4'],
        # Add user embedding features
        ratings['user_embedding_0'], ratings['user_embedding_1'],
        ratings['user_embedding_2'], ratings['user_embedding_3'],
        ratings['user_embedding_4'], ratings['user_embedding_5'],
        ratings['user_embedding_6'], ratings['user_embedding_7'],
        ratings['user_embedding_8'], ratings['user_embedding_9'],
        ratings['user_embedding_10'], ratings['user_embedding_11'],
        ratings['user_embedding_12'], ratings['user_embedding_13'],
        ratings['user_embedding_14'], ratings['user_embedding_15'],
        ratings['user_embedding_16'], ratings['user_embedding_17'],
        ratings['user_embedding_18'], ratings['user_embedding_19'],
        # Add movie embedding features
        ratings['movie_embedding_0'], ratings['movie_embedding_1'],
        ratings['movie_embedding_2'], ratings['movie_embedding_3'],
        ratings['movie_embedding_4'], ratings['movie_embedding_5'],
        ratings['movie_embedding_6'], ratings['movie_embedding_7'],
        ratings['movie_embedding_8'], ratings['movie_embedding_9'],
        ratings['movie_embedding_10'], ratings['movie_embedding_11'],
        ratings['movie_embedding_12'], ratings['movie_embedding_13'],
        ratings['movie_embedding_14'], ratings['movie_embedding_15'],
        ratings['movie_embedding_16'], ratings['movie_embedding_17'],
        ratings['movie_embedding_18'], ratings['movie_embedding_19']
        ))
    y = np.array(ratings['Rating'])
    return X, y

    

    def cross_validate(self, ratings, num_folds=5):
        X, y = self.extract_features(ratings)
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=888)

        rmse_scores, mae_scores, mse_scores = [], [], []
        start_time = time.time()  # Start timing for cross-validation

        for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

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
        return np.mean(rmse_scores), np.mean(mae_scores)

    def train_and_evaluate(self, ratings):
        X, y = self.extract_features(ratings)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=888)

        train_start_time = time.time()  # Start timing for training and evaluation
        self.model.fit(X_train, y_train)

        y_pred_test = self.model.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = mse_test ** 0.5
        mae_test = mean_absolute_error(y_test, y_pred_test)
        train_end_time = time.time()  # End timing for training and evaluation

        #print(f"Test MSE: {mse_test:.4f}, Test RMSE: {rmse_test:.4f}, Test MAE: {mae_test:.4f}")
        #print(f"Training and evaluation time: {train_end_time - train_start_time:.2f} seconds")

        return {"test_mse": mse_test, "test_rmse": rmse_test, "test_mae": mae_test}

    def predict(self, features):
        return self.model.predict(features)
