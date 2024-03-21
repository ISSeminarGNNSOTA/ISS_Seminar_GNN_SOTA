import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge

class LRModel:

    def __init__(self, ratings):
        self.ratings = ratings
        self.X, self.y = self.extract_features(ratings)
        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=888)
        self.model = None  # Initialize the model attribute

  
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

    
    def tune_hyperparameters(self, n_trials=10):
        def objective(trial):
            # Hyperparameters search space
            alpha = trial.suggest_loguniform('alpha', 0.0001, 10)
            model = Ridge(alpha=alpha)

            # Manual 2-fold cross-validation
            kf = KFold(n_splits=2, shuffle=True, random_state=888)
            rmse_scores = []

            for train_index, val_index in kf.split(self.X_train_val):
                X_train, X_val = self.X_train_val[train_index], self.X_train_val[val_index]
                y_train, y_val = self.y_train_val[train_index], self.y_train_val[val_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                rmse_scores.append(rmse)

            return np.mean(rmse_scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        self.best_alpha = study.best_params['alpha']
        print("Best alpha:", self.best_alpha)
        self.model = Ridge(alpha=self.best_alpha)  # Use the best alpha found

   
    def train_and_evaluate(self):
        if self.model is None:
            raise Exception("Hyperparameters not tuned yet. Please run tune_hyperparameters() first.")
        # Since self.model is already set to the best Ridge model, you can train it directly
        self.model.fit(self.X_train_val, self.y_train_val)
        y_pred_test = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred_test)
        print(f"Final Evaluation on Test Set - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    def print_actual_vs_predicted(self, dataset='test'):
        if self.model is None:
            raise Exception("Model not trained yet. Please run train_and_evaluate() first.")
 
        if dataset == 'train':
            X, y = self.X_train_val, self.y_train_val
            data_label = "Training"
        elif dataset == 'test':
            X, y = self.X_test, self.y_test
            data_label = "Test"
        else:
            raise ValueError("Invalid dataset specified. Choose 'train' or 'test'.")

        # Making predictions
        y_pred = self.model.predict(X)

        print(f"{data_label} Data - Actual vs. Predicted Ratings:")
        for actual, predicted in zip(y[:10], y_pred[:10]):  # Adjust the slice as needed
            print(f"Actual: {actual}, Predicted: {predicted:.4f}")
