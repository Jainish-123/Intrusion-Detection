import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class TeacherModel:
    def __init__(self, X_train=None, y_train=None, verbose=True):
        """ Initialize Teacher Model """
        self.X_train = X_train
        self.y_train = y_train
        self.verbose = verbose
        self.trained_models = {}
        self.autoencoder = None
        self.results_dir = "results"  # Directory to save results
        os.makedirs(self.results_dir, exist_ok=True)  # Create directory if it doesn't exist

    def _log(self, message):
        """ Print messages only if verbose mode is enabled. """
        if self.verbose:
            print(message)

    def _initialize_models(self):
        """ Initialize and return supervised learning models. """
        return {
            "XGBoost": XGBClassifier(eval_metric='logloss'),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42)
        }

    def train_with_cross_validation(self, dataset, folds=10):
        """ Train supervised models with cross-validation and save results. """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data is missing! Provide X_train and y_train.")

        self.models = self._initialize_models()
        results = []

        for name, model in self.models.items():
            scores = cross_val_score(model, self.X_train, self.y_train, cv=folds, scoring='accuracy')
            avg_score = scores.mean()
            results.append([name, avg_score])
            # self._log(f"{name} - Avg Accuracy ({folds}-fold CV): {avg_score:.4f}")

            # Train the model on full dataset
            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model

        # Save results to CSV
        df_results = pd.DataFrame(results, columns=['Model', 'Cross-Validation Accuracy'])
        df_results.to_csv(os.path.join(self.results_dir, f"{dataset}_cross_validation_results.csv"), index=False)

        print("\n Cross validation results in teacher model training")
        print(df_results)
        print()

        return df_results

    def evaluate_supervised_model(self, X_test, y_test, dataset):
        """Evaluate trained teacher models and autoencoder, save results to CSV"""

        results = []

        # Evaluate supervised teacher models
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results.append([name, acc, f1])
            self._log(f"{name} - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

        # Save to CSV
        df_results = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1-Score'])
        csv_path = os.path.join(self.results_dir, f"{dataset}_teacher_model_comparision.csv")
        df_results.to_csv(csv_path, index=False)

        # Print results
        print("\nTeacher model evaluation on Unseen test data")
        print(df_results)
        print()

        return df_results

    def _get_trained_model(self, model_name):
        """ Retrieve a trained model by name. """
        model = self.trained_models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found! Train it first.")
        return model

    def generate_pseudo_labels(self, unlabeled_df, model_name):
        """ Generate pseudo labels using a trained supervised model. """
        model = self._get_trained_model(model_name)
        return model.predict(unlabeled_df)