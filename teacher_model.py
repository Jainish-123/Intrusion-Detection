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

    def train_with_cross_validation(self, folds=10):
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
        df_results.to_csv(os.path.join(self.results_dir, "cross_validation_results.csv"), index=False)

        print("\n Cross validation results in teacher model training")
        print(df_results)
        print()

        return df_results

    def train_autoencoder(self, normal_train_df, encoding_dim=16, epochs=50, batch_size=32):
        """ Train an autoencoder for anomaly detection. """
        if normal_train_df is None:
            raise ValueError("Normal training data is required for autoencoder.")

        input_dim = normal_train_df.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation="relu")(input_layer)
        decoded = Dense(input_dim, activation="sigmoid")(encoded)

        self.autoencoder = Model(inputs=input_layer, outputs=decoded)
        self.autoencoder.compile(optimizer="adam", loss="mse")

        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

        self._log("Training Autoencoder on Normal Data...")
        self.autoencoder.fit(
            normal_train_df, normal_train_df,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=[early_stopping]
        )
        self._log("Autoencoder training completed!")

    def evaluate_models(self, X_test, y_test):
        """ Evaluate trained models on test dataset and save results """
        results = []
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results.append([name, acc, f1])
            self._log(f"{name} - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

            # Save Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - {name}')
            plt.savefig(os.path.join(self.results_dir, f"confusion_matrix_{name}.png"))
            plt.show()

        # Evaluate Autoencoder
        if self.autoencoder is not None:
            self._log("Evaluating Autoencoder on test data...")
            reconstructed = self.autoencoder.predict(X_test)
            reconstruction_error = np.mean(np.abs(reconstructed - X_test), axis=1)
            threshold = reconstruction_error.mean() + reconstruction_error.std()
            y_pred_autoencoder = (reconstruction_error > threshold).astype(int)
            acc_auto = accuracy_score(y_test, y_pred_autoencoder)
            f1_auto = f1_score(y_test, y_pred_autoencoder, average='weighted')
            results.append(["Autoencoder", acc_auto, f1_auto])
            self._log(f"Autoencoder - Accuracy: {acc_auto:.4f}, F1-Score: {f1_auto:.4f}")

            # Save Confusion Matrix for Autoencoder
            cm_auto = confusion_matrix(y_test, y_pred_autoencoder)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm_auto, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix - Autoencoder')
            plt.savefig(os.path.join(self.results_dir, "confusion_matrix_autoencoder.png"))
            plt.show()

        df_results = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1-Score'])
        df_results.to_csv(os.path.join(self.results_dir, "test_results.csv"), index=False)

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

    def generate_pseudo_labels_semi_supervised(self, unlabeled_df):
        """ Generate pseudo labels using the trained autoencoder. """
        if self.autoencoder is None:
            raise ValueError("Autoencoder model is not trained yet! Train it first.")

        self._log("Generating pseudo labels using Autoencoder...")
        reconstructed = self.autoencoder.predict(unlabeled_df)
        reconstruction_error = np.mean(np.abs(reconstructed - unlabeled_df), axis=1)
        threshold = reconstruction_error.mean() + reconstruction_error.std()

        return (reconstruction_error > threshold).astype(int)
