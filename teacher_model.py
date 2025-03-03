# TeacherModel Class (Supports Supervised & Semi-Supervised Learning)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

class TeacherModel:
    def __init__(self, X_train=None, y_train=None):
        """ Initialize Teacher Model """
        self.models = self._initialize_models()
        self.trained_models = {}

        # Initialize the autoencoder only if X_train and y_train are not provided
        if X_train is None or y_train is None:
            self.autoencoder = None
        else:
            self.X_train = X_train
            self.y_train = y_train

    def _initialize_models(self):
        """ Initialize Supervised Learning Models """
        return {
            "XGBoost": XGBClassifier(eval_metric='logloss'),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42)
        }

    def train_with_cross_validation(self, folds=10):
        """ Train Supervised Learning Models using Cross-Validation """
        best_scores = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, self.X_train, self.y_train, cv=folds, scoring='accuracy')
            avg_score = scores.mean()
            best_scores[name] = avg_score
            print(f"{name} - Avg Accuracy (10-fold CV): {avg_score:.4f}")

            # Train on full dataset
            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model
        return best_scores

    def train_autoencoder(self, normal_train_df, encoding_dim=16, epochs=50, batch_size=32):
        """ Train an Autoencoder on Normal Events for Semi-Supervised Learning """

        # Get feature count
        input_dim = normal_train_df.shape[1]

        # Define the Autoencoder architecture
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation="relu")(input_layer)
        decoded = Dense(input_dim, activation="sigmoid")(encoded)

        self.autoencoder = Model(inputs=input_layer, outputs=decoded)
        self.autoencoder.compile(optimizer="adam", loss="mse")

        # Train the Autoencoder
        print("Training Autoencoder on Normal Data...")
        self.autoencoder.fit(normal_train_df, normal_train_df, epochs=epochs, batch_size=batch_size, shuffle=True)

        print("Autoencoder training completed!")

    def generate_pseudo_labels(self, unlabeled_df, model_name):
        model = self.trained_models.get(model_name)
        if model:
            preds = model.predict(unlabeled_df)
            return preds
        else:
            print(f"Model {model_name} not found!")
            return None

    def generate_pseudo_labels_semi_supervised(self, unlabeled_df):
        if self.autoencoder:
                print("Generating pseudo labels using Autoencoder...")

                # Compute reconstruction error
                reconstructed = self.autoencoder.predict(unlabeled_df)
                reconstruction_error = np.mean(np.abs(reconstructed - unlabeled_df), axis=1)

                # Define threshold (Use mean + std deviation heuristic)
                threshold = np.mean(reconstruction_error) + np.std(reconstruction_error)
                pseudo_labels = np.where(reconstruction_error > threshold, 1, 0)

                return pseudo_labels
        else:
                print("Autoencoder model is not trained yet!")
                return None

