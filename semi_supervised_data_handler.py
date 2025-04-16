# Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, accuracy_score,
    f1_score, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class DataHandler:
    def __init__(self, file_path, test_size=0.3, random_state=42):
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.categorical_cols = ['proto', 'service', 'conn_state', 'http_method', 'http_version']

    def load_and_preprocess_network(self):
        """Load and preprocess the data"""
        data = pd.read_csv(self.file_path)
        data = pd.get_dummies(data, columns=self.categorical_cols)

        # Printing dataset information
        print(f"Total records: {len(data)}")
        normal_data = len(data[data['Label'] == 0])
        attack_data = len(data[data['Label'] == 1])
        print(f"Normal data: {normal_data}")
        print(f"Attack data: {attack_data}")

        return data.drop(columns=['Label']), data['Label']

    def load_and_preprocess_weather(self):
        """Load original features without interactions"""
        data = pd.read_csv(self.file_path)

        # Printing dataset information
        print(f"Total records: {len(data)}")
        normal_data = len(data[data['Label'] == 0])
        attack_data = len(data[data['Label'] == 1])
        print(f"Normal data: {normal_data}")
        print(f"Attack data: {attack_data}")

        X = data[['temperature', 'pressure', 'humidity']]  # Original features only
        y = data['Label']
        return X, y

    def split_data(self, X, y):
        """Split data into normal and attack sets"""
        X_normal = X[y == 0]
        X_attack = X[y == 1]
        return X_normal, X_attack

    def prepare_datasets_network(self):
        """Prepare train and test datasets"""
        X, y = self.load_and_preprocess_network()
        X_normal, X_attack = self.split_data(X, y)

        # Train-test split
        X_train, X_val_normal = train_test_split(
            X_normal, test_size=self.test_size, random_state=self.random_state
        )

        # Create test set (normal + attacks)
        X_test = pd.concat([X_val_normal, X_attack])
        y_test = pd.concat([
            pd.Series(0, index=X_val_normal.index),
            pd.Series(1, index=X_attack.index)
        ])

        # Scale and apply PCA
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)

        return X_train_pca, X_test_pca, y_test

    def prepare_datasets_weather(self):
        """Proper 80/20 split with validation attacks"""
        X, y = self.load_and_preprocess_weather()
        X_normal, X_attack = self.split_data(X, y)

        # Split normal data: 80% train, 20% validation
        X_train, X_val_normal = train_test_split(
            X_normal,
            test_size=self.test_size,  # 20% for validation
            random_state=self.random_state
        )

        # Create test set
        X_test = pd.concat([X_val_normal, X_attack])
        y_test = pd.concat([
            pd.Series(0, index=X_val_normal.index),
            pd.Series(1, index=X_attack.index)
        ])

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        # X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # return X_train_scaled, X_val_scaled, X_test_scaled, y_val, y_test
        return X_train_scaled, X_test_scaled, y_test