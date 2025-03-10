# DataHandler Class for Dataset Splitting (Supervised & Semi-Supervised)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataHandler:
    def __init__(self, file_path, label_column="Label", verbose=True):
        self.file_path = file_path
        self.label_column = label_column
        self.df = None
        self.verbose = verbose  # Flag to control debug output

    def _log(self, message):
        """ Print messages only if verbose is enabled. """
        if self.verbose:
            print(message)

    def load_dataset(self):
        """ Load dataset from file. """
        self.df = pd.read_csv(self.file_path)
        self._log(f"Dataset loaded successfully. Shape: {self.df.shape}")

    def split_labeled_unlabeled(self, labeled_fraction=0.05, random_state=42):
        """ Split dataset into labeled and unlabeled datasets. """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        self.df = self.df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        self.labeled_df = self.df.sample(frac=labeled_fraction, random_state=random_state)
        self.unlabeled_df = self.df.drop(self.labeled_df.index)
        self._log(f"Labeled: {self.labeled_df.shape}, Unlabeled: {self.unlabeled_df.shape}")

    def split_unlabeled_train_test(self, test_size=0.10, random_state=42):
        """ Split unlabeled data into train and test. """
        X_unlabeled = self.unlabeled_df.drop(columns=[self.label_column])
        y_unlabeled = self.unlabeled_df[self.label_column]

        self.unlabeled_train_df, self.unlabeled_test_df, self.train_labels, self.test_labels = train_test_split(
            X_unlabeled, y_unlabeled, test_size=test_size, random_state=random_state
        )
        self._log(f"Unlabeled Train: {self.unlabeled_train_df.shape}, Test: {self.unlabeled_test_df.shape}")

    def split_normal_attack_data(self):
        """ Split normal and attack data. """
        self.normal_df = self.df.loc[self.df[self.label_column] == 0]
        self.attack_df = self.df.loc[self.df[self.label_column] == 1]
        self._log(f"Normal: {self.normal_df.shape}, Attack: {self.attack_df.shape}")

    def split_autoencoder_train_data(self, normal_train_fraction=0.20, random_state=42):
        """ Split normal data into train and pseudo-labeling. """
        self.normal_train_df = self.normal_df.sample(frac=normal_train_fraction, random_state=random_state).drop(columns=[self.label_column])
        self.normal_pseudo_label_df = self.normal_df.drop(self.normal_train_df.index)
        self._log(f"Autoencoder Train: {self.normal_train_df.shape}, Pseudo: {self.normal_pseudo_label_df.shape}")

    def prepare_semi_supervised_data(self, test_size=0.10, random_state=42):
        """ Prepare data for semi-supervised learning. """
        combined_df = pd.concat([self.normal_pseudo_label_df, self.attack_df], ignore_index=True)
        y_combined = combined_df[self.label_column]
        X_combined = combined_df.drop(columns=[self.label_column])[self.normal_train_df.columns]

        self.semi_supervised_train_df, self.semi_supervised_test_df, self.semi_supervised_train_labels, self.semi_supervised_test_labels = train_test_split(
            X_combined, y_combined, test_size=test_size, random_state=random_state
        )
        self._log(f"Semi-Supervised Train: {self.semi_supervised_train_df.shape}, Test: {self.semi_supervised_test_df.shape}")

    def save_datasets(self, prefix="supervised"):
        """ Save datasets to CSV. """
        datasets = {
            "labeled": self.labeled_df,
            "unlabeled_train": self.unlabeled_train_df,
            "unlabeled_test": self.unlabeled_test_df,
            "train_labels": self.train_labels,
            "test_labels": self.test_labels,
        }
        for name, data in datasets.items():
            if data is not None:
                data.to_csv(f"{prefix}_{name}.csv", index=False)
        self._log(f"{prefix} datasets saved successfully!")

    def save_datasets_semi_supervised(self, prefix="semi_supervised"):
        """ Save semi-supervised datasets. """
        datasets = {
            "normal_train": self.normal_train_df,
            "normal_pseudo": self.normal_pseudo_label_df,
            "attack": self.attack_df,
            "train": self.semi_supervised_train_df,
            "test": self.semi_supervised_test_df,
            "test_labels": self.semi_supervised_test_labels,
        }
        for name, data in datasets.items():
            if data is not None:
                data.to_csv(f"{prefix}_{name}.csv", index=False)
        self._log(f"{prefix} datasets saved successfully!")

    def prepare_labeled_data_for_training(self, validation_split=0.20, random_state=42):
        """ Split labeled data into training and validation sets. """
        X = self.labeled_df.drop(columns=[self.label_column])
        y = self.labeled_df[self.label_column]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=random_state)
        self._log(f"Labeled Train: {X_train.shape}, Validation: {X_val.shape}")
        return X_train, X_val, y_train, y_val
