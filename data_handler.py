# DataHandler Class for Dataset Splitting and Saving

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataHandler:
    def __init__(self, file_path, label_column="Label"):
        self.file_path = file_path
        self.label_column = label_column
        self.df = None
        self.labeled_df = None
        self.unlabeled_df = None
        self.unlabeled_train_df = None
        self.unlabeled_test_df = None
        self.train_labels = None
        self.test_labels = None

    def load_dataset(self):
        self.df = pd.read_csv(self.file_path)
        print(f"Dataset loaded successfully. Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")

    def split_labeled_unlabeled(self, labeled_fraction=0.05):
        np.random.seed(42)
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.labeled_df = self.df.sample(frac=labeled_fraction, random_state=42)
        self.unlabeled_df = self.df.drop(self.labeled_df.index)
        print(f"Labeled Data Shape: {self.labeled_df.shape}")
        print(f"Unlabeled Data Shape: {self.unlabeled_df.shape}")

    def split_unlabeled_train_test(self, test_size=0.10):
        original_labels = self.unlabeled_df[self.label_column]
        unlabeled_df_no_label = self.unlabeled_df.drop(columns=[self.label_column])

        self.unlabeled_train_df, self.unlabeled_test_df, self.train_labels, self.test_labels = train_test_split(
            unlabeled_df_no_label, original_labels, test_size=test_size, random_state=42
        )

        print(f"Unlabeled Train Shape: {self.unlabeled_train_df.shape}")
        print(f"Unlabeled Test Shape: {self.unlabeled_test_df.shape}")
        return self.unlabeled_train_df, self.unlabeled_test_df, self.train_labels, self.test_labels

    def save_datasets(self):
        self.labeled_df.to_csv("labeled_data.csv", index=False)
        self.unlabeled_train_df.to_csv("unlabeled_train_data.csv", index=False)
        self.unlabeled_test_df.to_csv("unlabeled_test_data.csv", index=False)
        self.train_labels.to_csv("unlabeled_train_true_labels.csv", index=False)
        self.test_labels.to_csv("unlabeled_test_true_labels.csv", index=False)
        print("Datasets saved successfully!")

    def prepare_labeled_data_for_training(self, validation_split=0.20):
        X = self.labeled_df.drop(columns=[self.label_column])
        y = self.labeled_df[self.label_column]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        print(f"Training Data Shape: {X_train.shape}")
        print(f"Validation Data Shape: {X_val.shape}")
        return X_train, X_val, y_train, y_val