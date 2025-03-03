# DataHandler Class for Dataset Splitting (Supervised & Semi-Supervised)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataHandler:
    def __init__(self, file_path, label_column="Label"):
        self.file_path = file_path
        self.label_column = label_column
        self.df = None

        # Supervised Learning Datasets
        self.labeled_df = None
        self.unlabeled_df = None
        self.unlabeled_train_df = None
        self.unlabeled_test_df = None
        self.train_labels = None
        self.test_labels = None

        # Semi-Supervised Learning Datasets
        self.normal_df = None
        self.attack_df = None
        self.normal_train_df = None
        self.normal_pseudo_label_df = None
        self.semi_supervised_train_df = None
        self.semi_supervised_test_df = None
        self.semi_supervised_test_labels = None

    def load_dataset(self):
        """ Load the dataset from the provided file path. """
        self.df = pd.read_csv(self.file_path)
        print(f"Dataset loaded successfully. Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")

    # Supervised Learning Splits 
    def split_labeled_unlabeled(self, labeled_fraction=0.05):
        """ Split the dataset into 5% labeled and 95% unlabeled datasets. """
        np.random.seed(42)
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.labeled_df = self.df.sample(frac=labeled_fraction, random_state=42)
        self.unlabeled_df = self.df.drop(self.labeled_df.index)
        print(f"Labeled Data Shape: {self.labeled_df.shape}")
        print(f"Unlabeled Data Shape: {self.unlabeled_df.shape}")

    def split_unlabeled_train_test(self, test_size=0.10):
        """ Further split the unlabeled data into 90% train and 10% test. """
        # Store labels before dropping
        self.train_labels = self.unlabeled_df[self.label_column].copy()
        self.test_labels = self.unlabeled_df[self.label_column].copy()

        # Drop labels from features
        unlabeled_df_no_label = self.unlabeled_df.drop(columns=[self.label_column])

        self.unlabeled_train_df, self.unlabeled_test_df, _, _ = train_test_split(
            unlabeled_df_no_label, self.train_labels, test_size=test_size, random_state=42
        )

        print(f"Unlabeled Train Shape: {self.unlabeled_train_df.shape}")
        print(f"Unlabeled Test Shape: {self.unlabeled_test_df.shape}")

        return self.unlabeled_train_df, self.unlabeled_test_df, self.train_labels, self.test_labels

    # Semi-Supervised Learning Splits (Autoencoder-Based) 
    def split_normal_attack_data(self):
        """ Separate normal and attack data for Autoencoder training. """
        self.normal_df = self.df[self.df[self.label_column] == 0].copy()  # Normal samples
        self.attack_df = self.df[self.df[self.label_column] == 1].copy()  # Attack samples

        print(f"Normal Data Shape: {self.normal_df.shape}")
        print(f"Attack Data Shape: {self.attack_df.shape}")

    def split_autoencoder_train_data(self, normal_train_fraction=0.20):
        """ Split normal data into 20% for Autoencoder training & 80% for pseudo-labeling. """
        self.normal_train_df = self.normal_df.sample(frac=normal_train_fraction, random_state=42).drop(columns=[self.label_column])
        self.normal_pseudo_label_df = self.normal_df.drop(self.normal_train_df.index)

        print(f"Normal Training Data for Autoencoder Shape: {self.normal_train_df.shape}")
        print(f"Remaining Normal Data for Pseudo Labeling Shape: {self.normal_pseudo_label_df.shape}")

    def prepare_semi_supervised_data(self, test_size=0.10):
      """ Combine pseudo-label normal data + attack data → split into 90% Train / 10% Test. """

      # Combine Normal & Attack Data
      combined_df = pd.concat([self.normal_pseudo_label_df, self.attack_df], ignore_index=True)

      # Save the labels before dropping them
      original_labels = combined_df[self.label_column].copy()

      # Drop labels after saving them
      combined_df_no_label = combined_df.drop(columns=[self.label_column])

      # Ensure the dataset has the same columns as Autoencoder training data
      combined_df_no_label = combined_df_no_label[self.normal_train_df.columns]

      # Perform Train-Test Split & Assign Correct Labels
      self.semi_supervised_train_df, self.semi_supervised_test_df, self.semi_supervised_train_labels, self.semi_supervised_test_labels = train_test_split(
          combined_df_no_label, original_labels, test_size=test_size, random_state=42
      )

      print(f"Semi-Supervised Training Data Shape: {self.semi_supervised_train_df.shape}")
      print(f"Semi-Supervised Testing Data Shape: {self.semi_supervised_test_df.shape}")
      print(f"Semi-Supervised Training Labels Shape: {self.semi_supervised_train_labels.shape}")
      print(f"Semi-Supervised Testing Labels Shape: {self.semi_supervised_test_labels.shape}")



    def save_datasets(self):
        """ Save all datasets for both Supervised & Semi-Supervised Learning. """
        # Save Supervised Learning Data
        self.labeled_df.to_csv("labeled_data.csv", index=False)
        self.unlabeled_train_df.to_csv("unlabeled_train_data.csv", index=False)
        self.unlabeled_test_df.to_csv("unlabeled_test_data.csv", index=False)
        self.train_labels.to_csv("unlabeled_train_true_labels.csv", index=False)
        self.test_labels.to_csv("unlabeled_test_true_labels.csv", index=False)

        print("All datasets saved successfully!")

    def save_datasets_semi_supervised(self):
        """ Save Semi-Supervised Learning Data """
        self.normal_train_df.to_csv("normal_train_data.csv", index=False)
        self.normal_pseudo_label_df.to_csv("normal_pseudo_label_data.csv", index=False)
        self.attack_df.to_csv("attack_data.csv", index=False)
        self.semi_supervised_train_df.to_csv("semi_supervised_train_data.csv", index=False)
        self.semi_supervised_test_df.to_csv("semi_supervised_test_data.csv", index=False)
        self.semi_supervised_test_labels.to_csv("semi_supervised_test_labels.csv", index=False)

        print("Semi-Supervised datasets saved successfully!")

    def prepare_labeled_data_for_training(self, validation_split=0.20):
        """ Prepare labeled data (5%) for training (80% Train, 20% Validation). """
        X = self.labeled_df.drop(columns=[self.label_column])
        y = self.labeled_df[self.label_column]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        print(f"Training Data Shape: {X_train.shape}")
        print(f"Validation Data Shape: {X_val.shape}")

        return X_train, X_val, y_train, y_val
