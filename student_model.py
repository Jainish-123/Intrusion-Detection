# StudentModel Class

import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

class StudentModel:
    def __init__(self, pseudo_labels_df, true_labels):
        self.X_train = pseudo_labels_df  # Features from pseudo labels
        self.y_train = true_labels       # True labels for the test set
        self.models = self._initialize_models()
        self.trained_models = {}
        self.results = []
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0

    def _initialize_models(self):
        return {
            "XGBoost": XGBClassifier(eval_metric='logloss'),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42)
        }

    def train_and_evaluate(self, X_test, y_test, teacher_model_name):

        for name, model in self.models.items():
            # 10-Fold Cross-Validation on pseudo-labeled data
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=10, scoring='accuracy')
            avg_cv_score = cv_scores.mean()

            # Train model
            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model

            # Predictions on test data
            y_pred = model.predict(X_test)

            # Calculate Evaluation Metrics
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # Store results
            self.results.append({
                "Teacher Model": teacher_model_name,
                "Student Model": name,
                "Cross-Validation Accuracy": avg_cv_score,
                "Test Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1
            })

            # Save the best model based on accuracy
            if acc > self.best_score:
                self.best_score = acc
                self.best_model = model
                self.best_model_name = name

            # Generate and plot confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.title(f"Confusion Matrix ({teacher_model_name} → {name})")
            plt.show()

        print(f"Training complete for Teacher Model: {teacher_model_name}")

    def train_and_evaluate_semi_supervised(self, X_test, y_test):
      """Train student models using autoencoder-generated pseudo labels (semi-supervised learning)."""

      # Ensure input sizes match before training
      assert len(self.X_train) == len(self.y_train), f"Mismatch: X_train size {len(self.X_train)} != y_train size {len(self.y_train)}"
      assert len(X_test) == len(y_test), f"Mismatch: X_test size {len(X_test)} != y_test size {len(y_test)}"

      for name, model in self.models.items():
          # 10-Fold Cross-Validation on pseudo-labeled data (from autoencoder)
          cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=10, scoring='accuracy')
          avg_cv_score = cv_scores.mean()

          # Train model
          model.fit(self.X_train, self.y_train)
          self.trained_models[name] = model

          # Predictions on test data
          y_pred = model.predict(X_test)

          # Calculate Evaluation Metrics
          acc = accuracy_score(y_test, y_pred)
          precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
          recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
          f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

          # Store results
          self.results.append({
              "Teacher Model": "AutoEncoder",
              "Student Model": name,
              "Cross-Validation Accuracy": avg_cv_score,
              "Test Accuracy": acc,
              "Precision": precision,
              "Recall": recall,
              "F1-Score": f1
          })

          # Save the best model based on accuracy
          if acc > self.best_score:
              self.best_score = acc
              self.best_model = model
              self.best_model_name = name

          # Generate and plot confusion matrix
          cm = confusion_matrix(y_test, y_pred)
          plt.figure(figsize=(6, 5))
          sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
          plt.xlabel("Predicted Labels")
          plt.ylabel("True Labels")
          plt.title(f"Confusion Matrix (AutoEncoder → {name})")
          plt.show()

      print("Training complete for Semi-Supervised Learning (AutoEncoder).")


    def save_results(self, teacher_model_name, semi_supervised=False):
        """Save results separately for supervised and semi-supervised learning."""
        results_df = pd.DataFrame(self.results)

        # Ensure the results directory exists
        os.makedirs("results", exist_ok=True)

        # Define the file path
        file_path = "semi_supervised_student_results.csv" if semi_supervised else "student_model_results.csv"

        # If the file already exists, append new results without overwriting
        if os.path.exists(file_path):
            results_df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            results_df.to_csv(file_path, index=False)

        print(f"Results for {teacher_model_name} saved successfully!")

    def get_best_student_model(self):
        return self.best_model_name, self.best_score
