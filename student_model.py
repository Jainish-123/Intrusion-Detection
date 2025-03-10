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
    def __init__(self, pseudo_labels_df, true_labels, verbose=True):
        self.X_train = pseudo_labels_df  # Features from pseudo labels
        self.y_train = true_labels       # True labels for the test set
        self.verbose = verbose
        self.models = self._initialize_models()
        self.trained_models = {}
        self.results_df = pd.DataFrame()
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0

    def _log(self, message):
        """ Print messages only if verbose is enabled. """
        if self.verbose:
            print(message)

    def _initialize_models(self):
        return {
            "XGBoost": XGBClassifier(eval_metric='logloss'),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42)
        }

    def _update_best_model(self, name, model, acc):
        """ Update the best model based on accuracy. """
        if acc > self.best_score:
            self.best_score = acc
            self.best_model = model
            self.best_model_name = name

    def train_and_evaluate(self, X_test, y_test, teacher_model_name="AutoEncoder", mode="supervised", plot_cm=True):
        """ Train student models and evaluate performance. Supports supervised & semi-supervised learning. """

        self._log(f"Starting training for {mode} learning...")
        results = []

        for name, model in self.models.items():
            # 10-Fold Cross-Validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=10, scoring='accuracy')
            avg_cv_score = cv_scores.mean()

            # Train model
            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model

            # Predictions
            y_pred = model.predict(X_test)

            # Compute metrics
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # Store results
            results.append({
                "Teacher Model": teacher_model_name,
                "Student Model": name,
                "Cross-Validation Accuracy": avg_cv_score,
                "Test Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1
            })

            self._update_best_model(name, model, acc)

            # Generate confusion matrix
            if plot_cm:
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
                plt.xlabel("Predicted Labels")
                plt.ylabel("True Labels")
                plt.title(f"Confusion Matrix ({teacher_model_name} → {name})")
                plt.show()

        self.results_df = pd.DataFrame(results)
        self._log(f"Training complete for {teacher_model_name} ({mode} learning).")

    def save_results(self, semi_supervised=False):
        """ Save results for supervised and semi-supervised learning. """
        file_path = "semi_supervised_student_results.csv" if semi_supervised else "student_model_results.csv"

        os.makedirs("results", exist_ok=True)

        if os.path.exists(file_path):
            self.results_df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            self.results_df.to_csv(file_path, index=False)

        self._log(f"Results saved to {file_path}!")

    def get_best_student_model(self):
        """ Return the best student model and its accuracy. """
        return self.best_model_name, self.best_score
