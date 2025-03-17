import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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
        self.best_f1_score = 0  # Use F1-Score instead of Accuracy per professor's instruction
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)  # Ensure results directory exists

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

    def _update_best_model(self, name, model, f1):
        """ Update the best model based on F1-Score (instead of Accuracy). """
        if f1 > self.best_f1_score:
            self.best_f1_score = f1
            self.best_model = model
            self.best_model_name = name

    def train_models(self):
        """ Train student models using cross-validation """
        self._log("Starting training for student models...")
        results = []

        for name, model in self.models.items():
            # 10-Fold Cross-Validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=10, scoring='accuracy')
            avg_cv_score = cv_scores.mean()

            # Train model on full dataset
            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model

            results.append({
                "Student Model": name,
                "Cross-Validation Accuracy": avg_cv_score
            })

        self.results_df = pd.DataFrame(results)

        print("\n Cross validation results in student model training")
        print(self.results_df)

        self._log("\nTraining complete for student models.")

    def evaluate_models(self, X_test, y_test, teacher_model_name="AutoEncoder", plot_cm=True):
        """ Evaluate trained student models on test data """
        self._log("Starting evaluation on test dataset...")
        results = []

        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)

            # Compute metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            results.append({
                "Teacher Model": teacher_model_name,
                "Student Model": name,
                "Test Accuracy": acc,
                "F1-Score": f1
            })

            self._update_best_model(name, model, f1)  # Update best model based on F1-Score

            # Generate confusion matrix
            if plot_cm:
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
                plt.xlabel("Predicted Labels")
                plt.ylabel("True Labels")
                plt.title(f"Confusion Matrix ({teacher_model_name} → {name})")
                plt.savefig(os.path.join(self.results_dir, f"confusion_matrix_{teacher_model_name}_{name}.png"))
                plt.show()

        self.results_df = pd.DataFrame(results)

        print("\n Student model evaluation on unseen test data")
        print(self.results_df)

        self._log("\nEvaluation complete for student models.")

    def save_results(self, semi_supervised=False):
        """ Save results for supervised and semi-supervised learning. """
        file_path = os.path.join(self.results_dir, "semi_supervised_student_results.csv" if semi_supervised else "student_model_results.csv")

        if os.path.exists(file_path):
            self.results_df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            self.results_df.to_csv(file_path, index=False)

        self._log(f"Results saved to {file_path}!")

    def get_best_student_model(self):
        """ Return the best student model and its F1-Score. """
        return self.best_model_name, self.best_f1_score
