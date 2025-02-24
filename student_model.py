# StudentModel Class

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

class StudentModel:
    def __init__(self, pseudo_labels_df, true_labels):
        self.X_train = pseudo_labels_df  # Features from pseudo labels
        self.y_train = true_labels       # True labels for the test set
        self.models = self._initialize_models()
        self.trained_models = {}
        self.results = {}

    def _initialize_models(self):
        return {
            "XGBoost": XGBClassifier(eval_metric='logloss'),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42)
        }

    def train_and_evaluate(self, X_test, y_test):
        for name, model in self.models.items():
            # Cross-validation on pseudo-labeled data
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=10, scoring='accuracy')
            avg_cv_score = cv_scores.mean()

            # Train on all pseudo-labeled data
            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model

            # Evaluate on true labeled test data
            y_pred = model.predict(X_test)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Store results
            self.results[name] = {
                "Cross-Validation Accuracy": avg_cv_score,
                "Test Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1
            }

    def print_results(self, teacher_model_name):
        print(f"\nResults for Teacher Model: {teacher_model_name}")
        for student_name, metrics in self.results.items():
            print(f"\nStudent Model: {student_name}")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")