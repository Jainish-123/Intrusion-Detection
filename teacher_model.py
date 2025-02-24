# TeacherModel Class

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

class TeacherModel:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.models = self._initialize_models()
        self.trained_models = {}

    def _initialize_models(self):
        return {
            "XGBoost": XGBClassifier(eval_metric='logloss'),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42)
        }

    def train_with_cross_validation(self, folds=10):
        best_scores = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, self.X_train, self.y_train, cv=folds, scoring='accuracy')
            avg_score = scores.mean()
            best_scores[name] = avg_score
            print(f"{name} - Avg Accuracy (10-fold CV): {avg_score:.4f}")

            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model
        return best_scores

    def generate_pseudo_labels(self, unlabeled_df, model_name):
        model = self.trained_models.get(model_name)
        if model:
            preds = model.predict(unlabeled_df)
            return preds
        else:
            print(f"Model {model_name} not found!")
            return None