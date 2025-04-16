class Student:
    def __init__(self, model_params=None, random_state=42, verbose=True):
        self.random_state = random_state
        self.verbose = verbose
        self.models = self._initialize_models(model_params)
        self.trained_models = {}
        self.results = {
            'training': None,
            'evaluation': None
        }
        self.best_model_info = {
            'name': None,
            'model': None,
            'score': 0.0
        }

    def _log(self, message):
        """Print messages if verbose is enabled"""
        if self.verbose:
            print(message)

    def _initialize_models(self, model_params=None):
        """Initialize student models with optional custom parameters"""
        # Default configurations
        default_models = {
            'RandomForest': {
                'n_estimators': 100,
                'random_state': self.random_state
            },
            'XGBoost': {
                'eval_metric': 'logloss',
                'random_state': self.random_state
            },
            'DecisionTree': {
                'random_state': self.random_state
            },
            'Bagging': {
                'estimator': DecisionTreeClassifier(random_state=self.random_state),
                'n_estimators': 50,
                'random_state': self.random_state
            },
            'AdaBoost': {
                'n_estimators': 50,
                'random_state': self.random_state
            }
        }

        # Update defaults with any user-provided parameters
        if model_params:
            for model_name, params in model_params.items():
                if model_name in default_models:
                    default_models[model_name].update(params)

        # Instantiate models
        models = {
            'RandomForest': RandomForestClassifier(**default_models['RandomForest']),
            'XGBoost': XGBClassifier(**default_models['XGBoost']),
            'DecisionTree': DecisionTreeClassifier(**default_models['DecisionTree']),
            'Bagging': BaggingClassifier(**default_models['Bagging']),
            'AdaBoost': AdaBoostClassifier(**default_models['AdaBoost'])
        }

        return models

    def train(self, X_train, y_train, cv_folds=5):
        """Train student models using cross-validation"""
        self._log("\nTraining Student Models...")
        results = []

        for name, model in self.models.items():
            self._log(f"Training {name}...")

            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds, scoring='f1_weighted'
            )
            avg_score = np.mean(cv_scores)

            # Full training
            model.fit(X_train, y_train)
            self.trained_models[name] = model

            results.append({
                'model': name,
                'cv_f1_mean': avg_score,
                'cv_f1_scores': cv_scores,
                'status': 'completed'
            })

            # Update best model
            if avg_score > self.best_model_info['score']:
                self.best_model_info = {
                    'name': name,
                    'model': model,
                    'score': avg_score
                }

        self.results['training'] = pd.DataFrame(results)
        self._log("\nTraining Results:")
        self._log(self.results['training'][['model', 'cv_f1_mean']])

        return self.results['training']

    def evaluate(self, X_test, y_test, plot_cm=True, cm_labels=None):
        """Evaluate student models on test data"""
        if not self.trained_models:
            raise ValueError("No models trained yet. Call train() first.")

        self._log("\nEvaluating Student Models...")
        results = []
        all_predictions = []
        cm_labels = cm_labels or ['Normal', 'Attack']

        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            results.append({
                'Student Model': name,
                'Test Accuracy': acc,
                'F1-Score': f1
            })

            test_df = pd.DataFrame(X_test)  # Ensure it's a DataFrame
            test_df['model'] = name
            test_df['original_labels'] = y_test
            test_df['predicted_labels'] = y_pred
            # Store for returning
            all_predictions.append(test_df)

            # Plot confusion matrix
            if plot_cm:
                self._plot_confusion_matrix(
                    y_test, y_pred,
                    title=f"{name} Confusion Matrix",
                    labels=cm_labels
                )

        self.results['evaluation'] = pd.DataFrame(results)
        self._log("\nEvaluation Results:")
        self._log(self.results['evaluation'])

        if all_predictions:
          return self.results['evaluation'], pd.concat(all_predictions, ignore_index=True)
        else:
          return self.results['evaluation'], None

    def _plot_confusion_matrix(self, y_true, y_pred, title, labels):
        """Helper method to plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def get_best_model(self):
        """Get the best performing student model"""
        if not self.best_model_info['model']:
            raise ValueError("No models trained yet. Call train() first.")

        return self.best_model_info