class Teacher:
    def __init__(self, input_dim, learning_rate=0.001, latent_dim=32):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
        self.model = self._build_autoencoder()
        self.threshold = None

    def _build_autoencoder(self):
        """Build the autoencoder architecture"""
        input_layer = Input(shape=(self.input_dim,))

        # Encoder
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dense(self.latent_dim, activation='relu')(encoded)

        # Decoder
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(self.input_dim, activation='linear')(decoded)

        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mae')
        return autoencoder

    # def _build_weather_autoencoder(self):
    #     """Simplified architecture with regularization"""

    #     input_layer = Input(shape=(self.input_dim,))

    #     # Encoder
    #     encoded = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
    #     encoded = Dropout(0.3)(encoded)
    #     encoded = Dense(64, activation='relu')(input_layer)
    #     encoded = Dense(self.latent_dim, activation='relu')(encoded)

    #     # Decoder
    #     decoded = Dense(64, activation='relu')(encoded)
    #     decoded = Dense(self.input_dim, activation='linear')(decoded)

    #     autoencoder = Model(inputs=input_layer, outputs=decoded)
    #     autoencoder.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mae')
    #     return autoencoder

    def train(self, X_train, epochs=100, batch_size=512, validation_split=0.1, patience=5):
        """Train the autoencoder"""
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )

        history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1
        )
        return history

    def calculate_reconstruction_error(self, X):
        """Calculate MAE reconstruction error"""
        reconstructions = self.model.predict(X, verbose=0)
        return np.mean(np.abs(X - reconstructions), axis=1)

    def optimize_threshold(self, y_true, errors):
        """Find optimal threshold using ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, errors)
        optimal_idx = np.argmax(tpr - fpr)
        self.threshold = thresholds[optimal_idx]
        return self.threshold

    def generate_pseudo_labels(self, X, threshold=None):
        """Generate pseudo-labels for input data"""
        threshold = threshold or self.threshold
        if threshold is None:
            raise ValueError("Threshold must be provided or set using optimize_threshold()")

        errors = self.calculate_reconstruction_error(X)
        return (errors > threshold).astype(int), errors

    def evaluate_performance(self, X, y_true, threshold=None):
        """Evaluate model performance with metrics and plots"""
        threshold = threshold or self.threshold
        if threshold is None:
            raise ValueError("Threshold must be provided or set using optimize_threshold()")

        errors = self.calculate_reconstruction_error(X)
        y_pred = (errors > threshold).astype(int)

        # Calculate all metrics
        clf_report = classification_report(y_true, y_pred, output_dict=True)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_true, errors)
        avg_precision = average_precision_score(y_true, errors)

        # Print summary
        print("\nTeacher Model Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

        # Plots
        self._plot_metrics(y_true, errors)

        return {
            'classification_report': clf_report,
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'threshold': threshold,
            'predictions': y_pred
        }

    def _plot_metrics(self, y_true, errors):
        """Plot ROC and Precision-Recall curves"""
        fpr, tpr, _ = roc_curve(y_true, errors)
        precision, recall, _ = precision_recall_curve(y_true, errors)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc_score(y_true, errors):.2f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label=f'AP = {average_precision_score(y_true, errors):.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()

        plt.tight_layout()
        plt.show()