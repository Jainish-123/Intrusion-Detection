def main(dataset, file_path, test_size=0.3, random_state=42,
         learning_rate=0.001, latent_dim=32,
         epochs=100, batch_size=512, patience = 5, eval_size=0.1,
         student_model_params=None, student_cv_folds=5):
    """
    Run complete anomaly detection pipeline with parameterized inputs
    """
    # Step 1: Data preparation
    print("Loading and preprocessing data...")
    data_handler = DataHandler(file_path, test_size, random_state)

    if(dataset == "network"):
      X_train, X_remain, y_remain = data_handler.prepare_datasets_network()
    else:
      X_train, X_remain, y_remain = data_handler.prepare_datasets_weather()

    print(f"Training data size: {len(X_train)}, {(1-test_size)*100}% normal data")
    print(f"Remaining data size: {len(X_remain)}, normal and attack data")

    # Step 2: Initialize and train Teacher (autoencoder)
    print("\nTraining Teacher model...")
    teacher = Teacher(
        input_dim=X_train.shape[1],
        learning_rate=learning_rate,
        latent_dim=latent_dim
    )
    teacher.train(X_train, epochs=epochs, batch_size=batch_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_remain, y_remain, test_size=0.8, random_state=random_state
    )

    print(f"Validation data size: {len(X_val)}, 20% from remaining data")

    # Step 3: Split test data for evaluation and pseudo-labeling
    X_pseudo, X_eval, y_pseudo, y_eval = train_test_split(
        X_test, y_test, test_size=eval_size, random_state=random_state
    )

    print(f"Pseudo data size: {len(X_pseudo)}, {(1-eval_size)*100}% from 80% of remaining data")
    print(f"Unseen test data size: {len(X_eval)}, {(eval_size)*100}% from 80% of remaining data")

    # Step 4: Optimize threshold on evaluation set
    print("\nOptimizing threshold...")
    eval_errors = teacher.calculate_reconstruction_error(X_val)

    if(dataset == "network"):
      threshold = teacher.optimize_threshold(y_val, eval_errors)
      print(f"Optimal threshold: {threshold:.4f}")
    else:
      threshold = teacher.optimize_threshold(y_val, eval_errors)
      print(f"Optimal threshold: {threshold:.4f}")
      threshold = 0.0020

    # Step 5: Generate pseudo-labels
    print("\nGenerating pseudo-labels...")
    pseudo_labels, pseudo_errors = teacher.generate_pseudo_labels(X_pseudo, threshold)
    data_with_pseudo_labels = pd.DataFrame(X_pseudo)
    data_with_pseudo_labels['original_labels'] = y_pseudo
    data_with_pseudo_labels['pseudo_labels'] = pseudo_labels
    data_with_pseudo_labels.to_csv(f'{dataset}_data_with_pseudo_labels.csv', index=False)

    # Step 6: Evaluate teacher performance
    print("\nEvaluating Teacher on holdout set...")
    teacher_eval = teacher.evaluate_performance(X_eval, y_eval, threshold)
    data_with_unseen_test_labels = pd.DataFrame(X_eval)
    data_with_unseen_test_labels['original_labels'] = y_eval
    data_with_unseen_test_labels['predicted_labels'] = teacher_eval['predictions']
    data_with_unseen_test_labels.to_csv(f'{dataset}_teacher_data_with_test_labels.csv', index=False)

    # Save evaluation results to CSV
    teacher_results = pd.DataFrame([{
        'Model': 'Teacher',
        'Accuracy': teacher_eval['accuracy'],
        'F1 Score': teacher_eval['f1_score']
    }])
    teacher_results.to_csv(f'{dataset}_teacher_model_evaluation.csv', index=False)


    # Step 7: Train and evaluate Student models
    print("\nTraining Student models...")
    student = Student(
        model_params=student_model_params,
        random_state=random_state
    )

    # Train student models
    student_train_results = student.train(
        X_train=X_pseudo,
        y_train=pseudo_labels,
        cv_folds=student_cv_folds
    )

    # Evaluate student models
    student_eval_results, test_prediction_df = student.evaluate(
        X_test=X_eval,
        y_test=y_eval
    )

    student_eval_results.to_csv(f'{dataset}_student_model_evaluation.csv', index=False)
    if test_prediction_df is not None:
      test_prediction_df.to_csv(f'{dataset}_student_test_data_with_predictions.csv', index=False)

    # Get best student model
    best_student = student.get_best_model()

    # Return all artifacts
    return {
        'data_handler': data_handler,
        'teacher': {
            'instance': teacher,
            'evaluation': teacher_eval,
            'threshold': threshold
        },
        'student': {
            'instance': student,
            'training_results': student_train_results,
            'evaluation_results': student_eval_results,
            'best_model': best_student
        },
        'data': {
            'X_train': X_train,
            'X_pseudo': X_pseudo,
            'y_pseudo': pseudo_labels,
            'pseudo_errors': pseudo_errors,
            'X_eval': X_eval,
            'y_eval': y_eval
        },
        'parameters': {
            'test_size': test_size,
            'random_state': random_state,
            'learning_rate': learning_rate,
            'latent_dim': latent_dim,
            'epochs': epochs,
            'batch_size': batch_size,
            'eval_size': eval_size,
            'student_cv_folds': student_cv_folds,
            'student_model_params': student_model_params
        }
    }



if __name__ == '__main__':

    weather_params = {
        'dataset': 'weather',
        'file_path': 'Train_Test_IoT_WeatherNormalAttackP.csv',
        'test_size': 0.20,
        'random_state': 42,
        'learning_rate': 0.001,
        'latent_dim': 16,
        'epochs': 100,
        'batch_size': 512,
        'patience': 15,
        'eval_size': 0.1,
        'student_cv_folds': 10,
        'student_model_params': {
            'XGBoost': {'max_depth': 5, 'learning_rate': 0.1},
            'RandomForest': {'n_estimators': 200}
        }
    }

    results = main(**weather_params)

    # Example of accessing results:
    print("\nBest Student Model:", results['student']['best_model']['name'])
    print("Teacher F1 Score:", results['teacher']['evaluation']['f1_score'])
    print("Best Student F1 Score:", results['student']['evaluation_results'].loc[0, 'F1-Score'])

network_params = {
        'dataset': 'network',
        'file_path': 'train_test_networkP.csv',
        'test_size': 0.85,
        'random_state': 42,
        'learning_rate': 0.001,
        'latent_dim': 32,
        'epochs': 100,
        'batch_size': 512,
        'patience': 5,
        'eval_size': 0.1,
        'student_cv_folds': 10,
        'student_model_params': {
            'XGBoost': {'max_depth': 5, 'learning_rate': 0.1},
            'RandomForest': {'n_estimators': 200}
        }
    }

    results = main(**network_params)

    # Example of accessing results:
    print("\nBest Student Model:", results['student']['best_model']['name'])
    print("Teacher F1 Score:", results['teacher']['evaluation']['f1_score'])
    print("Best Student F1 Score:", results['student']['evaluation_results'].loc[0, 'F1-Score'])