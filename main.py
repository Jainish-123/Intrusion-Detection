# Main Class

from data_handler import DataHandler
from teacher_model import TeacherModel
from student_model import StudentModel
import pandas as pd

# Step 1: Initialize DataHandler
data_handler = DataHandler("train_test_networkP.csv")

# Step 2: Load and split dataset
data_handler.load_dataset()
data_handler.split_labeled_unlabeled()
unlabeled_train, unlabeled_test, train_labels, test_labels = data_handler.split_unlabeled_train_test()

# Step 3: Save datasets automatically
data_handler.save_datasets()

# Step 4: Prepare labeled data for training (Train/Validation Split)
X_train, X_val, y_train, y_val = data_handler.prepare_labeled_data_for_training()

# Step 5: Train Teacher Models with 10-Fold Cross-Validation
teacher = TeacherModel(X_train, y_train)
teacher_scores = teacher.train_with_cross_validation()

# Step 6: Train Student Models with pseudo-labels from each teacher model
for teacher_model_name in teacher.trained_models.keys():
    # Generate pseudo labels
    pseudo_labels = teacher.generate_pseudo_labels(data_handler.unlabeled_train_df, teacher_model_name)

    # Convert pseudo labels to Series (1D data for training)
    pseudo_labels_series = pd.Series(pseudo_labels, name='pseudo_label')

    # Initialize and Train Student Models
    student_model = StudentModel(
        data_handler.unlabeled_train_df,  # Original features for training
        pseudo_labels_series              # Pseudo labels as target values
    )

    # Train and evaluate the student models
    student_model.train_and_evaluate(data_handler.unlabeled_test_df, test_labels)

    # Print results for each teacher-student model pair
    student_model.print_results(teacher_model_name)

