# Main Class

# from data_handler import DataHandler
# from teacher_model import TeacherModel
# from student_model import StudentModel
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
        data_handler.unlabeled_train_df,  # Features
        pseudo_labels_series              # Pseudo Labels
    )

    # Train and evaluate the student models
    student_model.train_and_evaluate(data_handler.unlabeled_test_df, test_labels, teacher_model_name)

    student_model.save_results(teacher_model_name)



semi_supervised_data_handler = DataHandler("train_test_networkP.csv")

semi_supervised_data_handler.load_dataset()
semi_supervised_data_handler.split_normal_attack_data()
semi_supervised_data_handler.split_autoencoder_train_data()
semi_supervised_data_handler.prepare_semi_supervised_data()
semi_supervised_data_handler.save_datasets_semi_supervised()

# Train AutoEncoder on Normal Data
print("\n========== Training AutoEncoder Teacher Model (Semi-Supervised) ==========")
semi_supervised_teacher = TeacherModel()
semi_supervised_teacher.train_autoencoder(semi_supervised_data_handler.normal_train_df)

print("Autoencoder Training Shape:", semi_supervised_data_handler.normal_train_df.shape)
print("Pseudo Label Dataset Shape (Before Fix):", semi_supervised_data_handler.semi_supervised_train_df.shape)


# Generate Pseudo Labels (Normal + Attack) using AutoEncoder
semi_supervised_pseudo_labels = teacher.generate_pseudo_labels_semi_supervised(
    semi_supervised_data_handler.semi_supervised_train_df
)

# Save Pseudo Labels for Semi-Supervised Learning
semi_supervised_file = "pseudo_labeled_autoencoder.csv"
pd.DataFrame(semi_supervised_pseudo_labels, columns=["pseudo_label"]).to_csv(semi_supervised_file, index=False)
print("Pseudo-labels generated and saved for AutoEncoder.")

# Train and Evaluate Student Models (Semi-Supervised)
print("\n========== Training Student Models (Semi-Supervised) ==========")

if os.path.exists(semi_supervised_file):
    pseudo_labels_df = pd.read_csv(semi_supervised_file)

    semi_supervised_student_model = StudentModel(
        semi_supervised_data_handler.semi_supervised_train_df,  # Features for training
        semi_supervised_data_handler.semi_supervised_train_labels 
    )

    semi_supervised_student_model.train_and_evaluate_semi_supervised(
        semi_supervised_data_handler.semi_supervised_test_df,  # Features for testing
        semi_supervised_data_handler.semi_supervised_test_labels  # Correct testing labels
    )


    semi_supervised_student_model.save_results("AutoEncoder", semi_supervised=True)