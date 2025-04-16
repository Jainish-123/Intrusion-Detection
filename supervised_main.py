import pandas as pd
import os
# from data_handler import DataHandler
# from teacher_model import TeacherModel
# from student_model import StudentModel

def train_supervised(dataset_path):
    """ Supervised Learning Pipeline """
    print(f"\n========== Supervised Learning: {dataset_path} ==========")

    data_handler = DataHandler(dataset_path)
    data_handler.load_dataset()
    data_handler.split_labeled_unlabeled()
    data_handler.split_unlabeled_train_test()
    data_handler.save_datasets()

    X_train, X_val_test, y_train, y_val_test = data_handler.prepare_labeled_data_for_training()
    X_unseen_test, y_unseen_test = data_handler.unlabeled_test_df, data_handler.test_labels

    teacher = TeacherModel(X_train, y_train)
    teacher.train_with_cross_validation(dataset_path.replace(".csv", ""))  # Updated to separate training
    teacher.evaluate_supervised_model(X_val_test, y_val_test, dataset_path.replace(".csv", "_validation_test"))  # Separate evaluation
    teacher.evaluate_supervised_model(X_unseen_test, y_unseen_test, dataset_path.replace(".csv", "_unseen_test"))  # Separate evaluation

    for teacher_model_name in teacher.trained_models.keys():
        pseudo_labels = teacher.generate_pseudo_labels(data_handler.unlabeled_train_df, teacher_model_name)
        pseudo_labels_series = pd.Series(pseudo_labels, name='pseudo_label')

        student_model = StudentModel(data_handler.unlabeled_train_df, pseudo_labels_series)
        student_model.train_models()  # Separate training
        student_model.evaluate_models(X_unseen_test, y_unseen_test, teacher_model_name)  # Separate evaluation
        student_model.save_results(dataset_path.replace(".csv", ""))

if __name__ == "__main__":
    # Run Supervised Learning on multiple datasets
    supervised_datasets = ["train_test_networkP.csv", "Train_Test_IoT_WeatherNormalAttackP.csv"]
    for dataset in supervised_datasets:
        train_supervised(dataset)
