# Main Class

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

    X_train, X_val, y_train, y_val = data_handler.prepare_labeled_data_for_training()

    teacher = TeacherModel(X_train, y_train)
    teacher.train_with_cross_validation()

    for teacher_model_name in teacher.trained_models.keys():
        pseudo_labels = teacher.generate_pseudo_labels(data_handler.unlabeled_train_df, teacher_model_name)
        pseudo_labels_series = pd.Series(pseudo_labels, name='pseudo_label')

        student_model = StudentModel(data_handler.unlabeled_train_df, pseudo_labels_series)
        student_model.train_and_evaluate(data_handler.unlabeled_test_df, data_handler.test_labels, teacher_model_name)
        student_model.save_results()

def train_semi_supervised(dataset_path):
    """ Semi-Supervised Learning Pipeline """
    print(f"\n========== Semi-Supervised Learning: {dataset_path} ==========")

    data_handler = DataHandler(dataset_path)
    data_handler.load_dataset()
    data_handler.split_normal_attack_data()
    data_handler.split_autoencoder_train_data()
    data_handler.prepare_semi_supervised_data()
    data_handler.save_datasets_semi_supervised()

    teacher = TeacherModel()
    teacher.train_autoencoder(data_handler.normal_train_df)

    pseudo_labels = teacher.generate_pseudo_labels_semi_supervised(data_handler.semi_supervised_train_df)
    pseudo_labels_df = pd.DataFrame(pseudo_labels, columns=["pseudo_label"])
    pseudo_labels_file = "pseudo_labeled_autoencoder.csv"
    pseudo_labels_df.to_csv(pseudo_labels_file, index=False)

    print("Pseudo-labels generated and saved for AutoEncoder.")

    if os.path.exists(pseudo_labels_file):
        student_model = StudentModel(data_handler.semi_supervised_train_df, data_handler.semi_supervised_train_labels)
        student_model.train_and_evaluate(data_handler.semi_supervised_test_df, data_handler.semi_supervised_test_labels, "AutoEncoder", mode="semi-supervised")
        student_model.save_results(semi_supervised=True)

if __name__ == "__main__":
    # Run Supervised Learning on multiple datasets
    supervised_datasets = ["train_test_networkP.csv", "Train_Test_IoT_WeatherNormalAttackP.csv"]
    for dataset in supervised_datasets:
        train_supervised(dataset)

    # Run Semi-Supervised Learning
    train_semi_supervised("train_test_networkP.csv")
