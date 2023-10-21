from utils import predict_and_eval
from utils import train_test_dev_split, read_digits, preprocess_data, get_hyperparameter_combinations, tune_hparams
from joblib import dump, load
import pandas as pd


def main():
    test_sizes = [0.1, 0.2, 0.3]
    dev_sizes = [0.1, 0.2, 0.3]
    classifier_param_dict = {}
    gamma_list = [0.001, 0.01, 0.1, 1]
    C_list = [1, 10, 100, 1000]
    h_params_combinations = get_hyperparameter_combinations([gamma_list, C_list], ['gamma', 'C'])
    classifier_param_dict['svm'] = h_params_combinations
    # 2.2 Decision Tree
    max_depth_list = [5, 10, 15, 20, 50, 100]
    h_params_trees_combinations = get_hyperparameter_combinations([max_depth_list], ['max_depth'])
    classifier_param_dict['tree'] = h_params_trees_combinations
    total_run = 5
    results = []
    for run_num in range(total_run):
        for test_size in test_sizes:
            for dev_size in dev_sizes:
                train_size = "{:.2f}".format(1 - test_size - dev_size)
                X, y = read_digits()
                X_train, X_dev, X_test, y_train, y_dev, y_test = train_test_dev_split(X, y, test_size, dev_size)
                X_train = preprocess_data(X_train)
                X_test = preprocess_data(X_test)
                X_dev = preprocess_data(X_dev)
                # 1. SVM

                model_type = 'svm'
                print(model_type)
                list_hparam_combinations = classifier_param_dict.get('svm')
                best_hyper_params, best_model_path, best_accuracy, best_model = tune_hparams(X_train, y_train, X_dev,
                                                                                             y_dev,
                                                                                             list_hparam_combinations,
                                                                                             model_type=model_type)
                # loading of model
                # best_model = load(best_model_path)
                test_accuracy = "{:.2f}".format(predict_and_eval(best_model, X_test, y_test))
                dev_accuracy = "{:.2f}".format(best_accuracy)
                train_accuracy = "{:.2f}".format(predict_and_eval(best_model, X_train, y_train))
                print(
                    f"Train Size {train_size} Test Size {test_size} Dev Size {dev_size} - Train accuracy : {train_accuracy} Test accuracy : {test_accuracy} Dev accuracy : {dev_accuracy}")
                results.append({'run_num': run_num, 'model_type': model_type, 'train_accuracy': train_accuracy,
                                'val_accuracy': dev_accuracy, 'test_acc': test_accuracy,
                                'best_hparams': best_hyper_params})

                # 2. Decision Tree
                model_type = 'tree'
                print(model_type)
                list_hparam_combinations = classifier_param_dict.get('tree')
                best_hyper_params, best_model_path, best_accuracy, best_model = tune_hparams(X_train, y_train, X_dev,
                                                                                             y_dev,
                                                                                             list_hparam_combinations,
                                                                                             model_type=model_type)
                # loading of model
                # best_model = load(best_model_path)
                test_accuracy = "{:.2f}".format(predict_and_eval(best_model, X_test, y_test))
                dev_accuracy = "{:.2f}".format(best_accuracy)
                train_accuracy = "{:.2f}".format(predict_and_eval(best_model, X_train, y_train))
                print(
                    f"Train Size {train_size} Test Size {test_size} Dev Size {dev_size} - Train accuracy : {train_accuracy} Test accuracy : {test_accuracy} Dev accuracy : {dev_accuracy}")
                results.append({'run_num': run_num, 'model_type': model_type, 'train_accuracy': train_accuracy,
                                'val_accuracy': dev_accuracy, 'test_acc': test_accuracy,
                                'best_hparams': best_hyper_params})

    result_df = pd.DataFrame(results)
    print(result_df.groupby('model_type').describe().T)


if __name__ == "__main__":
    main()
