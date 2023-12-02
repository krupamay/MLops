from utils import predict_and_eval, train_model
from utils import train_test_dev_split, read_digits, preprocess_data, get_hyperparameter_combinations, tune_hparams
from sklearn.model_selection import cross_val_score
from joblib import dump, load
import pandas as pd
import os
import argparse


def main(args):
    test_sizes = [float(x) for x in args.test_sizes.split(",")]
    dev_sizes = [float(x) for x in args.dev_sizes.split(",")]
    classifier_param_dict = {}
    gamma_list = [float(x) for x in args.gamma_list.split(",")]
    C_list = [int(x) for x in args.C_list.split(",")]
    h_params_combinations = get_hyperparameter_combinations([gamma_list, C_list], ['gamma', 'C'])
    classifier_param_dict['svm'] = h_params_combinations
    # 2.2 Decision Tree
    max_depth_list = [5, 10, 15, 20, 50, 100]
    h_params_trees_combinations = get_hyperparameter_combinations([max_depth_list], ['max_depth'])
    classifier_param_dict['tree'] = h_params_trees_combinations
    solvers = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
    h_params_lr_combinations = get_hyperparameter_combinations([solvers], ['solver'])
    classifier_param_dict['lr'] = h_params_lr_combinations
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
                # print(model_type)
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
                # print(
                #     f"Train Size {train_size} Test Size {test_size} Dev Size {dev_size} - Train accuracy : {train_accuracy} Test accuracy : {test_accuracy} Dev accuracy : {dev_accuracy}")
                results.append({'run_num': run_num, 'model_type': model_type, 'train_accuracy': train_accuracy,
                                'val_accuracy': dev_accuracy, 'test_acc': test_accuracy,
                                'best_hparams': best_hyper_params})

                # 2. Decision Tree
                model_type = 'tree'
                # print(model_type)
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
                # print(
                #     f"Train Size {train_size} Test Size {test_size} Dev Size {dev_size} - Train accuracy : {train_accuracy} Test accuracy : {test_accuracy} Dev accuracy : {dev_accuracy}")
                results.append({'run_num': run_num, 'model_type': model_type, 'train_accuracy': train_accuracy,
                                'val_accuracy': dev_accuracy, 'test_acc': test_accuracy,
                                'best_hparams': best_hyper_params})

                # 3. Logistic Regression
                model_type = 'lr'
                list_hparam_combinations = classifier_param_dict.get('lr')
                models_dir = "./models"
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir)
                for params in list_hparam_combinations:
                    # Train model with current hyperparameters
                    model = train_model(X_train, y_train, model_params=params, model_type=model_type)
                    score = cross_val_score(model, X_train, y_train, cv=5)
                    mean = score.mean()
                    std = score.std()

                    model_file_name = 'M22AIE211_lr_' + params.get('solver') + ".joblib"
                    dump(model, os.path.join(models_dir, model_file_name))
                    test_accuracy = "{:.2f}".format(predict_and_eval(best_model, X_test, y_test))
                    dev_accuracy = "{:.2f}".format(predict_and_eval(best_model, X_dev, y_dev))
                    train_accuracy = "{:.2f}".format(predict_and_eval(best_model, X_train, y_train))
                    print(
                        f"Solver name {params.get('solver')} Train Size {train_size} Test Size {test_size} Dev Size {dev_size} - Train accuracy : {train_accuracy} Test accuracy : {test_accuracy} Dev accuracy : {dev_accuracy} Mean {mean} STD {std}")

                # results.append({'run_num': run_num, 'model_type': model_type, 'train_accuracy': train_accuracy,
                #                 'val_accuracy': dev_accuracy, 'test_acc': test_accuracy,
                #                 'best_hparams': best_hyper_params})

    result_df = pd.DataFrame(results)
    print(result_df.groupby('model_type').describe().T)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Take parameters from command line.")

    parser.add_argument("--test_sizes", type=str, default="0.1", help="Comma separated test sizes.")
    parser.add_argument("--dev_sizes", type=str, default="0.1", help="Comma separated dev sizes.")
    parser.add_argument("--gamma_list", type=str, default="0.001,0.01,0.1", help="Comma separated gamma values.")
    parser.add_argument("--C_list", type=str, default="1,10,100", help="Comma separated C values.")

    args = parser.parse_args()
    main(args)
