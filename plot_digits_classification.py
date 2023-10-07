from utils import predict_and_eval
from utils import train_test_dev_split, read_digits, preprocess_data, get_hyperparameter_combinations, tune_hparams
from joblib import dump, load


def main():
    test_sizes = [0.1, 0.2, 0.3]
    dev_sizes = [0.1, 0.2, 0.3]
    gamma_list = [0.001, 0.01, 0.1, 1]
    C_list = [1, 10, 100, 1000]
    params = {
        'gamma': gamma_list,
        'C': C_list
    }
    for test_size in test_sizes:
        for dev_size in dev_sizes:
            train_size = "{:.2f}".format(1 - test_size - dev_size)
            X, y = read_digits()
            X_train, X_dev, X_test, y_train, y_dev, y_test = train_test_dev_split(X, y, test_size, dev_size)
            X_train = preprocess_data(X_train)
            X_test = preprocess_data(X_test)
            X_dev = preprocess_data(X_dev)
            list_hparam_combinations = get_hyperparameter_combinations(params)
            best_hyper_params, best_model_path, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev,
                                                                             list_hparam_combinations)
            # loading of model
            best_model = load(best_model_path)
            print(f"Total samples in the train dataset {len(X_train)}")
            print(f"Total samples in the test dataset {len(X_test)}")
            print(f"Total samples in the dev dataset {len(X_dev)}")
            image_height, image_width = X.shape[1], X.shape[2]
            print(f"Height of image is {image_height} and width of the image is {image_width}")
            gamma_list = [0.081, 0.01, 0.1, 1, 10, 108]
            C_list = [0.1, 1, 2, 5, 10]
            list_of_all_param_combination_dictionaries = get_all_hyper_params(gamma_list, C_list)
            # print(list_of_all_param_combination_dictionaries)
            best_hyper_params, best_model, best_accuracy = tune_hyper_parameter(X_train, y_train, X_dev, y_dev, list_of_all_param_combination_dictionaries)

if __name__ == "__main__":
    main()
