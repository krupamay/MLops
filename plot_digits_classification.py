from utils import predict_and_eval
from utils import split_train_dev_test, get_data, preprocess_data, get_all_hyper_params, tune_hyper_parameter


def main():
    test_sizes = [0.1, 0.2, 0.3]
    dev_sizes = [0.1, 0.2, 0.3]
    for test_size in test_sizes:
        for dev_size in dev_sizes:
            train_size = "{:.2f}".format(1 - test_size - dev_size)
            X, y = get_data()
            X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X, y, test_size, dev_size)
            X_train = preprocess_data(X_train)
            X_test = preprocess_data(X_test)
            X_dev = preprocess_data(X_dev)
            list_of_all_param_combination_dictionaries = get_all_hyper_params()
            # print(list_of_all_param_combination_dictionaries)
            best_hyper_params, best_model, best_accuracy = tune_hyper_parameter(X_train, y_train, X_dev, y_dev, list_of_all_param_combination_dictionaries)
            test_accuracy = "{:.2f}".format(predict_and_eval(best_model, X_test, y_test))
            dev_accuracy = "{:.2f}".format(best_accuracy)
            train_accuracy = "{:.2f}".format(predict_and_eval(best_model, X_train, y_train))
            print(f"Train Size {train_size} Test Size {test_size} Dev Size {dev_size} - Train accuracy : {train_accuracy} Test accuracy : {test_accuracy} Dev accuracy : {dev_accuracy}")


if __name__ == "__main__":
    main()