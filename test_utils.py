from utils import get_hyperparameter_combinations, train_test_dev_split, read_digits, tune_hparams, preprocess_data
from sklearn import datasets
import numpy as np
from api.app import app
import os


# def test_for_hparam_combinations_count():
#     # a test case to check that all possible combinations of parameters are indeed generated
#     gamma_list = [0.001, 0.01, 0.1, 1]
#     C_list = [1, 10, 100, 1000]
#     h_params_combinations = get_hyperparameter_combinations([gamma_list, C_list], ['gamma', 'C'])
#
#     assert len(h_params_combinations) == len(gamma_list) * len(C_list)
#
#
# def create_dummy_hyperparameter():
#     gamma_list = [0.001, 0.01]
#     C_list = [1]
#     h_params_combinations = get_hyperparameter_combinations([gamma_list, C_list], ['gamma', 'C'])
#     return h_params_combinations


def read_digits():
    digits = datasets.load_digits()
    result = {}
    for digit in range(10):
        index = np.where(digits.target == digit)[0][0]
        result[digit] = digits.images[index]

    return result
#
#
# def create_dummy_data():
#     X, y = read_digits()
#
#     X_train = X[:100, :, :]
#     y_train = y[:100]
#     X_dev = X[:50, :, :]
#     y_dev = y[:50]
#
#     X_train = preprocess_data(X_train)
#     X_dev = preprocess_data(X_dev)
#
#     return X_train, y_train, X_dev, y_dev
#
#
# def test_for_hparam_combinations_values():
#     h_params_combinations = create_dummy_hyperparameter()
#
#     expected_param_combo_1 = {'gamma': 0.001, 'C': 1}
#     expected_param_combo_2 = {'gamma': 0.01, 'C': 1}
#
#     assert (expected_param_combo_1 in h_params_combinations) and (expected_param_combo_2 in h_params_combinations)
#
#
# def test_model_saving():
#     X_train, y_train, X_dev, y_dev = create_dummy_data()
#     h_params_combinations = create_dummy_hyperparameter()
#
#     _, best_model_path, _, _ = tune_hparams(X_train, y_train, X_dev,
#                                             y_dev, h_params_combinations, model_type='svm')
#     assert os.path.exists(best_model_path)
#
#
# def test_data_splitting():
#     print("test data splitting")
#     X, y = read_digits()
#     X = X[:100, :, :]
#     y = y[:100]
#     test_size = .1
#     dev_size = .6
#     X_train, X_dev, X_test, y_train, y_dev, y_test = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)
#     assert (len(X_train) == 30)
#     assert (len(X_test) == 10)
#     assert (len(X_dev) == 60)


def test_post_predict():
    digits_dict = read_digits()
    for digit, data in digits_dict.items():
        image_data = data.flatten().tolist()
        response = app.test_client().post("/predict", json={'image': image_data})
        # print(response.data)
        assert response.status_code == 200
        response_json = response.get_json()
        predicted_digit = response_json['predicted_digit']
        assert predicted_digit == digit
