from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def preprocess_data(data):
    # flatten the images
    n = len(data)
    data = data.reshape((n, -1))
    return data


def get_hyperparameter_combinations(gamma_list, C_list):
    params = {
        'gamma_ranges': gamma_list,
        'C_ranges': C_list
    }
    param_combinations = [{'gamma': gamma, 'C': C} for gamma in params['gamma_ranges'] for C in params['C_ranges']]
    return param_combinations


def get_data():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y


def train_model(x, y, model_params):
    gamma = model_params['gamma']
    C = model_params['C']
    model = svm.SVC(kernel='rbf', gamma=gamma, C=C)
    model.fit(x, y)
    return model


# Split data into 50% train and 50% test subsets
def split_data(x, y, test_size, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_test_dev_split(X, y, test_size, dev_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    num_dev_samples = int(len(X_train) * dev_size)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=num_dev_samples, shuffle=False)

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted)
    return accuracy


def tune_hparams(X_train, y_train, X_dev, y_dev, param_combinations):
    best_acc_so_far = -1
    best_model = None
    best_hparams = None
    for params in param_combinations:
        # Train model with current hyperparameters
        cur_model = train_model(X_train, y_train, params)

        # Evaluate the model on the development set
        cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev)

        # Select the hyperparameters that yield the best performance on DEV set
        if cur_accuracy > best_acc_so_far:
            best_acc_so_far = cur_accuracy
            best_model = cur_model
            best_hparams = params

    return best_hparams, best_model, best_acc_so_far
