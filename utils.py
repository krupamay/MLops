from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from joblib import dump, load


def preprocess_data(data):
    # flatten the images
    n = len(data)
    data = data.reshape((n, -1))
    return data


def get_hyperparameter_combinations(params):
    param_combinations = [{'gamma': gamma, 'C': C} for gamma in params['gamma'] for C in params['C']]
    return param_combinations


def read_digits():
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
    X_train_dev, X_test, Y_train_Dev, y_test = split_data(X, y, test_size=test_size, random_state=1)
    # print("train+dev = {} test = {}".format(len(Y_train_Dev), len(y_test)))
    X_train, X_dev, y_train, y_dev = split_data(X_train_dev, Y_train_Dev, dev_size / (1 - test_size), random_state=1)

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted)
    return accuracy


def tune_hparams(X_train, y_train, X_dev, y_dev, param_combinations):
    best_accuracy = -1
    best_model_path = ""
    best_model = None
    best_hparams = None
    for params in param_combinations:
        # Train model with current hyperparameters
        cur_model = train_model(X_train, y_train, params)

        # Evaluate the model on the development set
        cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev)

        # Select the hyperparameters that yield the best performance on DEV set
        if cur_accuracy > best_accuracy:
            best_accuracy = cur_accuracy
            best_model = cur_model
            best_hparams = params
            best_model_path = "./models/best_model" + "_".join(
                ["{}:{}".format(k, v) for k, v in params.items()]) + ".joblib"
        # save the best_model
        dump(best_model, best_model_path)

        # print("Model save at {}".format(best_model_path))
    return best_hparams, best_model_path, best_accuracy
