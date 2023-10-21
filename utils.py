from sklearn import datasets, svm, metrics, tree
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.tree import DecisionTreeClassifier
from itertools import product


def preprocess_data(data):
    # flatten the images
    n = len(data)
    data = data.reshape((n, -1))
    return data


def get_hyperparameter_combinations(list_of_param, param_names):
    list_of_param_combination = []
    for each in list(product(*list_of_param)):
        comb = {}
        for i in range(len(list_of_param)):
            comb[param_names[i]] = each[i]
        list_of_param_combination.append(comb)
    return list_of_param_combination


def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y


def train_model(X_train, y_train, model_params, model_type):
    if model_type == 'svm':
        clf = svm.SVC
    if model_type == 'tree':
        clf = tree.DecisionTreeClassifier
    model = clf(**model_params)
    # print(model_params)
    model.fit(X_train, y_train)
    return model


def train_model_decision_tree(x, y, model_params):
    max_depth = model_params.get('max_depth', None)
    min_samples_split = model_params.get('min_samples_split', 2)
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
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


def tune_hparams(X_train, y_train, X_dev, y_dev, param_combinations, model_type):
    best_accuracy = -1
    best_model_path = ""
    best_model = None
    best_hparams = None
    for params in param_combinations:
        # Train model with current hyperparameters
        cur_model = train_model(X_train, y_train, model_params=params, model_type=model_type)
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
    return best_hparams, best_model_path, best_accuracy, best_model
