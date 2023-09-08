from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def preprocess_data(data):
    # flatten the images
    n = len(data)
    data = data.reshape((n, -1))
    return data

def get_all_hyper_params():
    params = {
        'gamma_ranges' : [0.081, 0.01, 0.1, 1, 10, 108],
        'C_ranges' : [0.1, 1, 2, 5, 10]
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
    model.fit (x, y)
    return model

def split_train_dev_test(X, y, test_size, dev_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    num_dev_samples = int(len(X_train) * dev_size)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=num_dev_samples, shuffle=False)
    
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted)
    return accuracy
    # _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    # for ax, image, prediction in zip(axes, X_test[:4], predicted[:4]):
    #     ax.set_axis_off()
    #     image = image.reshape(8, 8)
    #     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    #     ax.set_title(f"Prediction: {prediction}")

    # # Classification report
    # print(
    #     f"Classification report for classifier {model}:\n"
    #     f"{metrics.classification_report(y_test, predicted)}\n"
    # )
 
    # # Confusion matrix
    # confusion_matrix = metrics.confusion_matrix(y_test, predicted)
    # disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    # disp.figure_.suptitle("Confusion Matrix")
    # print(f"Confusion matrix:\n{disp.confusion_matrix}")
    # plt.show()
    # return disp

def tune_hyper_parameter(X_train, y_train, X_dev, y_dev, param_combinations):
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