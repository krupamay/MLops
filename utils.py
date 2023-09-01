from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def split_train_dev_test(X, y, test_size, dev_size):
    # Split data into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Calculate the number of samples for the dev set
    num_dev_samples = int(len(X_train) * dev_size)
    
    # Split train subset into train and dev subsets
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=num_dev_samples, shuffle=False)
    
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test[:4], predicted[:4]):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    # Classification report
    print(
        f"Classification report for classifier {model}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    # Confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_test, predicted)
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    plt.show()
    return disp
    