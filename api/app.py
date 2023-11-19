from flask import Flask, request, jsonify
import numpy as np
from joblib import load

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, World!'


@app.route("/hello/<val>")
def hello_world(val):
    return "<p>Hello, World!</p>" + val


@app.route("/sum/<x>/<y>")
def sum(x, y):
    return str(int(x) + int(y))


@app.route('/predict', methods=['POST'])
def predict():
    # Extract image data from POST request
    try:
        model = load('models/svm_gamma-0.001_C-10.joblib')
        data = request.get_json()
        # print(data)

        image1 = data['image']
        # image2 = data['image2']
        image1 = list(map(float, image1))
        # image2 = list(map(float, image2))
        image1 = np.array(image1).reshape(1, -1)
        # image2 = np.array(image2).reshape(1, -1)
        predicted_digit = model.predict(image1)

        # p2 = model.predict(image2)
        # result = "False"
        # if p1[0] == p2[0]:
        #     print("Same")
        #     result = "True"
        # return result
        # print(predicted_digit)
        return jsonify({'predicted_digit': int(predicted_digit[0])})
    except Exception as e:
        print(e)
        return jsonify({'error': 'An error occurred during prediction.'}), 500


# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0')
