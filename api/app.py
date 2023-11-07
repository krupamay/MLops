from flask import Flask, request
import numpy as np
from joblib import load

app = Flask(__name__)


@app.route("/hello/<val>")
def hello_world(val):
    return "<p>Hello, World!</p>" + val


@app.route("/sum/<x>/<y>")
def sum(x, y):
    return str(int(x) + int(y))


@app.route('/predict', methods=['POST'])
def predict():
    # Extract image data from POST request
    model = load('/Users/krupamayghosal/Documents/IIT Jodhpur Mtech/MLops/updated_code_base/MLops/models/tree_max_depth-5.joblib')
    data = request.get_json()
    print(data)

    image1 = data['image1']
    image2 = data['image2']

    image1 = list(map(float, image1))
    image2 = list(map(float, image2))

    image1 = np.array(image1).reshape(1, -1)
    image2 = np.array(image2).reshape(1, -1)

    p1 = model.predict(image1)
    p2 = model.predict(image2)
    result = "False"
    if p1[0] == p2[0]:
        print("Same")
        result = "True"
    return result


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002)
