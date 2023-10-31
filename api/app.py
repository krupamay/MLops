from flask import Flask, request

app = Flask(__name__)


@app.route("/hello/<val>")
def hello_world(val):
    return "<p>Hello, World!</p>" + val


@app.route("/sum/<x>/<y>")
def sum(x, y):
    return str(int(x) + int(y))


@app.route("/model/predict", methods=['POST'])
def predict_model():
    request_data = request.get_json()
    x = request_data['x']
    y = request_data['y']
    return str(int(x) + int(y))


if __name__ == "__main__":
    app.run(debug=True)
