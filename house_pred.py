import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('house.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [eval(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output= np.round(prediction[0], 2)

    return render_template('index.html', prediction_text="House Price is {} Rs.".format(output))

@app.route('/result', methods=['POST'])
def result():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)