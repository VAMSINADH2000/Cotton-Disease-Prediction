
import flask
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import imageio
app = Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        file = request.files['image']
        if not file:
            return render_template('index.html', label="No file")
        img = imageio.imread(file)
        img_arr = np.array(img)
        img_size = 150
        img_arr = np.resize(img_arr, (img_size, img_size, img_arr.shape[2]))
        img_arr = np.array([img_arr])
        classifier = load_model('model.h5')
        prediction = np.argmax(classifier.predict(img_arr), axis=1)
        label = int(prediction[0])
        print(label)
        return render_template('index.html', label=label)


if __name__ == '__main__':
    app.run(debug=True)