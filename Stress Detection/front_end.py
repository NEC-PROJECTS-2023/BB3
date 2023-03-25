from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
cv = CountVectorizer()

data = pd.read_csv('cleaned_dataset.csv')
x = np.array(data["text"])
cv = CountVectorizer()
X = cv.fit_transform(x)


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    user = request.form.get('text')
    d = cv.transform([user]).toarray()
    pre = str((model.predict(d))[0])
    if pre == 'No Stress':
        prediction = "positive"
    else:
        prediction = 'Negative'

    return render_template('predict.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
