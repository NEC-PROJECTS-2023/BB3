import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import pickle

data = pd.read_csv("cleaned_dataset.csv")

x = np.array(data["text"])
y = np.array(data["label"])

cv = CountVectorizer()
X = cv.fit_transform(x)
x_train, xtest, y_train, ytest = train_test_split(X, y, test_size=0.33, random_state=42)

classifier = BernoulliNB()

classifier.fit(x_train, y_train)

pickle.dump(classifier, open("model.pkl", "wb"))
