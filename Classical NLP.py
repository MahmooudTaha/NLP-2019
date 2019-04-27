import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
dframe = pd.read_csv("ner.csv", encoding="ISO-8859-1", error_bad_lines=False)
dframe.dropna(inplace=True)
dframe = dframe[:5000]
x_df = dframe.drop(['Unnamed: 0', 'sentence_idx', 'tag'], axis=1)

vectorizer = DictVectorizer(sparse=False)
x = vectorizer.fit_transform(x_df.to_dict("records"))
y = dframe.tag.values
all_classes = np.unique(y)
all_classes.shape
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=53)
print(x_train.shape)
print(y_train.shape)

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
nb_classifier = MultinomialNB()

nb_classifier.fit(x_train, y_train)

pred = nb_classifier.predict(x_test)

metrics.accuracy_score(y_test, pred)

import tensorflow as tf
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, 'my_test_model')
