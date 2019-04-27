# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:21:09 2019

@author: Mahmoud Taha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from sklearn.metrics import classification_reportو precision_score, recall_score, f1_score
#from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

""" Load and prepare the Dataset """
data = pd.read_csv("ner_dataset.csv", encoding="latin1")
#data = data[:100000]
data = data.fillna(method="ffill")

# data.tail(10)

"""  Extracting the "word" and "tag" column from the dataset into two lists  """
words_col = list(set(data["Word"].values)) 
words_col.append("ENDPAD")
n_words = len(words_col); n_words
tags_col= list(set(data["Tag"].values)) 
n_tags = len(tags_col); n_tags

class SentenceGetter(object):
    def __init__ (self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                               s["POS"].values.tolist(),
                               s["Tag"].values.tolist())]
    
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
    def get_next_sentence (self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
        
""" Applying our dataset to getter """        
getter = SentenceGetter(data) 

sentences = getter.sentences

""" Visualizing how long the sentences are """
plt.style.use("ggplot")
plt.hist([len(s) for s in sentences], bins=50)
plt.show()


""" Equaling the sentences' lenghts inputs """
max_length = 50

""" Transforming the words and tags lists into dicts """
word2idx = {w: i for i, w in enumerate(words_col)}
tag2idx = {t: i for i, t in enumerate(tags_col)}

""" Pad sequencing for Words """
X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_length, sequences=X, padding="post", value=n_words - 1)

""" Pad sequencing for Tags """
y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_length, sequences=y, padding="post", value=tag2idx["O"])

""" For training the network we also need to change the labels y to categorial. """
y = [to_categorical(i, num_classes=n_tags) for i in y]

""" split in train and test set. """
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)

""" Training the model """
input = Input(shape=(max_length,))
model = Embedding(input_dim=n_words, output_dim=50, input_length=max_length)(input)
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)

model = Model(input, out)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=5, validation_split=0.1, verbose=1)


hist = pd.DataFrame(history.history)
plt.figure(figsize=(12,12))
plt.plot(hist["acc"])
plt.plot(hist["val_acc"])
plt.show()

""" Predictions """ 
i = 2318
p = model.predict(np.array([X_te[i]]))
p = np.argmax(p, axis=-1)
print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
for w, pred in zip(X_te[i], p[0]):
    print("{:15}: {}".format(words_col[w], tags_col[pred]))


# Evaluation 

test_pred = model.predict(X_te, verbose=1)

idx2tag = {i: w for w, i in tag2idx.items()}

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out
    
pred_labels = pred2label(test_pred)
test_labels = pred2label(y_te)

"""output = []
def reemovNestings (l):
    for i in l:
        if type(i) == list:
            reemovNestings(i)
        else:
            output.append(i)"""
            

#pred_labels_nestFree = reemovNestings(pred_labels)
#test_labels_nestFree = reemovNestings(test_labels)
#pred_labels_np = np.array(pred_labels)
#test_labels_np = np.array(test_labels)

print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
print(classification_report(test_labels, pred_labels))

