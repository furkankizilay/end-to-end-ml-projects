"""
1. import library
2. problem desciption / data import
3. eda
4. preprocessing
5. construct RNN
6. trainin RNN
7. evaluate result

"""

# 1. import library

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import imdb
from keras_preprocessing.sequence import pad_sequences # cümlelerin uzunluklarını fixlemek için kullanılır
from keras.models import Sequential
from keras.layers import Embedding # intgerları yoğunluk vektörlerine çevirmek için kullanılır
from keras.layers import SimpleRNN, Dense, Activation

# 2. problem desciption / data import

(X_train, Y_train),(X_test, Y_test) = imdb.load_data(path="imdb.npz",
               num_words=None, # kullanılacak kelime
               skip_top=0, # en çok kullanılan kelimeyi ignore etme
               maxlen=None, # uzunluğu kısaltma
               seed=113,
               start_char=1, # hangi karakterden başlayacağı
               oov_char=2,
               index_from=3)

print("X_train Type: ", type(X_train))
print("Y_train Type: ", type(Y_train))
print("X_train shape:", X_train.shape)
print("Y_train shape: ", Y_train.shape)

# 3. eda

print("Y_train values: ", np.unique(Y_train))
print("Y_test values: ", np.unique(Y_test))

unique, counts = np.unique(Y_train, return_counts = True)
print("Y_train distribution: ", dict(zip(unique,counts)))

unique, counts = np.unique(Y_test, return_counts = True)
print("Y_test distribution: ", dict(zip(unique,counts)))

plt.figure()
sns.countplot(Y_train)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("Y_train")
plt.show()

plt.figure()
sns.countplot(Y_test)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("Y_test")
plt.show()

print(X_train[0])
print(type(X_train[0][0]))
print(len(X_train[0]))

review_len_train = []
review_len_test = []

for i, ii in zip(X_train, X_test):
    review_len_train.append(len(i))
    review_len_test.append(len(ii))

print(review_len_train)
print(review_len_test)

sns.histplot(review_len_train, kde_kws={"alpha":0.3})
sns.histplot(review_len_test, kde_kws={"alpha":0.3})
plt.show() # positive sequance

print("Train mean: ", np.mean(review_len_train))
print("Train median: ", np.median(review_len_train))
print("Train mode: ", stats.mode(review_len_train))

# number of words

word_index = imdb.get_word_index()
print(type(word_index))
print(len(word_index))

# verilen sayıya göre hangi eklime olduğunu anlama
for keys, values in word_index.items():
    if values == 12:
        print(keys)

def whatItSay(index = 24):

    reverse_index = dict([(value, key) for (key, value) in word_index.items()])
    decode_review = " ".join([reverse_index.get(i - 3, "!") for i in X_train[index]])
    print(decode_review)
    print(Y_train[index])
    return decode_review

decoded_review = whatItSay(2)

# 4. preprocesisng

num_words = 15000

(X_train, Y_train),(X_test, Y_test) = imdb.load_data(num_words=num_words)

max_len = 130

print("X_train Type: ", type(X_train))
print("Y_train Type: ", type(Y_train))
print("X_train shape:", X_train.shape)
print("Y_train shape: ", Y_train.shape)

print(X_train[5])

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

print("X_train Type: ", type(X_train))
print("Y_train Type: ", type(Y_train))
print("X_train shape:", X_train.shape)
print("Y_train shape: ", Y_train.shape)

print("X_test Type: ", type(X_test))
print("Y_test Type: ", type(Y_test))
print("X_test shape:", X_test.shape)
print("Y_test shape: ", Y_test.shape)


print(X_train[5])

for i in X_train[:10]:
    print(len(i))

decoded_review = whatItSay(5)

# 5. construct RNN

rnn = Sequential()
rnn.add(Embedding(num_words, 32, input_length=len(X_train[0])))
rnn.add(SimpleRNN(16, input_shape=(num_words, max_len), return_sequences=False, activation="relu"))

rnn.add(Dense(1))
rnn.add(Activation("sigmoid"))

print(rnn.summary())
rnn.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# 6. training RNN

history = rnn.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size=128, verbose=1)

# 7. evaluate result

score = rnn.evaluate(X_test, Y_test)
print("Accuracy: %", score[1]*100)

print(history.history)

plt.figure()
plt.plot(history.history["accuracy"], label = "Train")
plt.plot(history.history["val_accuracy"], label = "Test")
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["loss"], label = "Train")
plt.plot(history.history["val_loss"], label = "Test")
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.show()



