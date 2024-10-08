import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from minRNN import MinRNN


with open("./data/sarcasm.json", 'r') as f:
    datastore = json.load(f)

dataset = []
label_dataset = []

for item in datastore:
    dataset.append(item["headline"])
    label_dataset.append(item["is_sarcastic"])


dataset = np.array(dataset)
label_dataset = np.array(label_dataset)

train_size = 0.8
size = int(len(dataset) * train_size)

train_sentence = dataset[:size]
test_sentence = dataset[size:]

train_label = label_dataset[:size]
test_label = label_dataset[size:]


vocab_size = len(train_sentence)
embedding_size = 64
max_length = 25

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentence)

train_sequences = tokenizer.texts_to_sequences(train_sentence)
test_sequences = tokenizer.texts_to_sequences(test_sentence)

padded_train_sequences = pad_sequences(train_sequences, maxlen=max_length, truncating="post", padding="post")
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_length, truncating="post", padding="post")


units = 128
embedding_size = 100
vocab_size = len(tokenizer.index_word) + 1
input_length = max_length

protonxrnn = MinRNN(units, embedding_size, vocab_size, input_length)


protonxrnn.compile(
    tf.keras.optimizers.Adam(0.001) , loss='binary_crossentropy', metrics=['acc']
)

print('---- Training ----')
protonxrnn.fit(padded_train_sequences, train_label, validation_data=(padded_test_sequences, test_label) ,batch_size=32, epochs=100)