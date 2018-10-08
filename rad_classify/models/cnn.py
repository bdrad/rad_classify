from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Flatten

import numpy as np

class CNN():
    def __init__(
        self,
        vocab_size,
        filter_sizes=[32],
        conv_dropouts=[0.1],
        strides=[1],
        hidden_sizes=[64],
        hidden_dropouts=[0.2],
        kernel_size=3,
        maxlen=100,
        embedding_dim=50,
        path=None
    ):
        self.maxlen = maxlen
        self.vocab_size = vocab_size

        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, embedding_dim, input_length=maxlen))

        if len(filter_sizes) != len(conv_dropouts) or len(filter_sizes) != len(strides):
            raise ValueError("Filter sizes must have the same length as conv dropouts and strides")

        if len(hidden_sizes) != len(hidden_dropouts):
            raise ValueError("Hidden layer sizes must have the same length as hidden dropouts")

        for i in range(len(filter_sizes)):
            self.model.add(Dropout(conv_dropouts[i]))
            self.model.add(Conv1D(
                filter_sizes[i],
                kernel_size,
                padding='valid',
                activation='relu',
                strides=strides[i]
            ))
            self.model.add(MaxPooling1D())

        for i in range(len(hidden_sizes)):
            self.model.add(Dense(hidden_sizes[i], activation='relu'))
            self.model.add(Dropout(hidden_dropouts[i]))

        self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())

    def document_to_token_sequences(self, doc):
        words = doc.split(" ")
        tokens = [self.tokenizer.word_index[w] if w in self.tokenizer.word_index.keys() else 0 for w in words]
        return np.array(tokens)

    def train(self, data, labels, batch_size=64, epochs=10):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(data)
        sequences = [self.document_to_token_sequences(doc) for doc in data]
        padded_data = sequence.pad_sequences(sequences, maxlen=self.maxlen)
        self.model.fit(padded_data, labels, batch_size=batch_size, epochs=epochs)

    def predict(self, reports):
        sequences = [self.document_to_token_sequences(doc) for doc in reports]
        padded_data = sequence.pad_sequences(sequences, maxlen=self.maxlen)
        return self.model.predict(padded_data)

    def save_model(self, path):
        raise NotImplementedError