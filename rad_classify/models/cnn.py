from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Flatten

class CNN():
    def __init__(
        self,
        vocab_size,
        filter_sizes=[200],
        conv_dropouts=[0.2], 
        strides=[1],
        hidden_sizes=[250],
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
            self.model.add(GlobalMaxPooling1D())

        self.model.add(Flatten())

        for i in range(len(hidden_sizes)):
            self.model.add(Dense(hidden_sizes[i]))
            self.model.add(Dropout(hidden_dropouts[i]))
            self.model.add(Activation('relu'))
        
        self.model.add(Dense(2))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    def train(self, data, labels, batch_size=64, epochs=10):
        padded_data = sequence.pad_sequences(data, maxlen=self.maxlen)
        self.model.fit(padded_data, labels, batch_size=batch_size, epochs=epochs)

    def predict(self, report):
        padded_data = sequence.pad_sequences(report, maxlen=self.maxlen)
        return self.predict(padded_data)

    def save_model(self, path):
        raise NotImplementedError
