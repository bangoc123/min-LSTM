import tensorflow as tf
from minLSTMCell import MinLSTMCell

class MinRNN(tf.keras.Model):
    def __init__(self, units, embedding_size, vocab_size, input_length):
        super(MinRNN, self).__init__()
        self.input_length = input_length
        self.units = units

        # Embedding to convert words into vectors
        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_size,
            input_length=input_length
        )

        # Use the LSTM cell programmed above
        self.lstm = MinLSTMCell(units, embedding_size)

        # Then pass the LSTM's hidden state history through a simple neural network
        self.classfication_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_shape=(units,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, sentence):
        """
        Parameters:
        sentence:
          Type: Tensor
          Description: The sentence
          Shape: (batch_size, input_length)
        out:
          Type: Tensor
          Description: Model prediction output
          Shape: (batch_size, 1)
        """

        batch_size = tf.shape(sentence)[0]

        # Initialize hidden_state
        pre_h = tf.zeros([batch_size, self.units])  # Only need to initialize h

        # Pass the sentence through the Embedding to get the vectors
        # embedded_sentence: (batch_size, input_length, embedding_size)
        embedded_sentence = self.embedding(sentence)

        # Pass the entire sequence through the LSTM + hidden_state
        for i in range(self.input_length):
            # Get the embedding for the word at position i
            word = embedded_sentence[:, i, :]  # (batch_size, embedding_size)
            pre_h = self.lstm(pre_h, word)  # Only update h (hidden state)

        # Use the final hidden_state for prediction
        return self.classfication_model(pre_h)  # Pass the final hidden state into the classification network
