import tensorflow as tf

class MinLSTMCell(tf.keras.layers.Layer):
    def __init__(self, units, inp_shape):
        super(MinLSTMCell, self).__init__()
        self.units = units
        self.inp_shape = inp_shape

        # Initialize the linear layers for the forget gate, input gate, and hidden state transformation
        self.linear_f = self.add_weight(name="linear_f", shape=(self.inp_shape, self.units))
        self.linear_i = self.add_weight(name="linear_i", shape=(self.inp_shape, self.units))
        self.linear_h = self.add_weight(name="linear_h", shape=(self.inp_shape, self.units))

    def call(self, pre_h, x_t):
        """
        x_t: (batch_size, input_size) - input at time step t
        pre_h: (batch_size, units) - previous hidden state (h_prev)
        """

        # Forget gate: f_t = sigmoid(W_f * x_t)
        f_t = tf.nn.sigmoid(tf.matmul(x_t, self.linear_f))  # (batch_size, units)

        # Input gate: i_t = sigmoid(W_i * x_t)
        i_t = tf.nn.sigmoid(tf.matmul(x_t, self.linear_i))  # (batch_size, units)

        # Hidden state: tilde_h_t = W_h * x_t
        tilde_h_t = tf.matmul(x_t, self.linear_h)  # (batch_size, units)

        # Normalize the gates
        sum_f_i = f_t + i_t
        f_prime_t = f_t / sum_f_i  # (batch_size, units)
        i_prime_t = i_t / sum_f_i  # (batch_size, units)

        # New hidden state: h_t = f_prime_t * pre_h + i_prime_t * tilde_h_t
        h_t = f_prime_t * pre_h + i_prime_t * tilde_h_t  # (batch_size, units)

        return h_t  # (batch_size, units)
