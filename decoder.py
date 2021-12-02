import tensorflow as tf


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, latent_size):
        super().__init__()
        self.hidden_size = latent_size
        self.vocab_size = vocab_size
        self.gru1 = tf.keras.layers.GRU(self.hidden_size, activation = 'relu', return_sequences = True, return_state = True)
        self.gru2 = tf.keras.layers.GRU(self.hidden_size, return_sequences = True, return_state = True)

        self.dense_1 = tf.keras.layers.Dense(self.hidden_size, activation = 'relu')
        self.dense_2 = tf.keras.layers.Dense(self.hidden_size, activation = 'relu')
        self.dense_3 = tf.keras.layers.Dense(self.vocab_size, activation = 'softmax')




    @tf.function
    def call(self, encoder_input, decoder_input):
        """
        :param decoder_input: the latent z that comes from the encoder and reparameterization trick
        :param encoder_input: the input sentance word embeddings. Sentance "I like pie" will be "<START> I like"
        :return probs: tensor of shape [batch_size, sentance length, vocab_size]
        """
        whole_seq_output_1, _ = self.gru1(encoder_input, initial_state = decoder_input)
        whole_seq_output_2, _ = self.gru2(whole_seq_output_1)

        dense1 = self.dense_1(whole_seq_output_2)
        dense2 = self.dense_2(dense1)
        probs = self.dense_3(dense2)

        return probs
