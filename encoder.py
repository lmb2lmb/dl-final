import numpy as np
import tensorflow as tf
import transformer_funcs_copy as transformer

class Transformer(tf.keras.Model):
	def __init__(self, window_size, vocab_size):

		super(Transformer, self).__init__()

		self.vocab_size = vocab_size
		self.window_size = window_size
		self.batch_size = 100
		self.embedding_size = 64

		self.E = tf.Variable(tf.random.normal([self.vocab_size, self.embedding_size], stddev=.1))
		self.positional_encoding = transformer.Position_Encoding_Layer(self.window_size, self.embedding_size)
		self.encoder = transformer.Transformer_Block(self.embedding_size, is_decoder=False)
		self.dense = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

	@tf.function
	def call(self, encoder_input):
		"""
		:param encoder_input: batched ids corresponding to french sentences
		:param decoder_input: batched ids corresponding to english sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""
	
		embeddings = self.positional_encoding(tf.nn.embedding_lookup(self.E, encoder_input))
		encoder_output = self.encoder(embeddings)
		return encoder_output

	def __call__(self, *args, **kwargs):
		return super(Transformer, self).__call__(*args, **kwargs)