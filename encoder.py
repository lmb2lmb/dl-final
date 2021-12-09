import numpy as np
import tensorflow as tf
import transformer_funcs_copy as transformer

class Transformer(tf.keras.Model):
	def __init__(self, window_size, vocab_size, embedding_size):

		super(Transformer, self).__init__()

		self.vocab_size = vocab_size
		self.window_size = window_size
		self.embedding_size = embedding_size
		self.batch_size = 100

		self.positional_encoding = transformer.Position_Encoding_Layer(self.window_size, self.embedding_size)
		self.encoder = transformer.Transformer_Block(self.embedding_size, is_decoder=False)

	@tf.function
	def call(self, encoder_input):
	
		embeddings = self.positional_encoding(encoder_input)
		encoder_output = self.encoder(embeddings)
		return encoder_output

	def __call__(self, *args, **kwargs):
		return super(Transformer, self).__call__(*args, **kwargs)