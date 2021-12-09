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

		#self.positional_encoding = transformer.Position_Encoding_Layer(self.window_size, self.embedding_size)
		self.decoder = transformer.Transformer_Block(self.embedding_size, is_decoder=False)

	@tf.function
	def call(self, embedded_encoder_input, embedded_decoder_input):
	
		decoder_output = self.decoder(embedded_encoder_input, context=embedded_decoder_input)
		return decoder_output

	def __call__(self, *args, **kwargs):
		return super(Transformer, self).__call__(*args, **kwargs)