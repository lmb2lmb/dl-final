import tensorflow as tf
import encoder
import decoder

class VAE(tf.keras.Model):
    def __init__(self, window_size, vocab_size, latent_size):
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.batch_size = 100
        self.hidden_dim = None
        self.latent_size = latent_size


        self.transformer =  encoder.Transformer(self.window_size, self.vocab_size)

        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.Dense(self.hidden_dim, activation = 'relu'))
        self.encoder.add(tf.keras.layers.Dense(self.hidden_dim, activation = 'relu'))
        self.encoder.add(tf.keras.layers.Dense(self.hidden_dim, activation = 'relu'))

        self.mu_layer = tf.keras.Sequential()
        self.mu_layer.add(tf.keras.layers.Dense(latent_size))

        self.lagvar_layer = tf.keras.Sequential()
        self.logvar_layer.add(tf.keras.layers.Dense(latent_size))

        self.decoder = decoder.Decoder(self.vocab_size, self.latent_size)



    def call(self, inputs, inputs_forcing):
        transformer_output = self.transformer.call(inputs)
        transformer_output = tf.reshape(transformer_output, [tf.shape(transformer_output)[0], -1])

        decoder_output = self.encoder(transformer_output)

        mu = self.mu_layer(decoder_output)

        logvar = self.logvar_layer(decoder_output)

        latent_sample = reparametrize(mu, logvar)

        reconstructed_sentances = self.decoder.call(inputs_forcing, latent_size)

        return reconstructed_sentances, mu, logvar




def reparametrize(mu, logvar):
    """
    Differentiably sample random Gaussian data with specified mean and variance using the
    reparameterization trick.

    Suppose we want to sample a random number z from a Gaussian distribution with mean mu and
    standard deviation sigma, such that we can backpropagate from the z back to mu and sigma.
    We can achieve this by first sampling a random value epsilon from a standard Gaussian
    distribution with zero mean and unit variance, then setting z = sigma * epsilon + mu.

    For more stable training when integrating this function into a neural network, it helps to
    pass this function the log of the variance of the distribution from which to sample, rather
    than specifying the standard deviation directly.

    Inputs:
    - mu: Tensor of shape (N, Z) giving means
    - logvar: Tensor of shape (N, Z) giving log-variances

    Returns:
    - z: Estimated latent vectors, where z[i, j] is a random value sampled from a Gaussian with
         mean mu[i, j] and log-variance logvar[i, j].
    """
    z = None

    var = tf.exp(logvar)
    sigma = tf.sqrt(var)

    epsilon = tf.random.normal(tf.shape(sigma))

    z = sigma * epsilon + mu

    return z
