import tensorflow as tf
import encoder
import decoder
import preprocessing
import VAE
import numpy as np

def train(model, sentences, padding_index):
    optimizer = tf.keras.optimizers.Adam()

    batch_size = model.batch_size
    sentences_to_encode = sentences[::, 1:]
    sentances_for_teacher_forcing = sentences[::, :-1]
    mask = tf.not_equal(sentences_to_encode, padding_index)
    mask = tf.cast(mask, tf.float32)


    total_batches = tf.shape(sentences)[0] // batch_size

    for i in range(total_batches):
        batch_sentances = sentences_to_encode[i*batch_size: (i+1)*batch_size]
        batch_sentances_forcing = sentances_for_teacher_forcing[i*batch_size: (i+1)*batch_size]
        batch_mask = mask[i*batch_size: (i+1)*batch_size]

        with tf.GradientTape() as tape:
            probs, mu, logvar = model.call(batch_sentances, batch_sentances_forcing)
            loss = loss_function(probs, mu, logvar, batch_sentances, batch_mask)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))



def reconstruction_loss(probs, sentances, mask):
    # Returns the average reconstruction over batch
    total_labels = tf.count_nonzero(mask)
    loss = tf.metrics.sparse_categorical_crossentropy(tf.cast(sentances, tf.float32), probs)
    loss = loss * mask
    loss = tf.reduce_sum(loss) / total_labels


def kl_div(mu, logvar):
    #Returns average KL divergence over batch
    KL = (1 + logvar - (mu*mu) - tf.exp(logvar))*(-0.5)
    KL = tf.reduce_sum(KL, 1)
    KL = tf.reduce_mean(KL)

def loss_function(probs, mu, logvar, labels, mask):
    #average loss
    recon_loss = reconstruction_loss(probs, labels, mask)
    KL = kl_div(mu, logvar)
    loss = recon_loss + KL
    return loss

def generate_sentences(model, word_to_index_dict, num_sentences):
    decoder = model.decoder
    size = model.hidden_size
    input = tf.random.normal(size)
    sequence = [word_to_index_dict["<START>"]]

    return ""

def main():
    print('about to preprocess')
    data, corpus, pad_token = preprocessing.preprocess()
    print('done preproc')
    num_sentences, len_sentence = np.shape(data)
    m = VAE.VAE(len_sentence - 1, num_sentences, 32)
    train(m, data, pad_token)
    print(generate_sentences(m, corpus, 5))

main()