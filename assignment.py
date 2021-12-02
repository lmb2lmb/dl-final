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
        print('batch' + str(i))
        batch_sentances = sentences_to_encode[i*batch_size: (i+1)*batch_size]
        batch_sentances_forcing = sentances_for_teacher_forcing[i*batch_size: (i+1)*batch_size]
        batch_mask = mask[i*batch_size: (i+1)*batch_size]

        with tf.GradientTape() as tape:
            probs, mu, logvar = model(batch_sentances, batch_sentances_forcing)
            loss = loss_function(probs, mu, logvar, batch_sentances, batch_mask)

        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

def reconstruction_loss(probs, sentances, mask):
    # Returns the average reconstruction over batch
    total_labels = tf.math.count_nonzero(mask)
    loss = tf.metrics.sparse_categorical_crossentropy(tf.cast(sentances, tf.float32), probs)
    loss = loss * mask
    loss = tf.reduce_sum(loss) / tf.cast(total_labels, tf.float32)
    return loss


def kl_div(mu, logvar):
    #Returns average KL divergence over batch
    KL = (1 + logvar - (mu*mu) - tf.exp(logvar))*(-0.5)
    KL = tf.reduce_sum(KL, 1)
    KL = tf.reduce_mean(KL)
    return KL

def loss_function(probs, mu, logvar, labels, mask):
    #average loss
    recon_loss = reconstruction_loss(probs, labels, mask)
    print('recon')
    print(recon_loss)
    KL = kl_div(mu, logvar)
    print('KL')
    print(KL)
    loss = recon_loss + KL
    return loss

def generate_sentences(model, word_to_index_dict, rev_word_to_index_dict, num_sentences):
    decoder = model.decoder
    size = model.latent_size
    the_sentences = []

    for j in range(num_sentences):
        input = tf.random.normal([1,size])
        sequence = [word_to_index_dict["*START*"]]
        i = 0
        while not(sequence[-1] == word_to_index_dict["*STOP*"]) and i < 15:
            input_seq = tf.reshape(sequence, [1,-1])
            input_seq = tf.nn.embedding_lookup(model.E, input_seq)
            _, final_state = decoder.gru1(input_seq, initial_state = input)
            probs = decoder.dense(final_state)
            probs = tf.reshape(probs, [-1])
            next_word = tf.argsort(probs)[-1].numpy()
            if next_word == word_to_index_dict['*STOP*']:
                next_word = tf.argsort(probs)[-2].numpy()
            sequence.append(next_word)
            i += 1

        for index, value in enumerate(sequence):
            sequence[index] = rev_word_to_index_dict[value]

        the_sentences.append(sequence)

    return the_sentences


def main():
    print('about to preprocess')
    data, corpus, pad_token, rev_corpus = preprocessing.preprocess()
    print('done preproc')
    num_sentences, len_sentence = np.shape(data)
    m = VAE.VAE(len_sentence - 1, num_sentences, 128)
    train(m, data, pad_token)
    print(generate_sentences(m, corpus, rev_corpus, 5))

main()