import tensorflow as tf
import preprocessing
import VAE
import numpy as np
from news_preproc import news_preprocess

def train(model, sentences, padding_index, num_epochs):
    optimizer = tf.keras.optimizers.Adam()

    batch_size = model.batch_size
    sentences_to_encode = sentences[::, 1:]
    sentances_for_teacher_forcing = sentences[::, :-1]
    mask = tf.not_equal(sentences_to_encode, padding_index)
    mask = tf.cast(mask, tf.float32)

    total_batches = tf.shape(sentences)[0] // batch_size
    for j in range(num_epochs):
        sum_loss = 0
        for i in range(total_batches):
            if i % 100 == 0:
                print(i)
            batch_sentances = sentences_to_encode[i*batch_size: (i+1)*batch_size]
            batch_sentances_forcing = sentances_for_teacher_forcing[i*batch_size: (i+1)*batch_size]
            batch_mask = mask[i*batch_size: (i+1)*batch_size]

            with tf.GradientTape() as tape:
                probs, mu, logvar = model(batch_sentances, batch_sentances_forcing)
                loss = loss_function(probs, mu, logvar, batch_sentances, batch_mask)
            
            sum_loss += loss
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        print('Epoch ' + str(j+1) + ' average loss: ' + str(sum_loss.numpy() / total_batches))

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
    KL = kl_div(mu, logvar)
    loss = recon_loss + KL
    return loss

def generate_sentences(model, word_to_index_dict, rev_word_to_index_dict, num_sentences, length_cutoff=20, unk=False):
    size = model.latent_size
    the_sentences = []

    for _ in range(num_sentences):
        input = tf.random.normal([1,size])
        sequence = joke_from_vector(
            vector=input, 
            model=model, 
            word_to_index_dict=word_to_index_dict, 
            rev_word_to_index_dict=rev_word_to_index_dict, 
            length_cutoff=length_cutoff,
            unk=unk)

        the_sentences.append(sequence)

    return the_sentences

def joke_from_vector(vector, model, word_to_index_dict, rev_word_to_index_dict, length_cutoff, unk):
    decoder = model.decoder
    sequence = [word_to_index_dict["*START*"]]
    i = 0
    while not(sequence[-1] == word_to_index_dict["*STOP*"]) and i < length_cutoff:
        input_seq = tf.reshape(sequence, [1,-1])
        input_seq = tf.nn.embedding_lookup(model.E, input_seq)

        input_reshape = tf.reshape(vector, [1,1,-1])
        input_reshape = tf.tile(input_reshape, [1,tf.shape(input_seq)[1],1])

        input_seq = tf.concat([input_seq, input_reshape], axis = -1)
        _, final_state, _ = decoder.lstm(input_seq)
        probs = decoder.dense(final_state)
        probs = tf.reshape(probs, [-1])
        final_three = tf.argsort(probs)[-3:].numpy()
        next_word = final_three[-1]
        if unk:
            unk_token = word_to_index_dict['*UNK*']
            stop_token = word_to_index_dict['*STOP*']
            if final_three[-1] != unk_token and final_three[-1] != stop_token:
                next_word = final_three[-1]
            elif final_three[-2] != unk_token and final_three[-2] != stop_token:
                next_word = final_three[-2]
            else:
                next_word = final_three[-3]

        elif next_word == word_to_index_dict['*STOP*']:
            next_word = tf.argsort(probs)[-2].numpy()
        sequence.append(next_word)
        i += 1

    for index, value in enumerate(sequence):
        sequence[index] = rev_word_to_index_dict[value]

    return sequence

def make_coherent_sentence(sequence):
    string_sentence = ""
    for token in sequence:
        if token != '*START*' and token != '*STOP*' and token != '*PAD*' and token != '*UNK*':
            string_sentence += token
            string_sentence += ' '

    return string_sentence

def main(epochs=1, length_cutoff=12, unk=False, unk_cutoff=1, sentences_to_generate=10, latent_size=128, use_news=False, news_len_cutoff=100):
    embeddings = None
    corpus = None
    rev_corpus = None
    pad_token = None
    data = None

    if use_news:
        news, news_corpus, news_pad_token,news_rev_corpus = news_preprocess(unk=unk, unk_cutoff=unk_cutoff, length_cutoff=news_len_cutoff)
        data, corpus, pad_token, rev_corpus = preprocessing.preprocess(unk=unk, unk_cutoff=unk_cutoff, length_cutoff=length_cutoff, pre_corpus=news_corpus, pre_rev_corpus=news_rev_corpus)
        _, len_sentence = np.shape(news)
        news_m = VAE.VAE(len_sentence - 1, len(corpus), latent_size=128, hidden_dim=128)
        train(news_m, news, news_pad_token, 1)
        embeddings = news_m.E
    else:
        data, corpus, pad_token, rev_corpus = preprocessing.preprocess(unk=unk, unk_cutoff=unk_cutoff, length_cutoff=length_cutoff)

    _, len_sentence = np.shape(data)
    m = VAE.VAE(len_sentence - 1, len(corpus), latent_size, has_preloaded=True, preloaded_embeddings=embeddings)
    train(m, data, pad_token, epochs)

    # ones = tf.ones([1, latent_size])
    # print(joke_from_vector(ones, m, corpus, rev_corpus, length_cutoff, unk))

    # zeros = tf.zeros([1, latent_size])
    # print(joke_from_vector(zeros, m, corpus, rev_corpus, length_cutoff, unk))
    sentences = generate_sentences(m, corpus, rev_corpus, sentences_to_generate, length_cutoff=length_cutoff, unk=unk)
    for s in sentences:
        print(make_coherent_sentence(s))

# PARAMETERS

# number of training epochs
EPOCHS = 15

# max length of input joke (tokens)
LENGTH_CUTOFF = 15

# whether to mark uncommon words as unknown
UNK = True

# number of occurences in the data at or under which a word will be marked as unknown 
# if UNK = True
UNK_CUTOFF = 5

# number of output sentences to generate
SENTENCES_TO_GENERATE = 50

# latent vector size
LATENT_SIZE = 150

#whether to pretrain embeddings with news dataset
USE_NEWS = True

#max length of articles in words for news
NEWS_CUTOFF = 20

main(epochs=EPOCHS, 
    length_cutoff=LENGTH_CUTOFF, 
    unk=UNK, 
    unk_cutoff=UNK_CUTOFF, 
    sentences_to_generate=SENTENCES_TO_GENERATE, 
    latent_size=LATENT_SIZE,
    use_news=USE_NEWS, 
    news_len_cutoff=NEWS_CUTOFF)


