import numpy as np
import re
import pandas as pd
from tensorflow.python.keras.backend import shape
from tensorflow.python.ops.math_ops import count_nonzero

def remove_banned():
    banned_list = []
    with open("badwords.txt", "r") as f:
        banned_list = f.read().splitlines()

    jokes = pd.read_csv('shortjokes.csv')['Joke'].tolist()

    filtered = []
    for i, raw_joke in enumerate(jokes):
        if i % 10000 == 0:
            print(i)
        # check if any innaproriate words are contained within the joke
        bad = False
        for t in banned_list:
            if t in raw_joke.lower():
                bad=True
        if len(raw_joke) < 400 and not bad:
            filtered.append(raw_joke)


    with open("filtered_jokes.txt", "w") as file:
        for j in filtered:
            file.write(j + "\n")

def compute_basic_similarity(j1, j2):
    w_count = 0
    for word in j1:
        if word in j2:
            w_count += 1

    return w_count / len(j1) + w_count / len(j2)

def remove_duplicates(split_jokes, cutoff=1.2):
    bad_indices = []

    for i, j1 in enumerate(split_jokes):
        for k, j2 in enumerate(split_jokes[i+1:]):
            if compute_basic_similarity(j1, j2) > cutoff:
                bad_indices.append(k)

    good_indices = list(range(len(split_jokes)) - set(bad_indices))
    new_jokes = np.array(split_jokes)[good_indices]

    return new_jokes, len(set(bad_indices))

## strip punctuation
## save ?, ..., *, -, :, ", ., ! as separate words
def proc(length_cutoff):
    split_jokes = []
    with open("filtered_jokes.txt", "r") as file:
        the_jokes = file.read().splitlines()
        for j in the_jokes:
            tokenized = re.findall("\w+\'+\w{1,2}|\"|\?|\:|\.{3}|-|\*|\.|\!|,|\w+", j.lower())
            if len(tokenized) < length_cutoff:
                split_jokes.append(tokenized)

    print(len(split_jokes))
    return split_jokes

# returns corpus (dict) and id of padding token
def make_corpus(jokes, unk=False, unk_cutoff=1):
    words_count = {}
    if unk:
        for j in jokes:
            for word in j:
                if word not in words_count:
                    words_count[word] = 1
                else:
                    words_count[word] += 1
    corpus = {}
    rev_corpus = {}
    i = 0
    for j in jokes:
        for word in j:
            if word not in corpus:
                if unk:
                    if words_count[word] <= unk_cutoff:
                        continue
                corpus[word] = i
                rev_corpus[i] = word
                i += 1

    # padding token is i
    corpus['*PAD*'] = i
    rev_corpus[i] = '*PAD*'
    corpus['*START*'] = i+1
    rev_corpus[i+1] = '*START*'
    corpus['*STOP*'] = i+2
    rev_corpus[i+2] = '*STOP*'
    if unk:
        corpus['*UNK*'] = i+3
        rev_corpus[i+3] = '*UNK*'

    return corpus, i, rev_corpus

def encode(jokes, corpus, unk=False):
    encoded_jokes = []
    max_len = max(list(map(len, jokes)))
    for j in jokes:
        encoded = [corpus['*START*']]
        i = 0
        for word in j:
            if unk:
                try:
                    encoded.append(corpus[word])
                    i += 1
                except:
                    encoded.append(corpus['*UNK*'])
                    i+=1
            else:
                encoded.append(corpus[word])
                i += 1
        encoded.append(corpus['*STOP*'])
        while i < max_len + 2:
            encoded.append(corpus['*PAD*'])
            i += 1 
        encoded_jokes.append(encoded)

    arr = np.array(encoded_jokes)
    print(np.shape(arr))
    return arr

def preprocess(unk=False, unk_cutoff=1, length_cutoff=12):
    split = proc(length_cutoff)
    no_duplications, num_duplicates = remove_duplicates(split)
    print('duplications: ' + str(num_duplicates))
    print('done split')
    corpus, pad_token, rev_corpus = make_corpus(no_duplications, unk=unk, unk_cutoff=unk_cutoff)
    print('made corpus')
    encoded_jokes = encode(split, corpus, unk=unk)
    print('encoded')
    return encoded_jokes, corpus, pad_token, rev_corpus

def proc_and_clean(dup_cutoff, filename):
    split = proc(length_cutoff=1000)
    no_duplications, num_duplicates = remove_duplicates(split, cutoff=dup_cutoff)
    print('removed duplications: ' + str(num_duplicates))
    rebuilt_jokes = []
    for j in no_duplications:
        built = ""
        for word in j:
            built += word
            built += " "
        rebuilt_jokes.append(built)

    with open(filename, "w") as file:
        for j in rebuilt_jokes:
            file.write(j + "\n")
