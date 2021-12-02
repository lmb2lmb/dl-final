import numpy as np
import re

# # should be a lowercase list of banned words
# banned_list = []
# with open("badwords.txt", "r") as f:
#     banned_list = f.read().splitlines()

# jokes = pd.read_csv('shortjokes.csv')['Joke'].tolist()

# filtered = []
# for i, raw_joke in enumerate(jokes):
#     if i % 10000 == 0:
#         print(i)
#     # check if any innaproriate words are contained within the joke
#     bad = False
#     for t in banned_list:
#         if t in raw_joke.lower():
#             bad=True
#     if len(raw_joke) < 400 and not bad:
#         filtered.append(raw_joke)

# with open("filtered_jokes.txt", "w") as file:
#     for j in filtered:
#         file.write(j + "\n")

## strip punctuation
## save ?, ..., *, -, :, ", ., ! as separate words
def proc():
    split_jokes = []
    with open("filtered_jokes.txt", "r") as file:
        the_jokes = file.read().splitlines()
        for j in the_jokes:
            tokenized = re.findall("\w+\'+\w{1,2}|\"|\?|\:|\.{3}|-|\*|\.|\!|,|\w+", j.lower())
            split_jokes.append(tokenized)

    return split_jokes

## alternatively
# split_jokes = []
# with open("filtered_jokes.txt", "r") as file:
#     the_jokes = file.read().splitlines()
#     for j in the_jokes:
#         split_jokes.append(word_tokenize(j))
#         print(word_tokenize(j))

# returns corpus (dict) and id of padding token
def make_corpus(jokes):
    corpus = {}
    i = 0
    for j in jokes:
        for word in j:
            if word not in corpus:
                corpus[word] = i
                i += 1

    # padding token is i
    corpus['*PAD*'] = i
    corpus['*START*'] = i+1
    corpus['*STOP*'] = i+2

    return corpus, i

def encode(jokes, corpus):
    encoded_jokes = []
    max_len = max(list(map(len, jokes)))
    for j in jokes:
        encoded = [corpus['*START*']]
        i = 0
        for word in j:
            encoded.append(corpus[word])
            i += 1
        encoded.append(corpus['*STOP*'])
        while i < max_len + 2:
            encoded.append(corpus['*PAD*'])
            i += 1 
        encoded_jokes.append(encoded)

    return np.array(encoded_jokes)

def preprocess():
    split = proc()
    print('done split')
    corpus, pad_token = make_corpus(split)
    print('made corpus')
    encoded_jokes = encode(split, corpus)
    print('encoded')
    return encoded_jokes, corpus, pad_token






    

