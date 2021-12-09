import numpy as np
import tensorflow as tf
import re
from preprocessing import preprocess, make_corpus, encode, proc_and_clean

import tensorflow_datasets as tfds

def news_preprocess(unk=False, unk_cutoff=1, length_cutoff=100):
    data = tfds.load('ag_news_subset', split=['train', 'test'])[0]
    proc_sentences = []
    for example in data:
        s = str(example['description'].numpy())[2:].replace(" #39;", "'").replace('\\', ' ').replace('quot;', "'")
        tokenized = re.findall("\w+\'+\w{1,2}|\"|\?|\:|\.{3}|-|\*|\.|\!|,|\w+", s.lower())
        if len(tokenized) < length_cutoff:
            proc_sentences.append(tokenized)
    
    corpus, pad_token, rev_corpus = make_corpus(proc_sentences, unk=unk, unk_cutoff=unk_cutoff)
    encoded_news = encode(proc_sentences, corpus, unk=unk)
    return encoded_news, corpus, pad_token, rev_corpus