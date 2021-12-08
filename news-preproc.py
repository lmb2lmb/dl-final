import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

data = tfds.load('ag_news_subset', split=['train', 'test'])[0]

for example in data:
    sentence = example['description']
    print(sentence)

