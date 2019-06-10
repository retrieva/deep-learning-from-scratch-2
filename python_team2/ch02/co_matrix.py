# -*- coding: utf-8

import sys
sys.path.append('..')
import numpy as np
from common.util import preprocess, create_co_matrix

text = 'You say goodbye and I say hello.'
print(text)

corpus, word_to_id, id_to_word = preprocess(text)

print(corpus)
print(id_to_word)

C = create_co_matrix(corpus, len(id_to_word))
print(C)

print(id_to_word[0])
print(C[0])

print('goodbye')
print(C[word_to_id['goodbye']])
print('say')
print(C[word_to_id['say']])
