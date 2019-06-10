# -*- coding: utf-8

import sys
sys.path.append('..')
from common.util import preprocess, create_co_matrix, most_similar

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

most_similar('you', word_to_id, id_to_word, C, top=5)

# [query] you
#  goodbye: 0.7071067691154799
#  i: 0.7071067691154799
#  hello: 0.7071067691154799
#  say: 0.0
#  and: 0.0
