# -*- coding: utf-8

# 2.3 カウントベースの手法

import numpy as np

text = "You say goodbye and I say hello."
if __name__ == '__main__':
    print(text)

text.lower()
text = text.lower()
text.replace(".", " .")
text = text.replace(".", " .")
if __name__ == '__main__':
    print(text)

words = text.split (' ')
if __name__ == '__main__':
    print(words)


word_to_id = {}
id_to_word = {}
for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word


if __name__ == '__main__':
    print(word_to_id)
    print(id_to_word)


corpus = [word_to_id[w] for w in words]
corpus = np.array(corpus)
if __name__ == '__main__':
    print(corpus)
