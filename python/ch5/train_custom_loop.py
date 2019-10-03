import sys 
sys.path.append('..')
import matplotlib.pypot as plt 
import numpy as np 
from common.optimizer import SGD 
from dataset import ptb 
from simple_rnnlm import SimpleRnnlm 

batch_size = 10 
wordvec_size = 100 
hidden_size = 100 
time_size = 5 
lr = 0.1 
max_epoch = 100 

corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1]
ts = corpus[1:]
data_size = len(xs)
print('corpus size: %d, vocabulary size: %d' %(corpus_size, vocab_size))

max_iters = data_size // (batch_size * time_size)
time_idx = 0
total_loss = 0
loss_count = 0 
ppl_list = []

model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

jump = (corpus_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)]

for epoch in range(max_epoch):
    for iter_ in range(max_iters):

