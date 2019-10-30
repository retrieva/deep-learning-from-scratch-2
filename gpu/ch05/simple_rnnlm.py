import sys
import cupy as cp
sys.path.append('..')
from common.time_layers import TimeAffine, TimeEmbedding, TimeRNN, TimeSoftmaxWithLoss


class SimpleRnnlm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = cp.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        rnn_Wx = (rn(D, H) / cp.sqrt(D)).astype('f')
        rnn_Wh = (rn(H, H) / cp.sqrt(H)).astype('f')
        rnn_b = cp.zeros(H).astype('f')
        affine_W = (rn(H, V) / cp.sqrt(H)).astype('f')
        affine_b = cp.zeros(V).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)

        loss = self.loss_layer.forward(xs, ts)
        return loss
            
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()
