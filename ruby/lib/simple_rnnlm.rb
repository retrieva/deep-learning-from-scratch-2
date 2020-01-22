require 'time_embedding'
require 'time_rnn'

class SimpleRnnlm
  def initialize(vocab_size, wordvec_size, hidden_size)
    v, d, h = vocab_size, wordvec_size, hidden_size

    embed_w = Numo::SFloat.new(v, d).rand_norm / 100
    rnn_wx = Numo::SFloat.new(d, h) / Numo::NMath.sqrt(d)
    rnn_wh = Numo::SFloat.new(h, h) / Numo::NMath.sqrt(h)
    rnn_b = Numo::SFloat.zeros(h)
    affine_w = Numo::SFloat.new(h, v) / Numo::NMath.sqrt(h)
    affine_b = Numo::SFloat.zeros(v)

    @layers = [
      TimeEmbedding.new(embed_w),
      TimeRNN.new(rnn_wx, rnn_wh, rnn_b, stateful
    ]
  end
