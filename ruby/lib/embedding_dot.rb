# frozen_string_literal: true

require 'numo/narray'
require 'embedding'

class EmbeddingDot
  def initialize(w)
    @embed = Embedding.new(w)
    @params = @embed.params
    @grads = @embed.grads
    @cache = nil
  end

  def forward(h, idx)
    target_w = @embed.forward(idx)
    out = (target_w * h).sum(axis: 1)

    @cache = [h, target_w]
    out
  end

  def backward(dout)
    h, target_w = @cache
    dout = dout.reshape(dout.shape[0], 1) # transformで良いのでは？

    dtarget_w = dout * h
    @embed.backward(dtarget_w)
    dh = dout * target_w
    dh
  end
end
