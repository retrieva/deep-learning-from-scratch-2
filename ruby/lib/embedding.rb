class Embedding
  attr_reader :params, :grads

  def initialize(w)
    @params = [w]
    @grads = [w.new_zeros]
    @idx = nil
  end

  def forward(idx)
    w = @params.first
    @idx = idx
    w[idx, true]
  end

  def backward(dout)
    dw = @grads.first
    dw.store(0)
    @idx.each_with_index do |word_id, i|
      dw[word_id, true].inplace + dout[i, true]
    end
    nil
  end
end
