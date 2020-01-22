require 'numo/narray'

class TimeAffine
  attr_accessor :params, :grads

  def initialize(w, b)
    @params = [w, b]
    @grads = [w.new_zeros, b.new_zeros]
  end

  def forward(x)
    n, t, _d = x.shape
    w, b = @params

    rx = x.reshape(n * t, true)
    out = rx.dot(w) + b
    @x = x
    out.reshape(n, t, true)
  end

  def backward(dout)
    n, t, _hv = dout.shape
    w, _b = @params
    rdout = dout.reshape(n * t, true)  # rdout: n*t x (h x v)

    db = rdout.sum(axis: 0)   # db: (h x v)
    dw = @x.reshape(n * t, true).transpose.dot(rdout)
    dx = rdout.dot(w.transpose).reshape(*@x.shape)   # w: (h x v)

    @grads[0].store dw
    @grads[1].store db
    dx
  end
end
