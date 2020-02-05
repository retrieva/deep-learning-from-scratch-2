require 'numo/narray'

class TimeAffine
  attr_accessor :params, :grads

  def initialize(w, b) # w: d x v, b: v
    @params = [w, b]
    @grads = [w.new_zeros, b.new_zeros]
  end

  def forward(x)                # x: n x t x d
    n, t, _d = x.shape
    w, b = @params

    rx = x.reshape(n * t, true) # (n*t) x d
    out = rx.dot(w) + b         # (n*t) x v
    @x = x
    out.reshape(n, t, true)     #  n x t x v
  end

  def backward(dout)
    n, t, _v = dout.shape
    w, _b = @params
    rdout = dout.reshape(n * t, true)                 # (n*t) x v

    db = rdout.sum(axis: 0)                           # v
    dw = @x.reshape(n * t, true).transpose.dot(rdout) # d x v
    dx = rdout.dot(w.transpose).reshape(*@x.shape)    # n x t x d

    @grads[0].store dw
    @grads[1].store db
    dx
  end
end
