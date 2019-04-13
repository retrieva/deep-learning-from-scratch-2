require "numo/narray"

class SoftmaxWithLoss
  def initialize
    @y = nil  # softmaxの出力
    @t = nil  # 教師ラベル
  end

  def forward(x, t)
    @t = t
    @y = softmax(x)
    
    return cross_entropy_error(@y, @t)
  end

  def backward(dout: 1)
    t = @t
    y = @y
    
    if y.ndim == 1
      t = t.reshape(1, t.size)
      y = y.reshape(1, y.size)
    end

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size
      t = t.max_index(axis: 1)
    end

    dx = y.copy()
    dx = dx[t]
    dx -= 1
    
    dx *= dout
    dx = dx / t.size
    
    return dx
  end

  def softmax(x)
    if x.ndim == 2
      x = x - x.max(axis: 1, keepdims: true)
      x = Numo::NMath.exp(x)
      x /= x.sum(axis: 1, keepdims: true)
    elsif x.ndim == 1
      x = x - x.max
      x = Numo::NMath.exp(x)
      x /= x.sum
    end

    return x
  end
  
  def cross_entropy_error(y, t)
    if y.ndim == 1
      t = t.reshape(1, t.size)
      y = y.reshape(1, y.size)
    end
    
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size
      t = t.max_index(axis: 1)
    end

    return -1 * Numo::NMath.log(y[t] + 1e-7).sum / t.size
  end
  
  def params
    [@t, @y]
  end
end
