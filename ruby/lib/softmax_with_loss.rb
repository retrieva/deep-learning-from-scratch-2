require "numo/narray"

class SoftmaxWithLoss
  def initialize
    @y = nil  # softmaxの出力
    @t = nil  # 教師ラベル
  end

  def forward(x, t)
    @t = t
    @y = softmax(x)

    # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
    if @t.size == @y.size
      @t = @t.max_index(axis: 1)
    end
    
    return cross_entropy_error(@y, @t)
  end

  def backward(dout: 1)
    batch_size = @t.shape[0]
    
    dx = @y.copy()
    dx[Numo::SFloat.new(batch_size).seq, @t] -= 1
    
    dx *= dout
    dx = dx / batch_size
    
    return dx
  end

  def softmax(x)
    if x.ndim == 2
      x = x - x.max(axis: 1, keepdims: true)
      x = Numo::NMath.exp(x)
      x /= x.sum(axis: 1, keepdims: true)
    elsif x.ndim == 1
      x = x - x.max
      x = Numo::NMath.exp(x) / Numo::NMath.exp(x).sum
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

    batch_size = y.shape[0]
    
    e1 = y.slice(Numo::SFloat.new(batch_size).seq, t)
    e2 = Numo::NMath.log(e1 + 1e-7)

    return -1 * e2.sum / batch_size
  end
  
  def params
    [@t, @y]
  end
end
