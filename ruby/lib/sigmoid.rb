require 'numo/narray'

class Sigmoid
    attr_accessor :params, :grads, :out
    def initialize
      @params = []
      @grads = []
      @out = nil
    end
  
    def forward(x)
      @out = 1.0 / (1.0 + Numo::NMath.exp(-x))
    end
  
    def backward(dout)
      dout * (1.0 - @out) * @out
    end
  end
end  