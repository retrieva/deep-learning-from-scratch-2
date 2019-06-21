# Stochastic Gradient Descent
class SGD
  def initialize(lr = 0.01)
    @lr = lr
  end

  def update(params, grads)
    params.length.times do |i|
      params[i].inplace - @lr * grads[i]
    end
  end
end
