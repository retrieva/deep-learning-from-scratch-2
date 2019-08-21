# frozen_string_literal: true

# Adam (http://arxiv.org/abs/1412.6980)
class Adam
  def initialize(lr = 0.001, beta1 = 0.9, beta2 = 0.999)
    @lr = lr
    @beta1 = beta1
    @beta2 = beta2
    @iter = 0
    @m = nil
    @v = nil
  end

  def update(params, grads)
    unless @m
      @m = []
      @v = []
      params.each do |param|
        @m.append(Numo::SFloat.zeros(param.shape))
        @v.append(Numo::SFloat.zeros(param.shape))
      end
    end

    @iter += 1
    lr_t = @lr * Numo::SFloat::Math.sqrt(1.0 - @beta2**@iter) /
           (1.0 - @beta1**@iter)

    params.length.times do |i|
      @m[i] += (1 - @beta1) * (grads[i] - @m[i])
      @v[i] += (1 - @beta2) * (grads[i]**2 - @v[i])

      params[i].inplace - lr_t * @m[i] / (Numo::SFloat::Math.sqrt(@v[i]) + 1e-7)
    end
  end
end
