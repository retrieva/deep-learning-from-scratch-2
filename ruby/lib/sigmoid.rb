class Sigmoid
  def forward(x)
    1 / (1 + Numo::NMath.exp(-x))
  end

  def params
    @params ||= []
  end

  def grads
    @grads ||= []
  end
end
