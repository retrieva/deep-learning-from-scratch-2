require "test_helper"
require "time_embedding"
require "numo/narray"

class TimeEmbeddingTest < Test::Unit::TestCase
  def setup
    @w = Numo::SFloat[[0.20071, -0.210761, 0.21761, 0.20861, 0.203998], 
                      [-0.279034, 0.275155, -0.26858, -0.284655, -0.227953], 
                      [0.257709, -0.26606, 0.270528, 0.259538, 0.264047], 
                      [-0.255427, 0.260104, -0.256653, -0.248147, -0.250488], 
                      [0.245554, -0.246388, 0.249393, 0.23673, 0.24366], 
                      [0.207265, -0.207376, 0.209755, 0.207179, 0.206551], 
                      [-0.213308, 0.195443, -0.139508, -0.207847, 0.102623]]
    @target = TimeEmbedding.new(@w)
  end

  def test_initialize
    assert_equal [@w], @target.params      
    assert_equal [@w.new_zeros], @target.grads
  end

  def test_forward
    output = @target.forward(Numo::Int32[[0,1], [2,3], [4,5]])
    expected = [@w[[0,1], true].to_a, @w[[2,3], true].to_a, @w[[4,5], true].to_a]
    assert_equal(expected, output.to_a)
  end

  def test_backward
    output = @target.forward(Numo::Int32[[0,1], [2,3], [4,5]])
    @target.backward(output)
    expected = Numo::SFloat.zeros(7, 5)
    expected = @w[0...6, true].concatenate(Numo::SFloat.zeros(1, 5))
    assert_delta_array expected, @target.grads.first
  end
end
