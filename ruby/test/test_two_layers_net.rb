require "test/unit"
require_relative "../examples/two_layers_net"
require "numo/narray"

class TestTwoLayersNet < Test::Unit::TestCase
  def setup
    @target = TwoLayersNet.new(input_size: 3, hidden_size: 5, output_size: 4)
  end

  def test_initialize
    affine1, sigmoid, affine2 = @target.layers
    assert_equal [3, 5], affine1.params[0].shape
    assert_equal Sigmoid, sigmoid.class
    assert_equal [5, 4], affine2.params[0].shape
  end

  def test_forward
    x = Numo::SFloat[[1,2,3],[4,5,6]]
    t = Numo::SFloat[[1,2,3,4],[5,6,7,8]]
    cross_entropy_error = @target.forward(x, t)
    assert_equal Float, cross_entropy_error.class
  end

  def test_backward1
    x = Numo::SFloat[[1,2,3],[4,5,6]]
    t = Numo::SFloat[[1,2,3,4],[5,6,7,8]]
    cross_entropy_error = @target.forward(x, t)
    dout = 0.8
    last_dout = @target.backward(dout)
    assert_equal [2,3], last_dout.shape
  end

  def test_backward2
    x = Numo::SFloat[1,2,3]
    t = Numo::SFloat[1,2,3,4]
    cross_entropy_error = @target.forward(x, t)
    dout = 0.8
    last_dout = @target.backward(dout)
    assert_equal [1,3], last_dout.shape
  end
end