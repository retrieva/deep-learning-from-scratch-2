require "test/unit"
require_relative "../examples/two_layers_net"
require "numo/narray"

class TestTwoLayersNet < Test::Unit::TestCase
  def setup
    @target = TwoLayersNet.new(3, 5, 4)
  end

  def test_initialize
    affine1, sigmoid, affine2 = @target.layers
    assert_equal [3, 5], affine1.params[0].shape
  end
end