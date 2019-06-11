require "test/unit"
require "simple_cbow"

class TestSimpleCBow < Test::Unit::TestCase
  def setup
    @simple_cbow = SimpleCBow.new(7, 3)
  end

  def test_initialize
    assert(true)
  end

  def test_forward
    contexts = Numo::NArray[[1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0]]
    target = Numo::NArray[[0, 1, 0, 0, 0, 0, 0]]

    @simple_cbow.forward(contexts, target)
    assert(true)
  end
end
