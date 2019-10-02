require "test/unit"
require "./lib/rnn"
require "numo/narray"

class TestRnn < Test::Unit::TestCase
  def setup
    # N = 5, D = 2, H = 2
    @wx = Numo::SFloat[[0.1, 0.2], [0.5, 0.8]] # D x H
    @wh = Numo::SFloat[[0.4, 0.2], [0.1, 0.9]] # H x H
    @b = Numo::SFloat[0.1]

    @target = Rnn.new(@wx, @wh, @b)
  end

  def test_initialize
    assert_equal @wx, @target.params[0]
    assert_equal [
                     Numo::SFloat[[0, 0], [0, 0]],
                     Numo::SFloat[[0, 0], [0, 0]],
                     Numo::SFloat[0]
                 ], @target.grads
  end

  def test_forward
    x = Numo::SFloat[0.4, 0.6] # D
    h_prev = Numo::SFloat[
        [0.3, 0.5],
        [0.1, 0.4],
        [0.7, 0.5],
        [0.3, 0.8],
        [0.2, 0.2]
    ] # N x H
    actual = @target.forward(x, h_prev)
    expected = [
      [0.544127, 0.824272],
      [0.4777, 0.777888],
      [0.64693, 0.848284],
      [0.5649, 0.893698],
      [0.492988, 0.706419]
    ]
    #assert_equal expected, actual

    actual.to_a.zip(expected).each do |actual_row, expected_row|
      actual_row.zip(expected_row) do |actual_value, expected_value|
        assert_in_delta actual_value, expected_value, 0.00001
      end
    end
  end
end
