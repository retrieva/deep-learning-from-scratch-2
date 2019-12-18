require "test_helper"
require "time_rnn"
require "numo/narray"

class TimeRNNTest < Test::Unit::TestCase
  def setup
    # N = 5, D = 2, H = 2
    @wx = Numo::SFloat[[0.1, 0.2], [0.5, 0.8]] # D x H
    @wh = Numo::SFloat[[0.4, 0.2], [0.1, 0.9]] # H x H
    @b = Numo::SFloat[0.1]

    @target = TimeRnn.new(@wx, @wh, @b)
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
    x = Numo::SFloat[  # N x T x D
                       [
                         [0.1, 0.4],
                         [0.7, 0.5]
                       ],
                       [
                         [0.1, 0.4],
                         [0.7, 0.5]
                       ],
                       [
                         [0.1, 0.4],
                         [0.7, 0.5]
                       ],
                    ]

    actual = @target.forward(x)

    expected = []

    # assert_delta_array(expected, actual)
    # TODO 上のメソッドを汎用化して使う。
  end
end
