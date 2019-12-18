require "test_helper"
require "rnn"
require "numo/narray"

class RnnTest < Test::Unit::TestCase
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
    x = Numo::SFloat[
        [0.1, 0.4],
        [0.7, 0.5],
        [0.3, 0.5],
        [0.3, 0.8],
        [0.1, 0.9]
    ] # N x D
    h_prev = Numo::SFloat[
        [0.3, 0.5],
        [0.1, 0.4],
        [0.7, 0.5],
        [0.3, 0.8],
        [0.2, 0.2]
    ] # N x H
    actual = @target.forward(x, h_prev)
    expected = Numo::SFloat[
        [0.446244, 0.739783],
        [0.462117, 0.769867],
        [0.610677, 0.817754],
        [0.623065, 0.918602],
        [0.578363, 0.785664]
    ]
    #assert_equal expected, actual

    assert_delta_array(expected, actual)
  end

  def test_backward
    dh_next = Numo::SFloat[
        [0.3, 0.5],
        [0.1, 0.4],
        [0.7, 0.5],
        [0.3, 0.8],
        [0.2, 0.2]
    ] # N x H
    x = Numo::SFloat[
        [0.1, 0.4],
        [0.7, 0.5],
        [0.3, 0.5],
        [0.3, 0.8],
        [0.1, 0.9]
    ] # N x D
    h_prev = Numo::SFloat[
        [0.3, 0.5],
        [0.1, 0.4],
        [0.7, 0.5],
        [0.3, 0.8],
        [0.2, 0.2]
    ] # N x H
    @target.forward(x, h_prev)
    actual_dx, actual_dh_prev = @target.backward(dh_next)
    expected_dx = Numo::SFloat[[0.0692981, 0.301218],
                               [0.0404489, 0.16966],
                               [0.077023, 0.351987],
                               [0.043341, 0.191718],
                               [0.0286192, 0.127787]]
    assert_delta_array(expected_dx, actual_dx)
    expected_dh_prev = Numo::SFloat[[0.141376, 0.22775],
                                    [0.0640424, 0.154494],
                                    [0.208708, 0.19297],
                                    [0.0984021, 0.130797],
                                    [0.068549, 0.0822017]]
    assert_delta_array(expected_dh_prev, actual_dh_prev)
  end
end
