require "test_helper"
require "time_rnn"
require "numo/narray"

class TimeRNNTest < Test::Unit::TestCase
  def setup
    # D = 2, H = 2
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
    x = Numo::SFloat[  # N x T x D ( 3 x 2 x 2 )
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

    expected = Numo::SFloat[
      [
        [0.300437, 0.413644],
        [0.523783, 0.790352]
      ],
      [
        [0.300437, 0.413644],
        [0.523783, 0.790352]
      ],
      [
        [0.300437, 0.413644],
        [0.523783, 0.790352]
      ]
    ]

    assert_delta_array(expected, actual)
  end

  def test_backward
    x = Numo::SFloat[  # N x T x D ( 3 x 2 x 2 )
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

    @target.forward(x)

    dh_next = Numo::SFloat[ # N x T x H ( 3 x 2 x 2 )
      [
        [0.3, 0.5],
        [0.7, 0.5]
      ],
      [
        [0.2, 0.2],
        [0.7, 0.5]
      ],
      [
        [0.3, 0.8],
        [0.2, 0.2]
      ]
    ]

    actual_dxs = @target.backward(dh_next)

    expected_dxs = Numo::SFloat[
      [
        [0.168503, 0.723202],
        [0.08833, 0.404116]
      ],
      [
        [0.109671, 0.47878],
        [0.08833, 0.404116]
      ],
      [
        [0.180169, 0.754616],
        [0.0295268, 0.13262]
      ]
    ]

    assert_delta_array(expected_dxs, actual_dxs)
  end
end
