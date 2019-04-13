require "test/unit"
require "softmax_with_loss"
require "numo/narray"

class TestSoftmaxWithLoss < Test::Unit::TestCase
  def setup
    @target = SoftmaxWithLoss.new
    assert_equal [nil, nil], @target.params
  end

  def test_softmax1
    # Softmaxの検算はこのサイトを利用しました。https://keisan.casio.jp/exec/system/1516841458
    assert_in_delta 0.86681333219734, @target.softmax(Numo::SFloat[3, 7, 5])[1], 0.00001
    assert_in_delta 0.86681333219734, @target.softmax(Numo::SFloat[[3, 7, 5], [1, 9, 2]])[1], 0.00001
    assert_in_delta 0.99875420933679, @target.softmax(Numo::SFloat[[3, 7, 5], [1, 9, 2]])[4], 0.00001
  end
  
  def test_softmax2
    assert_in_delta 1.0, @target.softmax(Numo::SFloat[3, 7, 5]).sum, 0.00001
  end
  
  def test_cross_entropy_error
    assert_in_delta 0.51082562376, @target.cross_entropy_error(Numo::SFloat[0.3, 0.6, 0.1], Numo::SFloat[0, 1, 0]), 0.00001
    assert_in_delta 0.10536051565, @target.cross_entropy_error(Numo::SFloat[0.0, 0.1, 0.9], Numo::SFloat[0, 0, 1]), 0.00001
    assert_in_delta 0.308093069705, @target.cross_entropy_error(Numo::SFloat[[0.3, 0.6, 0.1], [0.0, 0.1, 0.9]], Numo::SFloat[[0, 1, 0], [0, 0, 1]]), 0.00001
  end

end
