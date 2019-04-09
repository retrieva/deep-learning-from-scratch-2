require "test/unit"
require "./softmax_with_loss"
require "numo/narray"

class TestSoftmaxWithLoss < Test::Unit::TestCase
  def setup
    @np = Numo::SFloat
    @npm = Numo::SFloat::Math
  end

  def test_sample
    target = SoftmaxWithLoss.new
    assert_equal [nil, nil], target.params
    # Softmaxの期待する値はこのサイトの関数を流用 https://keisan.casio.jp/exec/system/1516841458
    assert_in_delta 0.86681333219734, target.softmax(@np[3, 7, 5])[1], 0.00001
    assert_in_delta 0.86681333219734, target.softmax(@np[[3, 7, 5],[1, 9, 2]])[1], 0.00001
    assert_in_delta 0.99875420933679, target.softmax(@np[[3, 7, 5],[1, 9, 2]])[4], 0.00001
  end
end
