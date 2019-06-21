require "test/unit"
require "numo/narray"
require "./lib/optimizer"

class TestSGD < Test::Unit::TestCase
  def setup
    @lr = 0.02
    @sgd = SGD.new(@lr)
  end

  def test_update
    params = [Numo::SFloat[0.1, 0.2, 0.3], Numo::SFloat[0.4, 0.5, 0.6]]
    grads = [Numo::SFloat[0.01, 0.02, 0.03], Numo::SFloat[0.04, 0.05, 0.06]]

    @sgd.update(params, grads)

    # [[0.1-0.02*0.01, 0.2-0.02*0.02, 0.3-0.02*0.03]
    #  [0.4-0.02*0.04, 0.5-0.02*0.05, 0.6-0.02*0.06]]
    # = [[0.0998, 0.1996, 0.2994],
    #    [0.3992, 0.499, 0.5988]]
    assert_equal [Numo::SFloat[0.0998, 0.1996, 0.2994],
                  Numo::SFloat[0.3992, 0.499, 0.5988]],
                 params
  end
end

