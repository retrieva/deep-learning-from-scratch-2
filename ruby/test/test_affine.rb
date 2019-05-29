require "test/unit"
require "./lib/affine"
require "numo/narray"

class TestAffine < Test::Unit::TestCase
  def setup
    @weight = Numo::SFloat[[1,2,3],[5,8,13]] # input: 2, hidden: 3
    @bias   = Numo::SFloat[0.2, 0.3, 0.4]
    @target = Affine.new(@weight, @bias)
  end

  def test_initialize
    assert_equal [@weight, @bias], @target.params
    assert_equal [ Numo::SFloat[[0,0,0],[0,0,0]],
                   Numo::SFloat[0,0,0]
                 ], @target.grads
  end

  def test_forward1
    x = Numo::SFloat[[3,1.5]]
    # x.dot(W)     = [3*1 + 1.5*5, 3*2 + 1.5*8, 3*3 + 1.5*13]
    #              = [3   + 7.5  , 6   + 12   , 9   + 19.5  ]
    #              = [10.5, 18  , 28.5]
    # x.dot(W) + b = [10.7, 18.3, 28.9]
    assert_equal Numo::SFloat[[10.7,18.3,28.9]], @target.forward(x)
  end

  def test_forward2
    x = Numo::SFloat[[2,7],[3,9],[11,13]]
    # x.dot(W) = [[2*1  + 7*5,  2*2  + 7*8,  2*3  + 7*13 ],
    #             [3*1  + 9*5,  3*2  + 9*8,  3*3  + 9*13 ],
    #             [11*1 + 13*5, 11*2 + 13*8, 11*3 + 13*13]]
    #          = [[37, 60, 97], [48, 78, 126], [76, 126, 202]]
    assert_equal Numo::SFloat[[37.2, 60.3, 97.4],
                              [48.2, 78.3, 126.4],
                              [76.2, 126.3, 202.4]], @target.forward(x)
  end

  def test_backward1
    x = Numo::SFloat[[2,7],[3,9],[11,13]]
    dout = Numo::SFloat[[1,0,0],[0,1,0],[0,0,1]] # hidden = 3, input = 2
    @target.forward(x)
    dLdx = @target.backward(dout)
    assert_equal @weight.transpose, dLdx
    assert_equal x.transpose, @target.grads[0]
    assert_equal Numo::SFloat[1,1,1], @target.grads[1]
  end

  def test_backward2
    x = Numo::SFloat[[2,7],[3,9],[11,13]]
    dout = Numo::SFloat[[1,0.5,0.5],[0.5,1,0.5],[0.5,0.5,1]] # hidden = 3, input = 2
    @target.forward(x)
    dLdx = @target.backward(dout)
    assert_equal Numo::SFloat[[3.5, 15.5], 
                              [4, 17], 
                              [4.5, 19.5]], dLdx
    assert_equal Numo::SFloat[[9, 9.5, 13.5], 
                              [18, 19, 21]], @target.grads[0]
    assert_equal Numo::SFloat[2,2,2], @target.grads[1]
  end

  def test_backward3
    x = Numo::SFloat[[3, 1.5]]
    dout = Numo::SFloat[[1],[0.5],[1]] # hidden = 3, input = 2
    @target.forward(x)
    dLdx = @target.backward(dout)
    assert_equal Numo::SFloat[[6,26],[3,13],[6,26]], dLdx #dx
    assert_equal Numo::SFloat[[7.5],[3.75]], @target.grads[0] #dW
    assert_equal Numo::SFloat[2.5], @target.grads[1] #repeat
  end
end