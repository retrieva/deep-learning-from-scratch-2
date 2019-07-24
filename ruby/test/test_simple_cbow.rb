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

  def test_backward
    Numo::NArray.srand(1)

    contexts = Numo::NArray[[1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0]]
    target = Numo::NArray[[0, 1, 0, 0, 0, 0, 0]]

    @simple_cbow.forward(contexts, target)
    @simple_cbow.backward

    expected = [[0.000617545, 0.00373067, 0.00794815],
                   [0.00201042, 0.00116041, 0.00344032],
                   [0.00539948, 0.00737815, 0.00165089],
                   [0.000508827, 0.00108065, 0.000687079],
                   [0.00904121, 0.00478644, 0.00342969],
                   [0.00164541, 0.0074603, 0.00138994],
                   [0.00411576, 0.00292532, 0.00869421]]

    @simple_cbow.word_vecs.to_a.zip(expected).each do |actual_row, expected_row|
      actual_row.zip(expected_row) do |actual, expected_value|
        assert_in_delta actual, expected_value, 0.00001
      end
    end
  end
end
