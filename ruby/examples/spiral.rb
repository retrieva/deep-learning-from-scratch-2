require "numo/narray"
require "datasets"
require "datasets/dataset"

class Spiral < Datasets::Dataset
  N = 100  # クラスごとのサンプル数
  DIM = 2  # データの要素数
  CLS_NUM = 3  # クラス数

  def initialize(seed=1984)
    super()
    @metadata.id = "spiral"
    @metadata.name = "Spiral"
    @metadata.url = "https://github.com/retrieva/deep-learning-from-scratch-2"
    @metadata.description = "Spiral dataset"

    random = Random.new(seed)

    @x = Numo::DFloat.zeros(N * CLS_NUM, DIM)
    @t = Numo::Int64.zeros(N * CLS_NUM, CLS_NUM)

    CLS_NUM.times do |j|
      N.times do |i|  # N*j, N*(j+1))
        rate = i.to_f / N
        radius = 1.0 * rate
        theta = j * 4.0 + 4.0 * rate + random.rand(0.2)

        ix = N * j + i
        @x[ix, true] = [radius * Math.sin(theta), radius * Math.cos(theta)]
        @t[ix, j] = 1
      end
    end
  end

  def each
    return to_enum(__method__) unless block_given?

    (N * CLS_NUM).times do |ix|
      yield [@x[ix, true], @t[ix, true]]
    end
  end
end
