require "numo/narray"
require "mat_mul.rb"

# サンプルのコンテキストデータ
c0 = Numo::NArray[[1, 0, 0, 0, 0, 0, 0]]
c1 = Numo::NArray[[0, 0, 1, 0, 0, 0, 0]]

# 重みの初期化
w_in = Numo::DFloat.new(7, 3).rand
w_out = Numo::DFloat.new(3, 7).rand

# レイヤの生成
in_layer0 = MatMul.new(w_in)
in_layer1 = MatMul.new(w_in)
out_layer = MatMul.new(w_out)

# 順伝搬
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)
s = out_layer.forward(h)

pp h0.to_a
#pp s.to_a
