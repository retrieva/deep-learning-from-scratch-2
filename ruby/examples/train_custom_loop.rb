require_relative 'spiral'
require_relative '../lib/optimizer'
require_relative 'two_layers_net'

# 1: ハイパーパラメータ設定
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

# 2: データ読み込み、モデルとオプティマイザ生成
samples = Spiral.new
x = samples.x # .shape => [300, 2]
t = samples.t # .shape => [300, 3]
model = TwoLayersNet.new(input_size: 2, hidden_size: hidden_size, output_size: 3)
optimizer = SGD.new(learning_rate)

data_size = x.shape.first # => 300
max_iters = (data_size / batch_size).floor # => 10
total_loss = 0
loss_count = 0
loss_list = []

max_epoch.times do |epoch|
  # 3: データのシャッフル
  # NOTE: Numoには random.permutation に対応する数列作成がないため、Arrayから作っている
  idx = Numo::Int64.new(data_size).store((0 ... data_size).to_a.shuffle)

  # NOTE: pythonのサンプルでは x = x[idx] となっているが、
  #       左辺はforループ内のローカル変数扱いなのでrubyでは変数名を変えている
  ex = x[idx, true]
  et = t[idx, true]

  max_iters.times do |iters|
    iter_range = (iters * batch_size) ... ((iters + 1) * batch_size)
    batch_x = ex[iter_range, true]
    batch_t = et[iter_range, true]

    # 4: 勾配を求め、パラメータを更新
    loss = model.forward(batch_x, batch_t)
    model.backward
    optimizer.update(model.params, model.grads)

    total_loss += loss
    loss_count += 1

    # 5: 定期的（10イテレーションに1回）に学習経過を出力
    if (iters + 1) % 10 == 0
      avg_loss = total_loss / loss_count
      puts "| epoch #{epoch+1} | iter #{iters+1} / #{max_iters} | loss #{avg_loss}"
      loss_list << avg_loss
      total_loss = 0
      loss_count = 0
    end
  end
end
