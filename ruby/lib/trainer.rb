require 'matplotlib/pyplot'
require_relative 'util'

class Trainer
  def initialize(model, optimizer)
    @model = model
    @optimizer = optimizer
    @loss_list = []
    @eval_interval = nil
    @current_epoch = 0
  end

  def fit(x, t, max_epoch: 10, batch_size: 32, max_grad: nil, eval_interval: 20)
    data_size = x.shape.first
    max_iters = (data_size / batch_size).floor
    @eval_interval = eval_interval
    total_loss = 0
    loss_count = 0

    start_time = Time.now
    max_epoch.times do |epoch|
      @current_epoch += 1
      # Shuffle
      idx = Numo::Int64.new(data_size).store((0 ... data_size).to_a.shuffle)
      ex = get_at_dim_index(x, 0, idx)
      et = get_at_dim_index(t, 0, idx)

      max_iters.times do |iters|
        batch_range = (iters * batch_size) ... ((iters + 1) * batch_size)
        batch_x = get_at_dim_index(ex, 0, batch_range)
        batch_t = get_at_dim_index(et, 0, batch_range)

        # 勾配をもとめ、Optimizerでパラメータを更新
        loss = @model.forward(batch_x, batch_t)
        @model.backward
        params, grads = remove_duplicate(@model.params, @model.grads) # 共有された重みを1つに集約
        clip_grads(grads, max_grad) unless max_grad.nil?
        @optimizer.update(params, grads)
        total_loss += loss
        loss_count += 1

        # 評価
        if !eval_interval.nil? && (iters + 1) % eval_interval == 0
          avg_loss = total_loss / loss_count
          elapsed_time = Time.now - start_time
          puts "| epoch #{@current_epoch + 1} | iter #{iters + 1} / #{max_iters} | time #{elapsed_time} | loss #{avg_loss}"
          @loss_list << avg_loss
          total_loss = 0
          loss_count = 0
        end
      end
    end
  end

  def plot(ylim = nil)
    plt = Matplotlib::Pyplot
    x = (0 ... @loss_list.length).to_a
    plt.ylim(ylim) unless ylim.nil?
    plt.plot(x, @loss_list, label: 'train')
    plt.xlabel("iterations (x#{@eval_interval})")
    plt.ylabel('loss')
    plt.show
  end
end

def remove_duplicate(_params, _grads)
  # パラメータ配列中の重複する重みをひとつに集約し、その重みに対応する勾配を加算する
  params = _params.clone
  grads = _grads.clone

  while true do
    find_flg = false
    l = params.length
    (l - 1).times do |i|
      ((i + 1) .. l).each do |j|
        if params[i] && params[j] 
          if params[i] == params[j]
            # 重みを共有する場合
            grads[i].inplace + grads[j] # 勾配を加算
            find_flg = true
            params.delete_at(j)
            grads.delete_at(j)
          elsif params[i].ndim == 2 && params[j].ndim == 2 && params[i].transpose.shape == params[j].transpose.shape && params[i].transpose == params[j]
            # 転置行列として重みを共有する場合 (weight tying)
            grads[i].inplace + grads[j].transpose
            find_flg = true
            params.delete_at(j)
            grads.delete_at(j)
          end
        end
        break if find_flg
      end
      break if find_flg
    end
    break unless find_flg
  end
  return params, grads
end
