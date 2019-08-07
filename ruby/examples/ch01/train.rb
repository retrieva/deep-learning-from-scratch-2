require_relative '../lib/optimizer' # SGD
require_relative '../lib/trainer'
require_relative 'two_layers_net'
require_relative 'spiral'

max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

spiral = Spiral.new
x = spiral.x
t = spiral.t
model = TwoLayersNet.new(input_size: 2, hidden_size: hidden_size, output_size: 3)
optimizer = SGD.new(learning_rate)

trainer = Trainer.new(model, optimizer)
trainer.fit(x, t, max_epoch: max_epoch, batch_size: batch_size, eval_interval: 10)
trainer.plot()