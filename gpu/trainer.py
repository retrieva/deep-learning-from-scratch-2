#!/usr/bin/env python

from optimizer import SGD
from trainer import Trainer
from spiral import spiral
from two_layer_net import TwoLayerNet


if __name__ == '__main__':
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    learning_rate = 1.0

    x, t = spiral.load_data()
    model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
    optimizer = SGD(lr=learning_rate)

    trainer = Trainer(model, optimizer)
    trainer.fit(x, t, max_epoch, batch_size, eval_interval=10)
    trainer.plot()
