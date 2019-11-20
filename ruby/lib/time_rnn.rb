# frozen_string_literal: true

require 'numo/narray'
require_relative '../lib/rnn'

class TimeRnn
  attr_accessor :params, :grads

  def initialize(wx, wh, b, stateful: false)
    @params = [wx, wh, b]
    @grads = [wx.new_zeros, wh.new_zeros, b.new_zeros]
    @layers = nil

    @h = nil
    @dh = nil
    @stateful = stateful
  end

  def forward(xs)
    wx, wh, b = @params
    n, t, d = xs.shape
    d, h = wx.shape

    @layers = []
    hs = Numo::SFloat.zeros(n, t, h)

    if !@stateful || @h.nil?
      @h = Numo::SFloat.zeros(n, h)
    end

    t.times do |ti|
      layer = Rnn.new(*@params)
      @h = layer.forward(xs[true, ti, true], @h)
      hs[true, ti, true] = @h
      @layers.append(layer)
    end

    hs
  end

  def backward(dhs)
    wx, wh, b = @params
    n, t, h = dhs.shape
    d, h = wx.shape

    dxs = Numo::SFloat.new(n, t, d)
    dh = 0
    grads = [0, 0, 0]

    (t - 1).downto(0).to_a do |ti|
      layer = @layers[ti]
      dx, dh = layer.backward(dhs[true, ti, true] + dh)
      dxs[true, ti, true] = dx

      layer.grads.each_with_index do |grad, i|
        grads[i] += grad
      end
    end

    grads.each_with_index do |grad, i|
      @grads[i].store(grad)
    end

    @dh = dh

    dxs
  end

  def state=(h)
    @h = h
  end

  def reset_state
    @h = nil
  end
end
