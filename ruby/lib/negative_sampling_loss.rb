# frozen_string_literal: true

require 'numo/narray'

class NegativeSamplingLoss
  def initialize(w, corpus, power = 0.75, sample_size = 5)
    @sample_size = sample_size
    @sampler = 
  end
end

class UnigramSampler
  def initialize(corpus, power, sample_size)
    @sample_size = sample_size
    @vocab_size = nil
    @word_p = nil

    counts = Hash.new(0)
    corpus.each do |word_id|
      counts[word_id] += 1
    end

    @vocab_size = counts.length

    @word_p = Numo::SFloat[*counts.values]

    @word_p = @word_p ** power
    @word_p /= @word_p.sum
  end

  def get_negative_sample(target)
    batch_size = target.shape[0]

    negative_sample = Numo::UInt32.zeros(batch_size, @sample_size)

    batch_size.times do |i|
      p = @word_p.dup
      target_idx = target[i]
      p[target_idx] = 0
      p /= p.sum
      negative_sample[i, true] = random_choice_without_replacement(
        @vocab_size, size: @sample_size, p: p, replacement: false
      )
    end
  end
end

# Implementation is based on the Weighted Random Sampling from this SO
# https://stackoverflow.com/a/2149533.
def random_choice_without_replacement(a, size: 1, p:)
  array = a.class == Integer ? (0...a).to_a : a
  items = array.zip(p)

  heap = rws_heap(items)

  size.times.map { rws_heap_pop(heap) }
end

Node = Struct.new(:w, :v, :tw)
Rand = Random.new

def rws_heap(items)
  h = [nil]
  items.each do |w, v|
    h.append(Node.new(w, v, w))
  end

  (h.length - 1).downto(2).each do |i|
    h[i >> 1].tw += h[i].tw
  end

  h
end

def rws_heap_pop(h)
  gas = h[1].tw * Rand.rand

  i = 1

  while gas >= h[i].w
    gas -= h[i].w
    i <<= 1
    if gas >= h[i].tw
      gas -= h[i].tw
      i += 1
    end
  end

  w = h[i].w
  v = h[i].v

  h[i].w = 0
  while i.positive?
    h[i].tw -= w
    i >>= 1
  end

  v
end
