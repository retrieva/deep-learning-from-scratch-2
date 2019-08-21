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
      negative_sample[i, true] # = np.random.choiceに代わるもの
    end
  end
end

# With Replacement なので、要追加実装
def wrs(freq)
  freq.max_by { |_, weight| rand ** (1.0 / weight) }.first }
end
