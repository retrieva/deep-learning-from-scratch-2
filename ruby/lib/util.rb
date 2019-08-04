# frozen_string_literal: true

def clip_grads(grads, max_norm)
  total_norm = 0
  grads.each { |grad| total_norm += (grad ** 2).sum }
  total_norm = Numo::NMath.sqrt(total_norm)

  rate = max_norm / (total_norm + 1e-6)
  if rate < 1
    grads.each { |grad| grad *= rate }
  end
end

def preprocess(text)
  text = text.downcase
             .gsub('.', ' .')
  words = text.split(' ')

  word_to_id = {}
  id_to_word = {}

  words.each do |word|
    unless word_to_id.include?(word)
      new_id = word_to_id.length
      word_to_id[word] = new_id
      id_to_word[new_id] = word
    end
  end

  corpus = Numo::NArray[*words.map { |w| word_to_id[w] }]

  [corpus, word_to_id, id_to_word]
end

def create_contexts_target(corpus, window_size: 1)
  target = corpus[window_size...-window_size]
  contexts = []

  (window_size...(corpus.length - window_size)).each do |idx|
    cs = []
    (-window_size..window_size).each do |t|
      next if t.zero?
      cs.append(corpus[idx + t])
    end
    contexts.append(cs)
  end
  n_contexts = Numo::UInt32.zeros(contexts.length, contexts[0].length)
  n_contexts[] = contexts

  n_target = Numo::UInt32.zeros(target.length)
  n_target[] = target

  [n_contexts, n_target]
end

def convert_one_hot(corpus, vocab_size)
  n = corpus.shape[0]

  if corpus.ndim == 1
    one_hot = Numo::UInt32.zeros(n, vocab_size)
    corpus.each_with_index do |word_id, idx|
      one_hot[idx, word_id] = 1
    end
  elsif corpus.ndim == 2
    c = corpus.shape[1]
    one_hot = Numo::UInt32.zeros(n, c, vocab_size)

    n.times do |idx0|
      word_ids = corpus[idx0, true]
      word_ids.each_with_index do |word_id, idx1|
        one_hot[idx0, idx1, word_id] = 1
      end
    end
  end

  one_hot
end
