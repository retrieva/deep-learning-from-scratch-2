# frozen_string_literal: true

require 'adam'
require 'simple_cbow'
require 'trainer'
require 'util'

# window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1_000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = word_to_id.length
contexts, target = create_contexts_target(corpus, window_size: 1)
contexts = convert_one_hot(contexts, vocab_size)
target = convert_one_hot(target, vocab_size)

model = SimpleCBow.new(vocab_size, hidden_size)
optimizer = Adam.new
trainer = Trainer.new(model, optimizer)

trainer.fit(contexts, target, max_epoch: max_epoch, batch_size: batch_size)
trainer.plot

word_vecs = model.word_vecs

id_to_word.each do |word_id, word|
  printf("%s: %s\n", word, word_vecs[word_id, true].to_a)
end
