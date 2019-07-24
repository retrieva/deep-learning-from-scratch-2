def clip_grads(grads, max_norm)
  total_norm = 0
  grads.each { |grad| total_norm += (grad ** 2).sum }
  total_norm = Numo::NMath.sqrt(total_norm)

  rate = max_norm / (total_norm + 1e-6)
  if rate < 1
    grads.each { |grad| grad *= rate }
  end
end