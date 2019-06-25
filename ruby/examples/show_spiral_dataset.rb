require 'matplotlib/pyplot'
require_relative "spiral"

plt = Matplotlib::Pyplot

spiral = Spiral.new.to_a

colors = ['yellow', 'green', 'red']

spiral.group_by{|x, t| t.to_a }.each do |t, x|
  x = x.map(&:first).map(&:to_a)
  plt.scatter(*x.transpose, c: colors[t.find_index(1)])
end
plt.savefig('spiral.png')
plt.show()
