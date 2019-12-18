$LOAD_PATH.unshift File.expand_path('../../lib', __FILE__)
require 'test/unit'

module Test
  module Unit
    module Assertions
      def assert_delta_array(expected, actual, delta = 0.00001, message = nil)
        assert_equal(actual.shape, expected.shape)
        actual.to_a.flatten.zip(expected.to_a.flatten).each do |actual_value, expected_value|
          assert_in_delta actual_value, expected_value, delta, message
        end
      end
    end
  end
end

