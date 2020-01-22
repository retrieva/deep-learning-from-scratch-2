$LOAD_PATH.unshift File.expand_path('../../lib', __FILE__)
require 'test/unit'

module Test
  module Unit
    module Assertions
      def assert_delta_array(expected, actual, delta = 0.00001, message = nil)
        assert_equal(expected.shape, actual.shape)
        expected.to_a.flatten.zip(actual.to_a.flatten).each do |expected_value, actual_value|
          assert_in_delta expected_value, actual_value, delta, message
        end
      end
    end
  end
end

