# frozen_string_literal: true

require 'inline'

# Service for high-performance correlation calculations using inline C
# Handles mathematical operations with Ruby fallback for reliability
class CorrelationCalculationService
  class CorrelationError < StandardError; end
  
  CORRELATION_RANGES = {
    (0...0.1) => 'No correlation',
    (0.1...0.3) => 'Weak correlation',
    (0.3...0.5) => 'Moderate correlation',
    (0.5...0.7) => 'Strong correlation',
    (0.7...0.9) => 'Very strong correlation',
    (0.9..1.0) => 'Extremely strong correlation'
  }.freeze

  # High-performance correlation calculator using inline C
  class FastCorrelationCalculator
    inline do |builder|
      # GCC 14+ makes this an error; RubyInline's generated rb_define_method casts trip it
      builder.add_compile_flags '-Wno-error=incompatible-pointer-types'
      builder.c <<-EOC
        double correlation_fast(VALUE x_array, VALUE y_array) {
          long n = RARRAY_LEN(x_array);
          
          // Check if arrays have same length
          if (n != RARRAY_LEN(y_array) || n == 0) {
            return 0.0/0.0; // Return NaN for invalid input
          }
          
          // Convert Ruby arrays to C arrays and calculate sums
          double sum_x = 0.0, sum_y = 0.0;
          long i;
          for (i = 0; i < n; i++) {
            sum_x += NUM2DBL(RARRAY_AREF(x_array, i));
            sum_y += NUM2DBL(RARRAY_AREF(y_array, i));
          }
          
          // Calculate means
          double mean_x = sum_x / n;
          double mean_y = sum_y / n;
          
          // Calculate correlation components
          double numerator = 0.0;
          double sum_sq_x = 0.0;
          double sum_sq_y = 0.0;
          
          for (i = 0; i < n; i++) {
            double xi = NUM2DBL(RARRAY_AREF(x_array, i));
            double yi = NUM2DBL(RARRAY_AREF(y_array, i));
            double diff_x = xi - mean_x;
            double diff_y = yi - mean_y;
            
            numerator += diff_x * diff_y;
            sum_sq_x += diff_x * diff_x;
            sum_sq_y += diff_y * diff_y;
          }
          
          // Calculate correlation coefficient
          double denominator = sqrt(sum_sq_x * sum_sq_y);
          
          if (denominator == 0.0) {
            return 0.0;
          }
          
          return numerator / denominator;
        }
      EOC
    end
  end

  def initialize
    @fast_calculator = nil
  end

  # Calculate Pearson correlation coefficient between two datasets
  # @param data_points [Array<Array>] Array of [x, y] value pairs
  # @return [Float, nil] Correlation coefficient or nil if invalid
  def calculate_pearson_correlation(data_points)
    return nil if data_points.empty?
    
    x_values = data_points.map(&:first)
    y_values = data_points.map(&:second)
    
    return nil if x_values.length < 2
    
    # Use inline C implementation by default for better performance
    begin
      correlation = fast_calculator.correlation_fast(x_values, y_values)
      
      # Handle NaN result (invalid input)
      return nil if correlation.respond_to?(:nan?) && correlation.nan?
      
      correlation
    rescue => e
      # Fallback to Ruby implementation if C version fails
      Rails.logger.warn "Inline C correlation failed (#{e.message}), falling back to Ruby implementation"
      calculate_pearson_correlation_ruby(data_points)
    end
  end

  # Calculate statistical interpretation of correlation strength
  # @param correlation [Float] Correlation coefficient
  # @return [Hash] Hash containing strength and direction information
  def interpret_correlation(correlation)
    return { strength: 'Invalid', direction: 'None', description: 'Invalid correlation value' } if correlation.nil?

    abs_correlation = correlation.abs
    
    strength = CORRELATION_RANGES.find { |range, _| range.cover?(abs_correlation) }&.last || 'Invalid correlation'
    direction, r_squared = correlation > 0 ? 'Positive' : 'Negative', correlation**2
    { strength: strength, direction: direction, description: "#{direction} #{strength.downcase}",
      r_squared: r_squared, variance_explained: (r_squared * 100).round(1) }
  end

  private

  # Lazy initialization of fast calculator to handle potential compilation issues
  def fast_calculator
    @fast_calculator ||= FastCorrelationCalculator.new
  end

  # Ruby fallback implementation for reliability
  # @param data_points [Array<Array>] Array of [x, y] value pairs
  # @return [Float, nil] Correlation coefficient or nil if invalid
  def calculate_pearson_correlation_ruby(data_points)
    return nil if data_points.empty?
    
    x_values = data_points.map(&:first)
    y_values = data_points.map(&:second)
    
    n = data_points.length
    return nil if n < 2
    
    # Calculate means
    mean_x = x_values.sum / n.to_f
    mean_y = y_values.sum / n.to_f
    
    # Calculate correlation components
    numerator = 0
    sum_sq_x = 0
    sum_sq_y = 0
    
    x_values.zip(y_values).each do |x, y|
      diff_x = x - mean_x
      diff_y = y - mean_y
      
      numerator += diff_x * diff_y
      sum_sq_x += diff_x * diff_x
      sum_sq_y += diff_y * diff_y
    end
    
    denominator = Math.sqrt(sum_sq_x * sum_sq_y)
    return nil if denominator == 0
    
    numerator / denominator
  end
end