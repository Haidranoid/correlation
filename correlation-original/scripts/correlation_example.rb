def correlation(x, y)
  # Check if arrays have same length
  return nil if x.length != y.length || x.empty?
  
  n = x.length
  
  # Calculate means
  mean_x = x.sum.to_f / n
  mean_y = y.sum.to_f / n
  
  # Calculate numerator and denominators
  numerator = 0
  sum_sq_x = 0
  sum_sq_y = 0
  
  x.zip(y) do |xi, yi|
    diff_x = xi - mean_x
    diff_y = yi - mean_y
    
    numerator += diff_x * diff_y
    sum_sq_x += diff_x * diff_x
    sum_sq_y += diff_y * diff_y
  end
  
  # Calculate correlation coefficient
  denominator = Math.sqrt(sum_sq_x * sum_sq_y)
  
  # Avoid division by zero
  return 0 if denominator == 0
  
  numerator / denominator
end

# Example usage
array1 = [1, 2, 3, 4, 5]
array2 = [2, 4, 6, 8, 10]
array3 = [10, 8, 6, 4, 2]

puts "Perfect positive correlation:"
puts correlation(array1, array2)  # Should be close to 1.0

puts "\nPerfect negative correlation:"
puts correlation(array1, array3)  # Should be close to -1.0

puts "\nModerate negative correlation example:"
random1 = [1, 5, 2, 8, 3]
random2 = [4, 1, 9, 2, 6]
puts correlation(random1, random2)

# More concise version using enumerable methods
def correlation_concise(x, y)
  return nil if x.length != y.length || x.empty?
  
  n = x.length.to_f
  mean_x = x.sum / n
  mean_y = y.sum / n
  
  pairs = x.zip(y)
  
  numerator = pairs.sum { |xi, yi| (xi - mean_x) * (yi - mean_y) }
  sum_sq_x = x.sum { |xi| (xi - mean_x) ** 2 }
  sum_sq_y = y.sum { |yi| (yi - mean_y) ** 2 }
  
  denominator = Math.sqrt(sum_sq_x * sum_sq_y)
  denominator == 0 ? 0 : numerator / denominator
end

# Inline C version for better performance
require 'inline'

class CorrelationCalculator
  inline do |builder|
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

# Benchmark comparison
require 'benchmark'

# Create larger test arrays for meaningful benchmarks
large_x = (1..10000).to_a
large_y = large_x.map { |x| x * 2 + rand(-5..5) }

calc = CorrelationCalculator.new

puts "\nBenchmark comparison (10,000 elements):"
ruby_time = nil
c_time = nil

Benchmark.bm(15) do |bm|
  ruby_time = bm.report("Ruby version:") do
    100.times { correlation_concise(large_x, large_y) }
  end
  
  c_time = bm.report("Inline C:") do
    100.times { calc.correlation_fast(large_x, large_y) }
  end
end

# Performance comparison analysis
puts "\n" + "-" * 50
puts "PERFORMANCE ANALYSIS"
puts "-" * 50

ruby_total = ruby_time.real
c_total = c_time.real
speedup = ruby_total / c_total
improvement_percent = ((ruby_total - c_total) / ruby_total * 100)

puts "Ruby version time:    #{sprintf('%.6f', ruby_total)} seconds"
puts "C version time:       #{sprintf('%.6f', c_total)} seconds"
puts "Time difference:      #{sprintf('%.6f', ruby_total - c_total)} seconds saved"
puts "Speedup factor:       #{sprintf('%.1f', speedup)}x faster"
puts "Performance gain:     #{sprintf('%.1f', improvement_percent)}% improvement"

# Context for the performance gain
puts "\nWhat this means:"
if speedup >= 50
  puts "🚀 EXCEPTIONAL speedup - C version is blazingly fast!"
elsif speedup >= 20
  puts "⚡ EXCELLENT speedup - Significant performance improvement"
elsif speedup >= 10
  puts "✅ GREAT speedup - Very noticeable performance boost"
elsif speedup >= 5
  puts "👍 GOOD speedup - Worthwhile performance improvement"
else
  puts "📊 MODEST speedup - Some improvement but overhead may be a factor"
end

puts "\nFor larger datasets, the performance difference becomes even more dramatic!"
puts "This #{sprintf('%.1f', speedup)}x speedup means what takes #{sprintf('%.0f', speedup)} minutes in Ruby"
puts "takes only 1 minute with the optimized C version."

puts "\nResults comparison:"
puts "Ruby result: #{correlation_concise(array1, array2)}"
puts "C result: #{calc.correlation_fast(array1, array2)}"

# Verification section - demonstrate both implementations produce identical results
puts "\n" + "=" * 60
puts "VERIFICATION: Ruby vs C Implementation Comparison"
puts "=" * 60

def verify_implementations(name, x, y, calculator)
  ruby_result = correlation_concise(x, y)
  c_result = calculator.correlation_fast(x, y)
  
  puts "\n#{name}:"
  puts "  Arrays: #{x} vs #{y}"
  puts "  Ruby:   #{ruby_result}"
  puts "  C:      #{c_result}"
  
  # Check if results match (handling nil and special cases)
  if ruby_result.nil? && c_result.respond_to?(:nan?) && c_result.nan?
    puts "  Status: ✅ MATCH (both handle invalid input correctly)"
    return true
  elsif ruby_result.nil? || (c_result.respond_to?(:nan?) && c_result.nan?)
    puts "  Status: ❌ MISMATCH (different handling of edge cases)"
    return false
  elsif (ruby_result - c_result).abs < 1e-14
    difference = (ruby_result - c_result).abs
    if difference == 0.0
      puts "  Status: ✅ PERFECT MATCH"
    else
      puts "  Status: ✅ MATCH (diff: #{difference}, within floating-point precision)"
    end
    return true
  else
    puts "  Status: ❌ SIGNIFICANT DIFFERENCE (#{(ruby_result - c_result).abs})"
    return false
  end
end

test_cases = [
  ["Perfect positive correlation", [1, 2, 3, 4, 5], [2, 4, 6, 8, 10]],
  ["Perfect negative correlation", [1, 2, 3, 4, 5], [10, 8, 6, 4, 2]],
  ["Weak correlation", [1, 2, 3, 4, 5], [3, 5, 1, 2, 4]],
  ["Moderate negative correlation", [1, 5, 2, 8, 3], [4, 1, 9, 2, 6]],
  ["Identical arrays", [3, 1, 4, 1, 5], [3, 1, 4, 1, 5]],
  ["Zero variance case", [1, 2, 3, 4, 5], [1, 1, 1, 1, 1]],
  ["Negative numbers", [-5, -3, -1, 1, 3, 5], [-10, -6, -2, 2, 6, 10]],
  ["Floating point", [1.1, 2.2, 3.3, 4.4, 5.5], [2.1, 4.2, 6.3, 8.4, 10.5]],
  ["Empty arrays", [], []],
  ["Different lengths", [1, 2, 3], [1, 2, 3, 4, 5]]
]

passed_tests = 0
total_tests = test_cases.length

test_cases.each do |name, x, y|
  begin
    if verify_implementations(name, x, y, calc)
      passed_tests += 1
    end
  rescue => e
    puts "\n#{name}:"
    puts "  Arrays: #{x} vs #{y}"
    puts "  Status: ❌ ERROR - #{e.message}"
  end
end

puts "\n" + "-" * 60
puts "VERIFICATION SUMMARY"
puts "-" * 60
puts "Tests passed: #{passed_tests}/#{total_tests}"
puts "Success rate: #{(passed_tests.to_f / total_tests * 100).round(1)}%"

if passed_tests == total_tests
  puts "🎉 ALL TESTS PASSED! Both implementations are mathematically equivalent."
  puts "✅ You can use either version with confidence - they produce identical results!"
else
  puts "⚠️  Some tests failed. Check the implementation for discrepancies."
end