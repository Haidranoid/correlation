# frozen_string_literal: true

# Ractor-compatible correlation calculation service
# This service is designed to work within Ractor constraints for parallel processing
class RactorCorrelationService
  # Create a Ractor for calculating correlation
  # @param data_points [Array<Array>] Array of [x, y] value pairs (must be shareable)
  # @return [Ractor] Ractor that will calculate correlation
  def self.create_correlation_ractor(data_points)
    # Ensure data is shareable for Ractor
    shareable_data = data_points.map(&:dup).freeze
    
    Ractor.new(shareable_data) do |data|
      # Calculate correlation directly in this Ractor (no method definition needed)
      correlation = if data.empty?
        nil
      else
        x_values = data.map(&:first)
        y_values = data.map(&:second)
        
        n = data.length
        if n < 2
          nil
        else
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
          if denominator == 0
            nil
          else
            numerator / denominator
          end
        end
      end
      
      # Return result with interpretation
      if correlation.nil?
        { correlation: nil, error: "Correlation calculation failed" }
      else
        abs_correlation = correlation.abs
        
        strength = case abs_correlation
                   when 0...0.1 then 'No correlation'
                   when 0.1...0.3 then 'Weak correlation'
                   when 0.3...0.5 then 'Moderate correlation'
                   when 0.5...0.7 then 'Strong correlation'
                   when 0.7...0.9 then 'Very strong correlation'
                   when 0.9..1.0 then 'Extremely strong correlation'
                   else 'Invalid correlation'
                   end
        
        direction = correlation > 0 ? 'Positive' : 'Negative'
        r_squared = correlation**2
        
        {
          correlation: correlation,
          r_squared: r_squared,
          strength: strength,
          direction: direction,
          description: "#{direction} #{strength.downcase}",
          variance_explained: (r_squared * 100).round(1)
        }
      end
    end
  end
  
  # Process multiple correlations in parallel using Ractors
  # @param correlation_tasks [Array<Hash>] Array of task hashes with :data_points and :metadata
  # @param max_ractors [Integer] Maximum number of concurrent Ractors (default: CPU count)
  # @return [Array<Hash>] Array of results with correlation data and metadata
  def self.process_correlations_in_parallel(correlation_tasks, max_ractors: nil)
    max_ractors ||= [Etc.nprocessors, 8].min # Limit to reasonable number
    
    results = []
    
    # Process tasks in batches
    correlation_tasks.each_slice(max_ractors) do |batch|
      # Create Ractors for this batch
      ractors = batch.map.with_index do |task, index|
        {
          ractor: create_correlation_ractor(task[:data_points]),
          metadata: task[:metadata],
          batch_index: index
        }
      end
      
      # Collect results from this batch
      batch_results = []
      ractors.each do |ractor_info|
        begin
          correlation_result = ractor_info[:ractor].take
          
          # Combine correlation result with metadata
          result = {
            **correlation_result,
            **ractor_info[:metadata],
            batch_index: ractor_info[:batch_index]
          }
          
          batch_results << result
        rescue => e
          # Handle Ractor errors gracefully
          error_result = {
            correlation: nil,
            error: "Ractor failed: #{e.message}",
            **ractor_info[:metadata],
            batch_index: ractor_info[:batch_index]
          }
          
          batch_results << error_result
        end
      end
      
      results.concat(batch_results)
    end
    
    results
  end
  
  # Create a correlation task for parallel processing
  # @param data_points [Array<Array>] Correlation data points
  # @param metadata [Hash] Additional information about this correlation
  # @return [Hash] Task ready for parallel processing
  def self.create_correlation_task(data_points, metadata = {})
    {
      data_points: data_points.map(&:dup).freeze, # Ensure shareability
      metadata: metadata.dup.freeze
    }
  end
  
  # Calculate single correlation (non-parallel version for comparison)
  # @param data_points [Array<Array>] Array of [x, y] value pairs
  # @return [Hash] Correlation result with interpretation
  def self.calculate_single_correlation(data_points)
    ractor = create_correlation_ractor(data_points)
    ractor.take
  rescue => e
    { correlation: nil, error: "Calculation failed: #{e.message}" }
  end
end