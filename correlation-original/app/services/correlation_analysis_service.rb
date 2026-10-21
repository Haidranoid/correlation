# frozen_string_literal: true

# Main orchestration service for correlation analysis
# Coordinates data fetching, calculation, and result persistence
class CorrelationAnalysisService
  class AnalysisError < StandardError; end

  def initialize(silent: false)
    @calculation_service = CorrelationCalculationService.new
    @silent = silent
  end

  # Perform single correlation analysis with specified parameters
  # @param symbol1 [String] First stock symbol
  # @param symbol2 [String] Second stock symbol
  # @param window1 [Integer] Time window for first symbol
  # @param window2 [Integer] Time window for second symbol
  # @param offset [Integer] Offset in days
  # @param start_date [Date] Analysis start date
  # @param end_date [Date] Analysis end date
  # @return [Hash] Analysis results
  def analyze_correlation(symbol1, symbol2, window1, window2, offset, start_date, end_date)
    Rails.logger.info "Starting correlation analysis: #{symbol1} vs #{symbol2}, windows(#{window1},#{window2}), offset(#{offset}), range(#{start_date} to #{end_date})"
    
    # Validate symbols are different
    raise AnalysisError, "Cannot analyze correlation between the same symbol: #{symbol1}" if StockCorrelation.same_symbols?(symbol1, symbol2)
    
    # Step 1: Fetch stock data for both symbols
    stock1_data = StockDataService.fetch_stock_data(symbol1, start_date, end_date, window1)
    stock2_data = StockDataService.fetch_stock_data(symbol2, start_date, end_date, window2, offset)
    
    validate_stock_data!(stock1_data, stock2_data, symbol1, symbol2)
    
    # Step 2: Calculate percentage changes
    changes1 = StockDataService.calculate_percentage_changes(stock1_data, window1)
    changes2 = StockDataService.calculate_percentage_changes(stock2_data, window2)
    
    Rails.logger.info "Calculated changes: #{symbol1}(#{changes1.length}), #{symbol2}(#{changes2.length})"
    
    # Step 3: Align data points by date
    aligned_data = StockDataService.align_data_points(changes1, changes2, offset)
    
    validate_aligned_data!(aligned_data)
    
    # Step 4: Calculate correlation
    correlation = @calculation_service.calculate_pearson_correlation(aligned_data)
    raise AnalysisError, "Correlation calculation failed" if correlation.nil?
    
    # Step 5: Generate comprehensive results
    build_analysis_result(symbol1, symbol2, window1, window2, offset, 
                         start_date, end_date, correlation, aligned_data)
  end

  # Find all statistically relevant correlations across all possible offsets
  # @param symbol1 [String] First stock symbol
  # @param symbol2 [String] Second stock symbol
  # @param window1 [Integer] Time window for first symbol
  # @param window2 [Integer] Time window for second symbol
  # @param start_date [Date] Analysis start date
  # @param end_date [Date] Analysis end date
  # @return [Hash] Statistically relevant correlation results with analysis summary
  def find_best_correlation(symbol1, symbol2, window1, window2, start_date, end_date)
    Rails.logger.info "Finding best correlation across all offsets for #{symbol1} vs #{symbol2}"
    
    # Validate symbols are different
    return { error: "Cannot analyze correlation between the same symbol: #{symbol1}" } if StockCorrelation.same_symbols?(symbol1, symbol2)
    
    # Determine maximum possible offset range  
    log_progress "  📊 Calculating maximum possible offset range from available data..."
    max_offset = calculate_max_theoretical_offset(symbol1, symbol2, start_date, end_date)
    
    log_progress "  🔍 Finding actual maximum viable offset..."
    actual_max_offset = find_actual_max_viable_offset(symbol1, symbol2, window1, window2, max_offset)
    
    log_progress "  ✅ Actual maximum viable offset: +#{actual_max_offset} days"
    log_progress "  🔄 Starting offset testing from most recent data, working backwards..."
    log_progress ""
    
    # Test all viable offsets
    correlations = test_offset_range(symbol1, symbol2, window1, window2, 
                                   start_date, end_date, actual_max_offset)
    
    return { error: "No valid correlations found" } if correlations.empty?
    
    # Filter for statistically relevant correlations
    # Criteria: |correlation| >= 0.3 AND R² >= 0.09 (moderate correlation or stronger)
    relevant_correlations = correlations.select do |result|
      abs_correlation = result[:correlation].abs
      abs_correlation >= 0.3 && result[:r_squared] >= 0.09
    end
    
    # Sort by absolute correlation value
    sorted_correlations = correlations.sort_by { |r| -r[:correlation].abs }
    sorted_relevant = relevant_correlations.sort_by { |r| -r[:correlation].abs }
    
    {
      best_correlation: sorted_relevant.first,
      relevant_correlations: sorted_relevant,
      total_tested: correlations.length,
      total_relevant: relevant_correlations.length,
      all_correlations: sorted_correlations
    }
  end

  # Save correlation result to database
  # @param result [Hash] Analysis result
  # @return [StockCorrelation] Saved correlation record
  def save_correlation_result(result)
    Rails.logger.info "Saving correlation to database..."
    StockCorrelation.create_from_analysis(result, result[:symbol1], result[:symbol2], result[:window1], result[:window2])
  end

  # Save all statistically relevant correlation results to database, replacing existing records with same parameters
  # @param relevant_results [Array] Array of statistically relevant correlation results
  # @param symbol1 [String] First stock symbol
  # @param symbol2 [String] Second stock symbol
  # @param window1 [Integer] Time window for first symbol
  # @param window2 [Integer] Time window for second symbol
  # @return [Integer] Number of correlations saved
  def save_relevant_correlations(relevant_results, symbol1, symbol2, window1, window2)
    StockCorrelation.save_relevant_correlations(relevant_results, symbol1, symbol2, window1, window2)
  end

  private

  def log_progress(message)
    puts message unless @silent
  end

  def validate_stock_data!(stock1_data, stock2_data, symbol1, symbol2)
    if stock1_data.empty? || stock2_data.empty?
      error_msg = "No data found for symbols: #{symbol1}(#{stock1_data.length}), #{symbol2}(#{stock2_data.length})"
      Rails.logger.error(error_msg)
      raise AnalysisError, error_msg
    end
  end

  def validate_aligned_data!(aligned_data)
    return if StockDataService.sufficient_data?(aligned_data)
    error_msg = "Insufficient aligned data points: #{aligned_data.length} (need at least 10)"
    Rails.logger.error(error_msg) && raise(AnalysisError, error_msg)
  end

  def build_analysis_result(symbol1, symbol2, window1, window2, offset, start_date, end_date, correlation, aligned_data)
    interpretation = @calculation_service.interpret_correlation(correlation)
    { symbol1: symbol1, symbol2: symbol2, window1: window1, window2: window2, offset: offset,
      start_date: start_date, end_date: end_date, correlation: correlation, 
      r_squared: interpretation[:r_squared], data_points: aligned_data.length,
      aligned_data: aligned_data, interpretation: interpretation, analysis_date: Time.current }
  end

  def calculate_max_theoretical_offset(symbol1, symbol2, start_date, end_date)
    # Get actual data ranges for both symbols
    symbol1_data = StockPrice.where(symbol: symbol1).where.not(close: [0, nil])
    symbol2_data = StockPrice.where(symbol: symbol2).where.not(close: [0, nil])
    
    # Check if we have any data at all
    symbol1_min = symbol1_data.minimum(:date)
    symbol1_max = symbol1_data.maximum(:date)
    symbol2_min = symbol2_data.minimum(:date)
    symbol2_max = symbol2_data.maximum(:date)
    
    return 0 if symbol1_min.nil? || symbol1_max.nil? || symbol2_min.nil? || symbol2_max.nil?
    
    symbol1_total_days = (symbol1_max - symbol1_min).to_i
    symbol2_total_days = (symbol2_max - symbol2_min).to_i
    data_range_days = (end_date - start_date).to_i
    
    # Conservative estimate: use the smaller of the data ranges as upper bound
    [symbol1_total_days, symbol2_total_days, data_range_days].min
  end

  def find_actual_max_viable_offset(symbol1, symbol2, window1, window2, max_theoretical)
    actual_max = 0
    
    (0..max_theoretical).step(50) do |test_offset|
      start_date, end_date = StockDataService.determine_max_date_range(
        symbol1, symbol2, window1, window2, test_offset, silent: true
      )
      
      if start_date && end_date
        actual_max = test_offset
      else
        break
      end
    end
    
    actual_max
  end

  def test_offset_range(symbol1, symbol2, window1, window2, start_date, end_date, max_offset)
    require 'parallel'
    
    log_progress "  📅 Processing #{max_offset + 1} offsets using Parallel gem"
    log_progress "  🚀 Using #{Parallel.processor_count} CPU cores"
    log_progress ""
    
    start_time = Time.now
    
    # Create array of all offsets to process
    offsets = (0..max_offset).to_a
    
    # Process in chunks for progress reporting
    chunk_size = [offsets.length / 10, 100].max # At least 10 progress updates, but not too many for small datasets
    chunks = offsets.each_slice(chunk_size).to_a
    
    log_progress "  🔄 Processing #{offsets.length} offsets in #{chunks.length} chunks for progress tracking..."
    log_progress ""
    
    all_correlations = []
    
    begin
      chunks.each_with_index do |chunk, chunk_index|
        chunk_start_time = Time.now
        
        log_progress "  📊 Processing chunk #{chunk_index + 1}/#{chunks.length} (offsets #{chunk.first}-#{chunk.last})"
        
        # Use Parallel.map for clean, simple parallel processing of this chunk
        chunk_correlations = Parallel.map(chunk, in_processes: Parallel.processor_count) do |current_offset|
          begin
            # Reconnect to database in each process
            ActiveRecord::Base.establish_connection
            
            # Calculate adjusted date range for this offset
            adjusted_start, adjusted_end = StockDataService.determine_max_date_range(
              symbol1, symbol2, window1, window2, current_offset, silent: true
            )
            
            next unless adjusted_start && adjusted_end
            
            # Fetch and process data for this offset
            stock1_data = StockDataService.fetch_stock_data(symbol1, adjusted_start, adjusted_end, window1)
            stock2_data = StockDataService.fetch_stock_data(symbol2, adjusted_start, adjusted_end, window2, current_offset)
            
            next if stock1_data.empty? || stock2_data.empty?
            
            # Calculate percentage changes
            changes1 = StockDataService.calculate_percentage_changes(stock1_data, window1)
            changes2 = StockDataService.calculate_percentage_changes(stock2_data, window2)
            
            # Align data points
            aligned_data = StockDataService.align_data_points(changes1, changes2, current_offset)
            
            next unless StockDataService.sufficient_data?(aligned_data)
            
            # Calculate correlation using inline method
            data_points = aligned_data.map { |change1, change2, _date1, _date2| [change1, change2] }
            correlation = calculate_correlation_inline(data_points)
            
            next if correlation.nil?
            
            # Return simple data structure (no complex objects for serialization)
            {
              offset: current_offset,
              correlation: correlation,
              r_squared: correlation**2,
              data_points: aligned_data.length,
              start_date: adjusted_start,
              end_date: adjusted_end
            }
            
          rescue => e
            Rails.logger.warn "Offset #{current_offset} failed: #{e.message}"
            next # Return nil, will be filtered out
          end
        end
        
        # Filter out nil results and add to total
        valid_chunk_correlations = chunk_correlations.compact
        all_correlations.concat(valid_chunk_correlations)
        
        # Progress reporting
        chunk_time = Time.now - chunk_start_time
        processed_count = (chunk_index + 1) * chunk_size
        processed_count = [processed_count, offsets.length].min # Don't exceed total
        percentage = (processed_count.to_f / offsets.length * 100).round(1)
        
        log_progress "  ✅ Chunk #{chunk_index + 1} completed: #{valid_chunk_correlations.length} correlations in #{chunk_time.round(1)}s"
        log_progress "  📈 Progress: #{processed_count}/#{offsets.length} offsets processed (#{percentage}%)"
        
        # Time estimation for remaining chunks
        if chunk_index < chunks.length - 1
          avg_time_per_chunk = (Time.now - start_time) / (chunk_index + 1)
          remaining_chunks = chunks.length - (chunk_index + 1)
          estimated_remaining = (avg_time_per_chunk * remaining_chunks).round(0)
          log_progress "  ⏱️  Estimated time remaining: #{estimated_remaining}s"
        end
        
        log_progress ""
      end
      
    rescue => e
      Rails.logger.error "Parallel processing failed: #{e.message}"
      return []
    end
    
    total_time = Time.now - start_time
    
    log_progress "  ✅ Parallel processing completed in #{total_time.round(2)}s"
    log_progress "  📊 Results: #{all_correlations.length} successful correlations found out of #{offsets.length} total offsets"
    log_progress "  🖥️  CPU utilization: #{Parallel.processor_count} cores used"
    
    if total_time > 0
      sequential_estimate = (max_offset + 1) * 0.05  # Estimate 50ms per correlation
      speedup_factor = sequential_estimate / total_time
      log_progress "  ⚡ Estimated speedup: #{speedup_factor.round(1)}x faster than sequential"
    end
    
    log_progress ""
    
    all_correlations
  end
  
  private
  
  # Inline correlation calculation for parallel processing
  def calculate_correlation_inline(data_points)
    return nil if data_points.empty?
    return nil if data_points.length < 2
    
    x_values, y_values = data_points.map(&:first), data_points.map(&:second)
    n, mean_x, mean_y = data_points.length, x_values.sum / data_points.length.to_f, y_values.sum / data_points.length.to_f
    
    numerator, sum_sq_x, sum_sq_y = x_values.zip(y_values).reduce([0, 0, 0]) do |(num, sq_x, sq_y), (x, y)|
      diff_x, diff_y = x - mean_x, y - mean_y
      [num + diff_x * diff_y, sq_x + diff_x * diff_x, sq_y + diff_y * diff_y]
    end
    
    denominator = Math.sqrt(sum_sq_x * sum_sq_y)
    denominator == 0 ? nil : numerator / denominator
  end
end