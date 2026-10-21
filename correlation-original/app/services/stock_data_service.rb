# frozen_string_literal: true

# Service for fetching and processing stock data for correlation analysis
# Handles data retrieval, percentage calculations, and alignment operations
class StockDataService
  class DataError < StandardError; end

  # Fetch stock data with extended date range for window calculations
  # @param symbol [String] Stock symbol
  # @param start_date [Date] Analysis start date
  # @param end_date [Date] Analysis end date  
  # @param window [Integer] Time window for calculations
  # @param offset [Integer] Days to offset (default: 0)
  # @return [Array<Array>] Array of [date, close_price] pairs
  def self.fetch_stock_data(symbol, start_date, end_date, window, offset = 0)
    StockPrice.fetch_for_analysis(symbol, start_date, end_date, window, offset)
  end

  # Calculate percentage changes over specified time window
  # @param stock_data [Array<Array>] Array of [date, price] pairs
  # @param window [Integer] Number of days for percentage change calculation
  # @return [Array<Array>] Array of [date, percentage_change] pairs
  def self.calculate_percentage_changes(stock_data, window)
    StockPrice.calculate_percentage_changes(stock_data, window)
  end

  # Align data points from two different time series with offset
  # @param changes1 [Array<Array>] First dataset [date, change] pairs
  # @param changes2 [Array<Array>] Second dataset [date, change] pairs  
  # @param offset [Integer] Days to offset second dataset
  # @return [Array<Array>] Array of [change1, change2, date1, date2] aligned pairs
  def self.align_data_points(changes1, changes2, offset)
    StockPrice.align_data_points(changes1, changes2, offset)
  end

  # Determine maximum available date range for two symbols
  # @param symbol1 [String] First stock symbol
  # @param symbol2 [String] Second stock symbol
  # @param window1 [Integer] Time window for first symbol
  # @param window2 [Integer] Time window for second symbol
  # @param offset [Integer] Offset in days
  # @param silent [Boolean] Whether to suppress output (default: false)
  # @return [Array<Date, Date>, Array<nil, nil>] Start and end dates or nils if no overlap
  def self.determine_max_date_range(symbol1, symbol2, window1, window2, offset, silent: false)
    StockPrice.determine_max_date_range(symbol1, symbol2, window1, window2, offset, silent: silent)
  end

  # Validate data sufficiency for correlation analysis
  # @param aligned_data [Array] Aligned data points
  # @param minimum_points [Integer] Minimum required data points (default: 10)
  # @return [Boolean] Whether data is sufficient
  def self.sufficient_data?(aligned_data, minimum_points: 10)
    StockPrice.sufficient_data?(aligned_data, minimum_points: minimum_points)
  end

  # Get data availability summary for symbols
  # @param symbols [Array<String>] Array of stock symbols
  # @return [Hash] Summary of data availability for each symbol
  def self.data_availability_summary(symbols)
    StockPrice.data_availability_summary(symbols)
  end

  # Instance methods that delegate to class methods for backwards compatibility
  def get_stock_data(symbol, start_date, end_date)
    StockPrice.for_symbol(symbol)
              .in_date_range(start_date, end_date)
              .ordered_by_date
  end

  def calculate_percentage_changes(stock_data, window)
    return [] if stock_data.length <= window
    
    changes = []
    stock_data.each_with_index do |stock_price, index|
      next if index < window
      
      current_close = stock_price.close
      previous_close = stock_data[index - window].close
      next if previous_close.nil? || previous_close == 0
      
      percentage_change = ((current_close - previous_close) / previous_close.to_f) * 100
      changes << { change: percentage_change, date: stock_price.date }
    end
    
    changes
  end

  def align_data_with_offset(changes1, changes2, offset)
    # Convert to hash for faster lookup
    hash2 = changes2.index_by { |change| change[:date] }
    
    aligned = []
    changes1.each do |change1|
      offset_date = change1[:date] + offset.days
      
      if hash2[offset_date]
        change2 = hash2[offset_date]
        aligned << [change1[:change], change2[:change], change1[:date], offset_date]
      end
    end
    
    aligned
  end
end