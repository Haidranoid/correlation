class StockPrice < ApplicationRecord
  validates :symbol, presence: true, length: { maximum: 10 }
  validates :date, presence: true
  validates :close, presence: true, numericality: { greater_than: 0 }
  validates :symbol, uniqueness: { scope: :date }
  
  before_save :upcase_symbol
  
  belongs_to :stock_info, primary_key: :symbol, foreign_key: :symbol
  
  scope :active, -> { joins(:stock_info).where(stock_infos: { inactive: false }) }
  scope :for_symbol, ->(symbol) { active.where(symbol: symbol.upcase) }
  scope :in_date_range, ->(start_date, end_date) { where(date: start_date..end_date) }
  scope :ordered_by_date, -> { order(:date) }
  
  # Include inactive records when explicitly needed
  scope :include_inactive, -> { unscoped }
  
  def self.available_symbols
    active.distinct.pluck(:symbol).sort
  end
  
  def self.date_range_for_symbol(symbol)
    prices = for_symbol(symbol)
    return [nil, nil] if prices.empty?
    
    [prices.minimum(:date), prices.maximum(:date)]
  end
  
  def to_s
    "#{symbol} - #{date}: $#{close}"
  end
  
  def percentage_change_from(other_price)
    raise ZeroDivisionError if other_price.close == 0
    ((close - other_price.close) / other_price.close.to_f) * 100
  end
  
  # Alias for backward compatibility
  def close_price
    close
  end
  
  def close_price=(value)
    self.close = value
  end
  
  private
  
  def upcase_symbol
    self.symbol = symbol.upcase if symbol
  end
  
  # Class methods for correlation analysis data operations
  class << self
    # Fetch stock data with extended date range for window calculations
    def fetch_for_analysis(symbol, start_date, end_date, window, offset = 0)
      # Extend date range to account for window and offset calculations
      extended_start = start_date - window.days - offset.abs.days - 30.days
      extended_end = end_date + offset.abs.days + 30.days
      
      for_symbol(symbol)
        .where(date: extended_start..extended_end)
        .where.not(close: [0, nil])
        .ordered_by_date
        .pluck(:date, :close)
    end
    
    # Calculate percentage changes over specified time window
    def calculate_percentage_changes(stock_data, window)
      changes = []
      
      stock_data.each_with_index do |(date, close), index|
        next if index < window
        
        previous_close = stock_data[index - window][1]
        next if previous_close.nil? || previous_close == 0
        
        percentage_change = ((close - previous_close) / previous_close.to_f) * 100
        changes << [date, percentage_change]
      end
      
      changes
    end
    
    # Align data points from two different time series with offset
    def align_data_points(changes1, changes2, offset)
      # Create hashes for faster lookup - following Rails performance best practices
      hash1 = changes1.to_h
      hash2 = changes2.to_h
      
      aligned = []
      
      hash1.each do |date1, change1|
        # Calculate the offset date for symbol2
        offset_date = date1 + offset.days
        
        if hash2.key?(offset_date)
          change2 = hash2[offset_date]
          aligned << [change1, change2, date1, offset_date]
        end
      end
      
      aligned
    end
    
    # Determine maximum available date range for two symbols
    def determine_max_date_range(symbol1, symbol2, window1, window2, offset, silent: false)
      Rails.logger.info "Checking data availability for #{symbol1}..." unless silent
      symbol1_data = for_symbol(symbol1).where.not(close: [0, nil])
      symbol1_min = symbol1_data.minimum(:date)
      symbol1_max = symbol1_data.maximum(:date)
      
      Rails.logger.info "Checking data availability for #{symbol2}..." unless silent
      symbol2_data = for_symbol(symbol2).where.not(close: [0, nil])
      symbol2_min = symbol2_data.minimum(:date)
      symbol2_max = symbol2_data.maximum(:date)
      
      if symbol1_min.nil? || symbol1_max.nil?
        Rails.logger.error "No valid data found for #{symbol1}" unless silent
        return [nil, nil]
      end
      
      if symbol2_min.nil? || symbol2_max.nil?
        Rails.logger.error "No valid data found for #{symbol2}" unless silent
        return [nil, nil]
      end
      
      unless silent
        Rails.logger.info "#{symbol1} data range: #{symbol1_min} to #{symbol1_max} (#{(symbol1_max - symbol1_min).to_i} days)"
        Rails.logger.info "#{symbol2} data range: #{symbol2_min} to #{symbol2_max} (#{(symbol2_max - symbol2_min).to_i} days)"
      end
      
      # Calculate the maximum overlapping range, accounting for windows and offset
      # We need enough data before the start for window calculations
      effective_start = [symbol1_min + window1.days, symbol2_min + window2.days].max
      effective_end = [symbol1_max, symbol2_max].min
      
      # Adjust for offset - if we're looking at future data for symbol2, we need to end earlier
      if offset > 0
        effective_end = effective_end - offset.days
      elsif offset < 0
        effective_start = effective_start - offset.days
      end
      
      if effective_start >= effective_end
        unless silent
          Rails.logger.error "No overlapping date range after accounting for windows and offset"
          Rails.logger.error "Effective start: #{effective_start}, Effective end: #{effective_end}"
        end
        return [nil, nil]
      end
      
      unless silent
        Rails.logger.info "Maximum usable range: #{effective_start} to #{effective_end} (#{(effective_end - effective_start).to_i} days)"
      end
      
      [effective_start, effective_end]
    end
    
    # Validate data sufficiency for correlation analysis
    def sufficient_data?(aligned_data, minimum_points: 10)
      aligned_data.length >= minimum_points
    end
    
    # Get data availability summary for symbols
    def data_availability_summary(symbols)
      summary = {}
      
      symbols.each do |symbol|
        data = for_symbol(symbol).where.not(close: [0, nil])
        
        summary[symbol] = {
          total_records: data.count,
          date_range: {
            start: data.minimum(:date),
            end: data.maximum(:date)
          },
          days_available: data.maximum(:date) && data.minimum(:date) ? 
                         (data.maximum(:date) - data.minimum(:date)).to_i : 0
        }
      end
      
      summary
    end
  end
end
