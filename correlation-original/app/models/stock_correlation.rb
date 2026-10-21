class StockCorrelation < ApplicationRecord
  validates :symbol1, :symbol2, presence: true, length: { maximum: 10 }
  validates :window1_days, :window2_days, presence: true, 
            numericality: { greater_than: 0, less_than_or_equal_to: 365 }
  validates :offset_days, presence: true, numericality: { only_integer: true }
  validates :correlation_coefficient, presence: true,
            numericality: { greater_than_or_equal_to: -1.0, less_than_or_equal_to: 1.0 }
  validates :r_squared, presence: true,
            numericality: { greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0 }
  validates :data_points, presence: true, numericality: { greater_than: 0 }
  validates :start_date, :end_date, :analysis_date, presence: true
  
  validate :end_date_after_start_date
  validate :symbols_are_different
  
  before_save :upcase_symbols

  scope :by_symbol_pair, ->(sym1, sym2) { where(symbol1: sym1, symbol2: sym2) }
  scope :by_correlation_strength, -> { order(correlation_coefficient: :desc) }
  scope :strong_correlations, -> { where('ABS(correlation_coefficient) >= 0.7') }
  scope :recent, -> { where('analysis_date >= ?', 30.days.ago) }

  def correlation_strength
    CorrelationCalculationService::CORRELATION_RANGES.find { |range, _| range.cover?(correlation_coefficient.abs) }&.last || "Invalid correlation"
  end

  def correlation_direction
    correlation_coefficient > 0 ? "Positive" : "Negative"
  end

  def analysis_period_days
    (end_date - start_date).to_i
  end

  private

  def end_date_after_start_date
    return unless start_date && end_date
    
    errors.add(:end_date, "must be after start date") if end_date <= start_date
  end

  def symbols_are_different
    return unless symbol1 && symbol2
    
    errors.add(:symbol2, "must be different from symbol1") if symbol1.upcase == symbol2.upcase
  end
  
  def upcase_symbols
    self.symbol1 = symbol1.upcase if symbol1
    self.symbol2 = symbol2.upcase if symbol2
  end
  
  # Class methods for correlation analysis operations
  class << self
    # Create correlation record from analysis result
    def create_from_analysis(result, symbol1, symbol2, window1, window2)
      create!(
        symbol1: symbol1.upcase,
        symbol2: symbol2.upcase,
        window1_days: window1,
        window2_days: window2,
        offset_days: result[:offset],
        correlation_coefficient: result[:correlation],
        r_squared: result[:r_squared],
        start_date: result[:start_date],
        end_date: result[:end_date],
        data_points: result[:data_points],
        analysis_date: Time.current
      )
    end
    
    # Save statistically relevant correlations, replacing existing ones
    def save_relevant_correlations(results, symbol1, symbol2, window1, window2)
      return 0 if results.empty?
      
      # Filter for statistically relevant correlations
      relevant_results = results.select do |result|
        abs_correlation = result[:correlation].abs
        abs_correlation >= 0.3 && result[:r_squared] >= 0.09
      end
      
      return 0 if relevant_results.empty?
      
      # Delete existing correlations for this symbol pair and window combination
      where(
        symbol1: symbol1.upcase,
        symbol2: symbol2.upcase,
        window1_days: window1,
        window2_days: window2
      ).delete_all
      
      # Save new correlations
      saved_count = 0
      relevant_results.each do |result|
        begin
          create_from_analysis(result, symbol1, symbol2, window1, window2)
          saved_count += 1
          Rails.logger.info "Correlation saved (offset: #{result[:offset]}, correlation: #{result[:correlation].round(4)})"
        rescue => e
          Rails.logger.error "Failed to save correlation: #{e.message}"
        end
      end
      
      saved_count
    end
    
    # Check if symbols are the same (case-insensitive)
    def same_symbols?(symbol1, symbol2)
      symbol1.upcase == symbol2.upcase
    end
  end
end