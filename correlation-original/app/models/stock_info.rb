class StockInfo < ApplicationRecord
  validates :symbol, presence: true, uniqueness: true
  
  # Associations
  has_many :stock_prices, primary_key: :symbol, foreign_key: :symbol, dependent: :restrict_with_error
  has_many :stock_correlations_as_symbol1, class_name: 'StockCorrelation', primary_key: :symbol, foreign_key: :symbol1
  has_many :stock_correlations_as_symbol2, class_name: 'StockCorrelation', primary_key: :symbol, foreign_key: :symbol2
  
  # Scopes
  scope :active, -> { where(inactive: false) }
  scope :inactive, -> { where(inactive: true) }
  scope :by_sector, ->(sector) { where(sector: sector) }
  scope :by_industry, ->(industry) { where(industry: industry) }
  scope :by_country, ->(country) { where(country: country) }
  
  # Helper methods
  def market_cap_formatted
    return 'N/A' unless market_cap
    
    if market_cap >= 1_000_000_000_000
      "#{(market_cap / 1_000_000_000_000).round(2)}T"
    elsif market_cap >= 1_000_000_000
      "#{(market_cap / 1_000_000_000).round(2)}B"
    elsif market_cap >= 1_000_000
      "#{(market_cap / 1_000_000).round(2)}M"
    else
      market_cap.to_s
    end
  end
  
  def employees_formatted
    return 'N/A' unless employees
    employees.to_s.reverse.gsub(/(\d{3})(?=\d)/, '\\1,').reverse
  end
  
  def age
    return 'N/A' unless founded
    Date.current.year - founded
  end
end