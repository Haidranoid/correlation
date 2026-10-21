class StockApiProviderStatus < ApplicationRecord
  validates :symbol, presence: true
  validates :provider_name, presence: true, uniqueness: { scope: :symbol }
  validates :status, presence: true, inclusion: { in: %w[not_found found rate_limited error] }
  validates :last_checked_at, presence: true

  scope :not_found, -> { where(status: 'not_found') }
  scope :found, -> { where(status: 'found') }
  scope :rate_limited, -> { where(status: 'rate_limited') }
  scope :for_symbol, ->(symbol) { where(symbol: symbol) }
  scope :for_provider, ->(provider) { where(provider_name: provider) }
  
  # Check if a symbol was not found in a specific provider within the last 24 hours
  def self.recently_not_found?(symbol, provider_name)
    where(symbol: symbol, provider_name: provider_name, status: 'not_found')
      .where('last_checked_at > ?', 24.hours.ago)
      .exists?
  end
  
  # Record the status of a symbol for a provider
  def self.record_status(symbol, provider_name, status)
    find_or_initialize_by(symbol: symbol, provider_name: provider_name).tap do |record|
      record.status = status
      record.last_checked_at = Time.current
      record.save!
    end
  end
end
