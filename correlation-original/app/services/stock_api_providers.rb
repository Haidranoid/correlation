class StockApiProviders
  PROVIDERS = [
    {
      name: :fmp,
      class_name: 'ApiProviders::FmpApiProvider',
      daily_limit: 250,
      requests_per_minute: 300,
      env_key: 'FMP_API_KEY',
      reset_time: '00:00 UTC'
    },
    {
      name: :alpha_vantage,
      class_name: 'ApiProviders::AlphaVantageApiProvider', 
      daily_limit: 500,
      requests_per_minute: 5,
      env_key: 'ALPHA_VANTAGE_API_KEY',
      reset_time: '00:00 UTC'
    },
    {
      name: :iex_cloud,
      class_name: 'ApiProviders::IexCloudApiProvider',
      monthly_limit: 50000,
      requests_per_minute: 100,
      env_key: 'IEX_CLOUD_API_KEY',
      reset_time: '00:00 UTC'
    },
    {
      name: :finnhub,
      class_name: 'ApiProviders::FinnhubApiProvider',
      requests_per_minute: 60,
      env_key: 'FINNHUB_API_KEY',
      reset_time: '00:00 UTC'
    }
  ].freeze

  class << self
    def available_providers
      PROVIDERS.select { |provider| api_key_present?(provider) }
    end

    def get_provider_instance(provider_name)
      provider_config = PROVIDERS.find { |p| p[:name] == provider_name }
      return nil unless provider_config

      provider_class = provider_config[:class_name].constantize
      provider_class.new(ENV[provider_config[:env_key]])
    rescue NameError
      Rails.logger.warn "Provider class #{provider_config[:class_name]} not found"
      nil
    end

    def next_available_provider(exhausted_providers = [])
      available_providers.find do |provider|
        !exhausted_providers.include?(provider[:name]) && 
        !provider_exhausted?(provider[:name])
      end
    end

    def mark_provider_exhausted(provider_name, reason = 'rate_limit')
      exhausted_file = Rails.root.join('tmp', "api_exhausted_#{provider_name.to_s}")
      data = {
        reason: reason,
        exhausted_at: Time.current.iso8601
      }
      File.write(exhausted_file, data.to_json)
    end

    def provider_exhausted?(provider_name)
      exhausted_file = Rails.root.join('tmp', "api_exhausted_#{provider_name.to_s}")
      return false unless File.exist?(exhausted_file)
      
      # Check if file is less than 24 hours old
      file_age = Time.current - File.mtime(exhausted_file)
      if file_age > 24.hours
        File.delete(exhausted_file)
        return false
      end
      
      true
    end

    def reset_provider_status(provider_name)
      exhausted_file = Rails.root.join('tmp', "api_exhausted_#{provider_name.to_s}")
      File.delete(exhausted_file) if File.exist?(exhausted_file)
    end

    def provider_status
      PROVIDERS.map do |provider|
        {
          name: provider[:name],
          available: api_key_present?(provider),
          exhausted: provider_exhausted?(provider[:name]),
          daily_limit: provider[:daily_limit],
          requests_per_minute: provider[:requests_per_minute]
        }
      end
    end

    private

    def api_key_present?(provider)
      ENV[provider[:env_key]].present?
    end
  end
end