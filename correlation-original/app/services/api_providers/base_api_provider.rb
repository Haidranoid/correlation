require 'http'

module ApiProviders
  class BaseApiProvider
    attr_reader :api_key, :provider_name

    def initialize(api_key)
      @api_key = api_key
      @provider_name = self.class.name.demodulize.underscore.gsub('_api_provider', '')
    end

    def fetch_company_info(symbol)
      raise NotImplementedError, "Subclasses must implement fetch_company_info"
    end

    protected

    def make_request(url, headers = {})
      response = HTTP.timeout(10).get(url, headers: headers)
      
      case response.status
      when 200
        data = JSON.parse(response.body)
        { success: true, data: data, provider: provider_name }
      when 401
        { success: false, error: :invalid_api_key, message: "Invalid API key for #{provider_name}", should_stop: true }
      when 403
        { success: false, error: :quota_exceeded, message: "API quota exceeded for #{provider_name}", should_stop: true }
      when 429
        { success: false, error: :rate_limit, message: "Rate limit exceeded for #{provider_name}", should_stop: true }
      when 404
        { success: false, error: :symbol_not_found, message: "Symbol not found in #{provider_name}" }
      else
        { success: false, error: :api_error, message: "API error (#{response.status}) from #{provider_name}" }
      end
    rescue HTTP::TimeoutError
      { success: false, error: :timeout, message: "Request timeout for #{provider_name}" }
    rescue JSON::ParserError
      { success: false, error: :invalid_response, message: "Invalid JSON response from #{provider_name}" }
    rescue => e
      { success: false, error: :network_error, message: "Network error for #{provider_name}: #{e.message}" }
    end

    def normalize_company_data(raw_data)
      {
        name: extract_name(raw_data),
        description: extract_description(raw_data),
        sector: extract_sector(raw_data),
        industry: extract_industry(raw_data),
        market_cap: extract_market_cap(raw_data),
        exchange: extract_exchange(raw_data),
        country: extract_country(raw_data),
        website: extract_website(raw_data),
        employees: extract_employees(raw_data),
        provider: provider_name
      }.compact
    end

    # These methods should be overridden by each provider
    def extract_name(data); nil; end
    def extract_description(data); nil; end
    def extract_sector(data); nil; end
    def extract_industry(data); nil; end
    def extract_market_cap(data); nil; end
    def extract_exchange(data); nil; end
    def extract_country(data); nil; end
    def extract_website(data); nil; end
    def extract_employees(data); nil; end
  end
end