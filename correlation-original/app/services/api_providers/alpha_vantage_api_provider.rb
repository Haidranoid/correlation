module ApiProviders
  class AlphaVantageApiProvider < BaseApiProvider
    BASE_URL = 'https://www.alphavantage.co/query'

    def fetch_company_info(symbol)
      url = "#{BASE_URL}?function=OVERVIEW&symbol=#{symbol}&apikey=#{api_key}"
      response = make_request(url)
      
      return response unless response[:success]
      
      data = response[:data]
      
      # Check for rate limit responses first
      if data&.dig('Information')&.include?('call frequency')
        return { success: false, error: :rate_limit, message: "Rate limit exceeded for Alpha Vantage", should_stop: true }
      end

      if data&.dig('Note')&.include?('Thank you for using Alpha Vantage')
        return { success: false, error: :rate_limit, message: "Rate limit exceeded for Alpha Vantage", should_stop: true }
      end

      # Empty response or missing symbol means symbol not found
      if data.nil? || data.empty? || data['Symbol'].blank?
        return { success: false, error: :symbol_not_found, message: "Symbol not found in Alpha Vantage" }
      end

      normalized_data = normalize_company_data(data)
      { success: true, data: normalized_data, provider: provider_name }
    end

    protected

    def extract_name(data)
      data['Name']
    end

    def extract_description(data)
      data['Description']
    end

    def extract_sector(data)
      data['Sector']
    end

    def extract_industry(data)
      data['Industry']
    end

    def extract_market_cap(data)
      return nil unless data['MarketCapitalization']
      
      cap = data['MarketCapitalization'].to_s.gsub(/[,$]/, '')
      cap.to_i if cap.match?(/^\d+$/)
    end

    def extract_exchange(data)
      data['Exchange']
    end

    def extract_country(data)
      data['Country']
    end

    def extract_website(data)
      data['OfficialSite']
    end

    def extract_employees(data)
      data['FullTimeEmployees']&.to_i
    end
  end
end