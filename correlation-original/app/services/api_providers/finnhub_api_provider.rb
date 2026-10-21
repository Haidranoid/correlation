module ApiProviders
  class FinnhubApiProvider < BaseApiProvider
    BASE_URL = 'https://finnhub.io/api/v1'

    def fetch_company_info(symbol)
      # Finnhub requires two API calls for complete data
      profile_response = fetch_company_profile(symbol)
      return profile_response unless profile_response[:success]

      # Get additional company metrics
      metrics_response = fetch_company_metrics(symbol)
      
      # Merge the data
      profile_data = profile_response[:data]
      metrics_data = metrics_response[:success] ? metrics_response[:data] : {}
      
      combined_data = profile_data.merge(metrics_data)
      normalized_data = normalize_company_data(combined_data)
      
      { success: true, data: normalized_data, provider: provider_name }
    end

    private

    def fetch_company_profile(symbol)
      url = "#{BASE_URL}/stock/profile2?symbol=#{symbol}&token=#{api_key}"
      response = make_request(url)
      
      return response unless response[:success]
      
      data = response[:data]
      if data.nil? || data.empty? || data['name'].blank?
        return { success: false, error: :symbol_not_found, message: "Symbol not found in Finnhub" }
      end
      
      response
    end

    def fetch_company_metrics(symbol)
      url = "#{BASE_URL}/stock/metric?symbol=#{symbol}&metric=all&token=#{api_key}"
      make_request(url)
    end

    protected

    def extract_name(data)
      data['name']
    end

    def extract_description(data)
      # Finnhub doesn't provide description in profile endpoint
      nil
    end

    def extract_sector(data)
      data['finnhubIndustry']
    end

    def extract_industry(data)
      data['finnhubIndustry']
    end

    def extract_market_cap(data)
      # Market cap comes from metrics endpoint
      data.dig('metric', 'marketCapitalization')&.to_i
    end

    def extract_exchange(data)
      data['exchange']
    end

    def extract_country(data)
      data['country']
    end

    def extract_website(data)
      data['weburl']
    end

    def extract_employees(data)
      data['shareOutstanding']&.to_i # Finnhub doesn't have employee count
    end
  end
end