module ApiProviders
  class FmpApiProvider < BaseApiProvider
    BASE_URL = 'https://financialmodelingprep.com/api/v3'

    def fetch_company_info(symbol)
      url = "#{BASE_URL}/profile/#{symbol}?apikey=#{api_key}"
      response = make_request(url)
      
      return response unless response[:success]
      
      data = response[:data]
      if data.nil? || data.empty?
        return { success: false, error: :no_data, message: "No data returned for #{symbol} from FMP" }
      end

      company_data = data.is_a?(Array) ? data.first : data
      normalized_data = normalize_company_data(company_data)
      
      { success: true, data: normalized_data, provider: provider_name }
    end

    protected

    def extract_name(data)
      data['companyName']
    end

    def extract_description(data)
      data['description']
    end

    def extract_sector(data)
      data['sector']
    end

    def extract_industry(data)
      data['industry']
    end

    def extract_market_cap(data)
      return nil unless data['mktCap']
      
      cap = data['mktCap'].to_s.gsub(/[,$]/, '')
      cap.to_i if cap.match?(/^\d+$/)
    end

    def extract_exchange(data)
      data['exchangeShortName']
    end

    def extract_country(data)
      data['country']
    end

    def extract_website(data)
      data['website']
    end

    def extract_employees(data)
      data['fullTimeEmployees']&.to_i
    end
  end
end