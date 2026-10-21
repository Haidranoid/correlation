module ApiProviders
  class IexCloudApiProvider < BaseApiProvider
    BASE_URL = 'https://cloud.iexapis.com/stable'

    def fetch_company_info(symbol)
      url = "#{BASE_URL}/stock/#{symbol}/company?token=#{api_key}"
      response = make_request(url)
      
      return response unless response[:success]
      
      data = response[:data]
      if data.nil? || data.empty?
        return { success: false, error: :symbol_not_found, message: "Symbol not found in IEX Cloud" }
      end

      normalized_data = normalize_company_data(data)
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
      # IEX Cloud doesn't provide market cap in company endpoint
      nil
    end

    def extract_exchange(data)
      data['exchange']
    end

    def extract_country(data)
      data['country']
    end

    def extract_website(data)
      data['website']
    end

    def extract_employees(data)
      data['employees']&.to_i
    end
  end
end