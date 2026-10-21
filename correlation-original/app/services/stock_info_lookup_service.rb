require 'net/http'
require 'json'

class StockInfoLookupService
  def initialize(symbol)
    @symbol = symbol.upcase
    @exhausted_providers = []
  end

  def call
    # Try to get existing record first
    existing_record = StockInfo.find_by(symbol: @symbol)
    
    # If record exists and has real info (not placeholder), return it
    if existing_record&.name.present? && 
       !existing_record.name.include?("Name not available") &&
       !is_placeholder_record?(existing_record)
      return { success: true, stock_info: existing_record, source: :existing }
    end

    # Try to fetch from multiple API providers with fallback
    api_result = fetch_from_apis
    
    # Handle API errors
    if api_result.is_a?(Hash) && api_result[:error]
      case api_result[:error]
      when :all_providers_exhausted
        return { 
          success: false, 
          error: :all_providers_exhausted, 
          message: "All available API providers have been exhausted",
          should_stop: true 
        }
      when :no_providers_available
        return { 
          success: false, 
          error: :no_providers_available, 
          message: "No API providers are configured or available",
          should_stop: true 
        }
      when :all_providers_failed
        return { 
          success: false, 
          error: :all_providers_failed, 
          message: "All providers failed due to temporary errors",
          should_stop: true 
        }
      when :symbol_not_found
        # Symbol not found in all tested providers - create fallback but don't mark globally
        stock_info = create_basic_info
        return { 
          success: true, 
          stock_info: stock_info, 
          source: :fallback_not_found 
        }
      else
        # Network/temporary errors - create fallback but continue
        Rails.logger.warn "Multi-API error for #{@symbol}: #{api_result[:error]} - #{api_result[:message]}"
        stock_info = create_basic_info
        return { 
          success: true, 
          stock_info: stock_info, 
          source: :fallback_error 
        }
      end
    end
    
    # Success - create from API data
    if api_result.is_a?(Hash) && !api_result[:error]
      stock_info = create_from_api_data(api_result)
      return { 
        success: true, 
        stock_info: stock_info, 
        source: :api 
      } if stock_info
    end
    
    # Complete failure - create basic fallback
    Rails.logger.warn "Complete API failure for #{@symbol}, creating basic record"
    stock_info = create_basic_info
    { 
      success: true, 
      stock_info: stock_info, 
      source: :fallback_complete_failure 
    }
  end

  private

  def is_placeholder_record?(record)
    return true if record.description&.include?("This is a placeholder")
    
    # Check if name looks like a generated placeholder
    generated_patterns = [
      "#{record.symbol} Corporation",
      "#{record.symbol} Inc.",
      "#{record.symbol} Company"
    ]
    
    generated_patterns.include?(record.name)
  end

  def create_basic_info
    # Extract some basic info from the symbol and existing stock_prices data
    stock_data = StockPrice.where(symbol: @symbol).first
    
    # Get some basic stats
    market = stock_data&.market
    country = stock_data&.country
    
    # Make educated guesses based on symbol patterns and market data
    name = generate_company_name
    sector = guess_sector
    exchange = guess_exchange(market)
    
    # Create or update the record
    stock_info = StockInfo.find_or_initialize_by(symbol: @symbol)
    
    stock_info.assign_attributes(
      name: name,
      sector: sector,
      exchange: exchange,
      country: country,
      description: "Company information for #{@symbol}. This is a placeholder record that can be updated with more detailed information."
    )
    
    if stock_info.save
      stock_info
    else
      nil
    end
  end

  def generate_company_name
    # Common stock symbol patterns
    case @symbol
    when /^[A-Z]{1,2}$/
      "#{@symbol} Corporation"
    when /^[A-Z]{3}$/
      "#{@symbol} Inc."
    when /^[A-Z]{4,}$/
      "#{@symbol} Company"
    else
      "#{@symbol} (Name to be updated)"
    end
  end

  def guess_sector
    # Basic sector guessing based on common symbol patterns
    tech_patterns = %w[TECH DATA SOFT COMP CYBER CLOUD NET WEB APP]
    finance_patterns = %w[BANK FINA LOAN CRED MORT INSU PAY]
    healthcare_patterns = %w[BIOT MED PHAR HEAL DRUG CARE]
    energy_patterns = %w[OIL GAS ENRG COAL PETRO FUEL]
    
    symbol_base = @symbol.gsub(/[0-9]/, '')
    
    return 'Technology' if tech_patterns.any? { |pattern| symbol_base.include?(pattern) }
    return 'Financials' if finance_patterns.any? { |pattern| symbol_base.include?(pattern) }
    return 'Healthcare' if healthcare_patterns.any? { |pattern| symbol_base.include?(pattern) }
    return 'Energy' if energy_patterns.any? { |pattern| symbol_base.include?(pattern) }
    
    # Default based on symbol length (common patterns)
    case @symbol.length
    when 1..2
      'Industrials'  # Often major corporations
    when 3
      'Technology'   # Many tech companies use 3-letter symbols
    when 4
      'Consumer Discretionary'  # Often consumer brands
    else
      'Miscellaneous'
    end
  end

  def guess_exchange(market)
    return market if market.present?
    
    # Default guessing based on symbol patterns
    case @symbol.length
    when 1..3
      'NYSE'
    when 4..5
      'NASDAQ'
    else
      'OTC'
    end
  end

  def fetch_from_apis
    available_providers = StockApiProviders.available_providers
    
    if available_providers.empty?
      return { error: :no_providers_available, message: "No API providers are configured with valid API keys" }
    end

    # Check which providers are available for this specific symbol
    usable_providers = available_providers.reject do |provider_config|
      provider_name = provider_config[:name]
      
      # Skip if provider is exhausted
      next true if @exhausted_providers.include?(provider_name) || StockApiProviders.provider_exhausted?(provider_name)
      
      # Skip if this symbol was recently not found in this provider
      next true if StockApiProviderStatus.recently_not_found?(@symbol, provider_name.to_s)
      
      false
    end
    
    # If no providers are usable for this symbol, but some providers exist globally, 
    # this is a symbol-specific issue, not a global exhaustion
    if usable_providers.empty?
      global_exhausted = available_providers.all? do |provider_config|
        provider_name = provider_config[:name]
        @exhausted_providers.include?(provider_name) || StockApiProviders.provider_exhausted?(provider_name)
      end
      
      if global_exhausted
        return { error: :all_providers_exhausted, message: "All available API providers have been exhausted" }
      else
        # Some providers exist but all skip this symbol - treat as symbol not found
        return { error: :symbol_not_found, message: "Symbol skipped by all available providers" }
      end
    end

    symbol_not_found_count = 0
    providers_tested = 0
    
    usable_providers.each do |provider_config|
      provider_name = provider_config[:name]
      providers_tested += 1
      
      begin
        Rails.logger.info "Trying #{provider_name} API for #{@symbol}..."
        
        provider = StockApiProviders.get_provider_instance(provider_name)
        next unless provider
        
        result = provider.fetch_company_info(@symbol)
        
        if result[:success]
          Rails.logger.info "✅ Successfully fetched #{@symbol} from #{provider_name}"
          # Record successful find
          StockApiProviderStatus.record_status(@symbol, provider_name.to_s, 'found')
          return result[:data]
        else
          case result[:error]
          when :rate_limit, :quota_exceeded, :invalid_api_key
            # Mark this provider as exhausted
            Rails.logger.warn "🚫 #{provider_name} exhausted for #{@symbol}: #{result[:message]}"
            StockApiProviders.mark_provider_exhausted(provider_name, result[:error])
            StockApiProviderStatus.record_status(@symbol, provider_name.to_s, 'rate_limited')
            @exhausted_providers << provider_name
            
            # Continue to next provider
            next
            
          when :symbol_not_found
            Rails.logger.info "📭 Symbol #{@symbol} not found in #{provider_name}"
            # Record that this provider doesn't have this symbol
            StockApiProviderStatus.record_status(@symbol, provider_name.to_s, 'not_found')
            symbol_not_found_count += 1
            
            # If all tested providers say symbol not found, return that error
            if symbol_not_found_count >= providers_tested
              return { error: :symbol_not_found, message: "Symbol not found in any provider" }
            end
            
            # Continue to next provider
            next
            
          else
            # Temporary/network errors - log and continue to next provider
            Rails.logger.warn "⚠️ Temporary error from #{provider_name} for #{@symbol}: #{result[:message]}"
            StockApiProviderStatus.record_status(@symbol, provider_name.to_s, 'error')
            next
          end
        end
        
      rescue => e
        Rails.logger.error "💥 Error with #{provider_name} for #{@symbol}: #{e.message}"
        next
      end
    end
    
    # If we get here, all providers either failed or are exhausted
    if @exhausted_providers.length >= available_providers.length
      return { error: :all_providers_exhausted, message: "All available API providers have been exhausted" }
    else
      return { error: :all_providers_failed, message: "All providers failed due to temporary errors" }
    end
  end
  
  def create_from_api_data(data)
    return nil unless data.is_a?(Hash)
    
    # Data is already normalized by the API providers
    stock_info = StockInfo.find_or_initialize_by(symbol: @symbol)
    
    stock_info.assign_attributes({
      name: data[:name] || data['name'],
      description: data[:description] || data['description'],
      sector: data[:sector] || data['sector'],
      industry: data[:industry] || data['industry'],
      market_cap: data[:market_cap] || data['market_cap'],
      exchange: data[:exchange] || data['exchange'],
      country: data[:country] || data['country'],
      website: data[:website] || data['website'],
      employees: data[:employees] || data['employees']
    }.compact)
    
    if stock_info.save
      provider_name = data[:provider] || data['provider'] || 'API'
      Rails.logger.info "Successfully created/updated #{@symbol} from #{provider_name}: #{stock_info.name}"
      stock_info
    else
      Rails.logger.error "Failed to save #{@symbol}: #{stock_info.errors.full_messages.join(', ')}"
      nil
    end
  rescue => e
    Rails.logger.error "Error creating stock info for #{@symbol}: #{e.message}"
    nil
  end
  
  def parse_market_cap(mkt_cap)
    return nil unless mkt_cap.present?
    
    # FMP returns market cap as a number
    case mkt_cap
    when Numeric
      mkt_cap.to_f
    when String
      # Handle string format if needed
      mkt_cap.gsub(/[^\d.]/, '').to_f
    else
      nil
    end
  end
end