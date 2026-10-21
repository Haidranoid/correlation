namespace :stock_info do
  
  desc "Show status of all API providers"
  task provider_status: :environment do
    puts "🔍 API Provider Status Report"
    puts "=" * 50
    puts "Checking actual API status..."
    puts ""
    
    StockApiProviders.provider_status.each do |provider|
      provider_name = provider[:name]
      print "#{provider_name.to_s.upcase.ljust(15)}: "
      
      if !provider[:available]
        puts "❌ No API key configured"
        puts "   Daily Limit: #{provider[:daily_limit] || 'N/A'}"
        puts "   Rate Limit: #{provider[:requests_per_minute]} req/min"
        puts ""
        next
      end
      
      # Check exhaustion file first
      if StockApiProviders.provider_exhausted?(provider_name)
        puts "🚫 Exhausted (file check)"
        puts "   Status: Exhausted"
      else
        # Test the actual API
        begin
          api_provider = StockApiProviders.get_provider_instance(provider_name)
          if api_provider
            # Test with a simple symbol that should exist
            test_result = api_provider.fetch_company_info('MSFT')
            
            case test_result[:error]
            when :rate_limit, :quota_exceeded
              puts "🚫 Rate limited/Quota exceeded"
              puts "   Status: Exhausted"
              # Mark as exhausted if not already
              StockApiProviders.mark_provider_exhausted(provider_name, test_result[:error])
            when :invalid_api_key
              puts "❌ Invalid API key"
              puts "   Status: API key issue"
            when :symbol_not_found
              puts "✅ Working (symbol not found is normal for test)"
              puts "   Status: Available"
            when nil
              if test_result[:success]
                puts "✅ Working"
                puts "   Status: Available"
              else
                puts "⚠️  Unknown response"
                puts "   Status: Unknown"
              end
            else
              puts "⚠️  #{test_result[:error]}"
              puts "   Status: Temporary issue"
            end
          else
            puts "❌ Provider class not found"
            puts "   Status: Configuration error"
          end
        rescue => e
          puts "💥 Error: #{e.message}"
          puts "   Status: Connection failed"
        end
      end
      
      # Add delay between provider tests to respect rate limits
      sleep(1)
      
      puts "   Exhausted file: #{StockApiProviders.provider_exhausted?(provider_name) ? 'Yes' : 'No'}"
      puts "   Daily Limit: #{provider[:daily_limit] || 'N/A'}"
      puts "   Rate Limit: #{provider[:requests_per_minute]} req/min"
      puts ""
    end
    
    available_count = StockApiProviders.available_providers.count
    total_count = StockApiProviders::PROVIDERS.count
    
    puts "Summary: #{available_count}/#{total_count} providers available"
    
    if available_count == 0
      puts ""
      puts "⚠️  No API providers are configured!"
      puts "Set up at least one API key:"
      StockApiProviders::PROVIDERS.each do |provider|
        puts "  export #{provider[:env_key]}=your_key_here"
      end
    end
  end
  
  desc "Reset exhausted provider status (for testing)"
  task reset_providers: :environment do
    StockApiProviders::PROVIDERS.each do |provider|
      StockApiProviders.reset_provider_status(provider[:name])
    end
    puts "✅ All provider statuses reset"
  end
  
  desc "Show provider-specific symbol statuses"
  task provider_symbol_status: :environment do
    puts "🔍 Provider-Specific Symbol Status Report"
    puts "=" * 60
    
    StockApiProviders::PROVIDERS.each do |provider|
      provider_name = provider[:name].to_s
      
      puts "\n#{provider_name.upcase}:"
      puts "-" * 20
      
      statuses = StockApiProviderStatus.for_provider(provider_name)
      
      %w[found not_found rate_limited error].each do |status|
        count = statuses.where(status: status).count
        next if count == 0
        
        icon = case status
               when 'found' then '✅'
               when 'not_found' then '📭'
               when 'rate_limited' then '🚫'
               when 'error' then '⚠️'
               end
        
        puts "  #{icon} #{status.humanize}: #{count} symbols"
        
        # Show recent examples for 'not_found'
        if status == 'not_found' && count > 0
          examples = statuses.where(status: status).limit(5).pluck(:symbol)
          puts "    Examples: #{examples.join(', ')}"
        end
      end
      
      total = statuses.count
      puts "  Total tracked: #{total} symbols" if total > 0
    end
    
    puts "\n" + "=" * 60
    puts "Note: Symbols are only tracked after being tested with each provider"
  end
  
  desc "Refresh existing stock info data"
  task refresh_existing: :environment do
    puts "Refreshing existing stock info records..."
    
    existing_symbols = StockInfo.pluck(:symbol)
    puts "Found #{existing_symbols.count} existing records to refresh"
    
    success_count = 0
    error_count = 0
    
    existing_symbols.each_with_index do |symbol, index|
      begin
        result = StockInfoLookupService.new(symbol).call
        
        if result[:success]
          stock_info = result[:stock_info]
          puts "✓ [#{index + 1}/#{existing_symbols.count}] Refreshed #{symbol}: #{stock_info.name}"
          success_count += 1
        else
          puts "⚠ [#{index + 1}/#{existing_symbols.count}] Error for #{symbol}: #{result[:error]}"
          error_count += 1
        end
      rescue => e
        puts "✗ [#{index + 1}/#{existing_symbols.count}] Error refreshing #{symbol}: #{e.message}"
        error_count += 1
      end
      
      # Small delay between requests
      sleep(0.5)
      
      # Longer delay every 10 requests
      sleep(2) if (index + 1) % 10 == 0
    end
    
    puts "\n" + "="*50
    puts "Stock info refresh completed!"
    puts "Successfully refreshed: #{success_count}"
    puts "Errors/No data: #{error_count}"
    puts "Total processed: #{existing_symbols.count}"
    puts "="*50
  end
  
  
  desc "Update with Multi-API providers - Smart resumable version (specify LIMIT=100)"
  task update_with_api: :environment do
    # Check if any providers are available
    available_providers = StockApiProviders.available_providers
    if available_providers.empty?
      puts "❌ No API providers are configured!"
      puts "Set up at least one API key:"
      StockApiProviders::PROVIDERS.each do |provider|
        puts "  export #{provider[:env_key]}=your_key_here"
      end
      next
    end
    
    limit = ENV['LIMIT']&.to_i
    if limit
      puts "🚀 Updating stock info using Multi-API providers for #{limit} symbols..."
    else
      puts "🚀 Updating stock info using Multi-API providers for all incomplete symbols..."
    end
    
    puts "Available providers: #{available_providers.map { |p| p[:name] }.join(', ')}"
    
    # Find symbols that need processing - prioritize missing entirely, then placeholders
    all_symbols = StockPrice.active.distinct.pluck(:symbol)
    
    # Symbols with no stock_info at all
    existing_symbols = StockInfo.pluck(:symbol)
    missing_symbols = all_symbols - existing_symbols
    
    # Symbols with placeholder data (fallback records) - be very precise
    placeholder_symbols = StockInfo.where(
      "description LIKE ? OR description LIKE ?",
      "%placeholder%", "%This is a placeholder%"
    ).select do |record|
      # Skip if it has real company description (not a placeholder)
      next false if record.description.present? && 
                   !record.description.include?("placeholder") &&
                   !record.description.include?("This is a placeholder") &&
                   record.description.length > 100
      
      # Check if it's explicitly a placeholder by description
      next true if record.description&.include?("This is a placeholder")
      next true if record.description&.include?("placeholder record")
      
      # Check if name looks like a generated placeholder (exact match only)
      generated_patterns = [
        "#{record.symbol} Corporation",
        "#{record.symbol} Inc.",
        "#{record.symbol} Company",
        "#{record.symbol} (Name to be updated)"
      ]
      generated_patterns.include?(record.name)
    end.map(&:symbol)
    
    # Prioritize missing symbols, then placeholders by likelihood of success
    all_candidates = (missing_symbols + placeholder_symbols).uniq
    
    # Sort symbols by priority (most likely to be found first)
    symbols_to_process = all_candidates.sort_by do |symbol|
      case symbol
      when /^[A-Z]{1,4}$/      # High priority: 1-4 letters
        [0, symbol]
      when /^[A-Z]{5,}$/       # Medium priority: 5+ letters
        [1, symbol]
      when /^[A-Z]+_[A-Z]$/    # Low priority: preferred shares, etc
        [2, symbol]
      else                     # Very low priority: complex patterns
        [3, symbol]
      end
    end.map(&:last)
    
    # Filter out symbols that have no usable providers
    puts "🔍 Filtering symbols with no usable providers..."
    puts "DEBUG: First 10 symbols before filtering: #{symbols_to_process.first(10).join(', ')}"
    
    symbols_with_usable_providers = []
    symbols_to_process.each do |symbol|
      has_usable_provider = false
      
      available_providers.each do |provider_config|
        provider_name = provider_config[:name]
        
        # Skip if provider is exhausted
        next if StockApiProviders.provider_exhausted?(provider_name)
        
        # Skip if this symbol was recently not found in this provider
        next if StockApiProviderStatus.recently_not_found?(symbol, provider_name.to_s)
        
        has_usable_provider = true
        break
      end
      
      symbols_with_usable_providers << symbol if has_usable_provider
    end
    
    filtered_count = symbols_to_process.count - symbols_with_usable_providers.count
    puts "  Filtered out #{filtered_count} symbols with no usable providers"
    puts "DEBUG: First 10 symbols after filtering: #{symbols_with_usable_providers.first(10).join(', ')}"
    
    symbols_to_process = symbols_with_usable_providers
    if limit
      symbols_to_process = symbols_to_process.first(limit)
    end
    
    # Count provider-specific statuses
    provider_stats = {}
    StockApiProviders::PROVIDERS.each do |provider|
      provider_name = provider[:name].to_s
      not_found_count = StockApiProviderStatus.for_provider(provider_name).not_found.count
      provider_stats[provider_name] = not_found_count
    end
    
    puts "🔍 Analysis:"
    puts "  Total active symbols: #{all_symbols.count}"
    puts "  Missing stock_info entirely: #{missing_symbols.count}" 
    puts "  Has placeholder data: #{placeholder_symbols.count}"
    puts "  Provider 'not found' counts:"
    provider_stats.each do |provider, count|
      puts "    #{provider}: #{count} symbols"
    end
    puts "  Symbols to process this run: #{symbols_to_process.count}"
    
    if symbols_to_process.empty?
      puts "✅ No symbols need processing!"
      next
    end
    
    puts "Processing #{symbols_to_process.count} symbols with FMP API..."
    puts
    
    success_count = 0
    api_success = 0
    fallback_count = 0
    error_count = 0
    api_stopped = false
    stop_reason = nil
    last_processed_index = 0
    
    symbols_to_process.each_with_index do |symbol, index|
      last_processed_index = index
      
      begin
        print "[#{index + 1}/#{symbols_to_process.count}] #{symbol}... "
        
        # Call the enhanced service
        result = StockInfoLookupService.new(symbol).call
        
        if result[:success]
          case result[:source]
          when :api
            puts "🎯 #{result[:stock_info].name} (API)"
            api_success += 1
          when :existing
            puts "✅ #{result[:stock_info].name} (existing)"
          when :fallback_not_found
            puts "📝 #{result[:stock_info].name} (not found in tested providers)"
            fallback_count += 1
          when :fallback_error, :fallback_complete_failure
            puts "📝 #{result[:stock_info].name} (fallback)"
            fallback_count += 1
          end
          success_count += 1
        else
          # Handle different types of failures
          if result[:should_stop]
            case result[:error]
            when :all_providers_exhausted
              puts "🛑 All API providers exhausted"
              api_stopped = true
              stop_reason = "All API providers have hit their rate limits"
            when :no_providers_available
              puts "🛑 No API providers available"
              api_stopped = true  
              stop_reason = "No API providers are configured"
            when :all_providers_failed
              puts "🛑 All API providers failed"
              api_stopped = true
              stop_reason = "All providers failed due to temporary errors"
            else
              puts "🛑 #{result[:error]}: #{result[:message]}"
              api_stopped = true
              stop_reason = result[:message]
            end
            break
          else
            puts "❌ #{result[:error]}: #{result[:message]}"
            error_count += 1
          end
        end
        
      rescue => e
        puts "💥 Unexpected error: #{e.message}"
        error_count += 1
        Rails.logger.error "Unexpected error processing #{symbol}: #{e.message}\n#{e.backtrace.join("\n")}"
      end
      
      # Respectful delay for API
      sleep(0.3)
      
      # Progress update every 20 records
      if (index + 1) % 20 == 0
        puts "\n📊 Progress Update:"
        puts "   API Success: #{api_success}"
        puts "   Fallbacks: #{fallback_count}" 
        puts "   Errors: #{error_count}"
        puts "   Completed: #{index + 1}/#{symbols_to_process.count}"
        puts ""
        sleep(2) # Longer pause for progress
      end
    end
    
    puts
    if api_stopped
      puts "🛑" + "="*48
      puts "API IMPORT STOPPED DUE TO LIMITS"
      puts "="*50
      puts "Stop reason: #{stop_reason}"
      puts "Processed: #{last_processed_index + 1}/#{symbols_to_process.count} symbols"
      puts "Successfully processed: #{success_count}"
      puts "  - Real API data: #{api_success}"
      puts "  - Fallback data: #{fallback_count}"
      puts "Errors: #{error_count}"
      puts
      puts "💡 TO RESUME:"
      puts "Wait for your API quotas to reset (usually daily for most providers)"
      puts "You can check provider status with: rake stock_info:provider_status"
      puts "Then run the same command again - it will automatically"
      puts "continue from where it left off by processing remaining symbols."
      puts "Existing placeholder records will be replaced with real data."
      puts
      puts "Current status:"
      active_symbols = StockPrice.active.distinct.pluck(:symbol)
      existing_info_for_active = StockInfo.where(symbol: active_symbols).count
      remaining_missing = active_symbols.count - existing_info_for_active
      # Calculate remaining placeholders using same logic as symbol selection
      remaining_placeholders = StockInfo.where(
        "description LIKE ? OR description LIKE ? OR name LIKE ? OR name LIKE ? OR name LIKE ? OR name LIKE ?",
        "%placeholder%", "%This is a placeholder%", "%(Name not available)",
        "%Corporation", "%Inc.", "%Company"
      ).select do |record|
        next true if record.description&.include?("This is a placeholder")
        generated_patterns = [
          "#{record.symbol} Corporation",
          "#{record.symbol} Inc.",
          "#{record.symbol} Company"
        ]
        generated_patterns.include?(record.name)
      end.count
      puts "  - Missing stock_info entirely: #{remaining_missing}"
      puts "  - With placeholder data: #{remaining_placeholders}"
    else
      puts "🎉" + "="*48
      puts "MULTI-API UPDATE COMPLETED!"
      puts "Successfully processed: #{success_count}"
      puts "  - Real API data: #{api_success}"
      puts "  - Fallback data: #{fallback_count}"
      puts "Errors: #{error_count}"
      
      active_symbols = StockPrice.active.distinct.pluck(:symbol)
      existing_info_for_active = StockInfo.where(symbol: active_symbols).count
      remaining_missing = active_symbols.count - existing_info_for_active
      # Calculate remaining placeholders using same logic as symbol selection
      remaining_placeholders = StockInfo.where(
        "description LIKE ? OR description LIKE ? OR name LIKE ? OR name LIKE ? OR name LIKE ? OR name LIKE ?",
        "%placeholder%", "%This is a placeholder%", "%(Name not available)",
        "%Corporation", "%Inc.", "%Company"
      ).select do |record|
        next true if record.description&.include?("This is a placeholder")
        generated_patterns = [
          "#{record.symbol} Corporation",
          "#{record.symbol} Inc.",
          "#{record.symbol} Company"
        ]
        generated_patterns.include?(record.name)
      end.count
      puts "Remaining work:"
      puts "  - Missing stock_info entirely: #{remaining_missing}"
      puts "  - With placeholder data: #{remaining_placeholders}"
    end
    puts "="*50
    
    if api_success > 0
      puts "\n🔍 Sample successful API results:"
      StockInfo.where.not(description: nil)
               .where.not("description LIKE ?", "%placeholder%")
               .limit(3).each do |stock|
        puts "  #{stock.symbol}: #{stock.name} (#{stock.sector})"
      end
    end
  end
end