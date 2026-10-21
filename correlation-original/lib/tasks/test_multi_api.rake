namespace :test do
  desc "Test multi-API provider system with a known symbol"
  task multi_api: :environment do
    test_symbol = ENV['SYMBOL'] || 'AAPL'
    
    puts "🧪 Testing Multi-API Provider System"
    puts "=" * 50
    puts "Test Symbol: #{test_symbol}"
    puts ""
    
    # Show provider status first
    puts "📊 Available Providers:"
    StockApiProviders.available_providers.each do |provider|
      puts "  ✅ #{provider[:name]} (#{provider[:daily_limit]} daily limit)"
    end
    puts ""
    
    # Test the lookup
    puts "🔍 Testing StockInfoLookupService..."
    service = StockInfoLookupService.new(test_symbol)
    result = service.call
    
    if result[:success]
      stock_info = result[:stock_info]
      puts "✅ SUCCESS!"
      puts "   Source: #{result[:source]}"
      puts "   Name: #{stock_info.name}"
      puts "   Sector: #{stock_info.sector}"
      puts "   Industry: #{stock_info.industry}"
      puts "   Exchange: #{stock_info.exchange}"
      puts "   Country: #{stock_info.country}"
      puts ""
      
      # Show which providers are now exhausted
      exhausted_providers = StockApiProviders::PROVIDERS.select do |provider|
        StockApiProviders.provider_exhausted?(provider[:name])
      end
      
      if exhausted_providers.any?
        puts "🚫 Exhausted Providers:"
        exhausted_providers.each do |provider|
          puts "   #{provider[:name]}"
        end
      else
        puts "🟢 No providers exhausted"
      end
    else
      puts "❌ FAILED!"
      puts "   Error: #{result[:error]}"
      puts "   Message: #{result[:message]}"
    end
    
    puts "\n" + "=" * 50
    puts "Test completed. Use SYMBOL=MSFT to test with different symbol."
  end
end