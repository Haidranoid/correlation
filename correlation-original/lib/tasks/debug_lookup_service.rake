namespace :debug do
  desc "Debug the StockInfoLookupService step by step"
  task lookup_service: :environment do
    test_symbol = ENV['SYMBOL'] || 'NEWTEST'
    
    puts "🔍 Debugging StockInfoLookupService for #{test_symbol}"
    puts "=" * 50
    
    service = StockInfoLookupService.new(test_symbol)
    
    # Test each provider individually
    puts "1️⃣ Testing each provider individually:"
    StockApiProviders.available_providers.each do |provider_config|
      provider_name = provider_config[:name]
      
      begin
        provider = StockApiProviders.get_provider_instance(provider_name)
        if provider
          puts "\n🧪 Testing #{provider_name}..."
          result = provider.fetch_company_info(test_symbol)
          puts "   Success: #{result[:success]}"
          puts "   Error: #{result[:error]}" if result[:error]
          puts "   Message: #{result[:message]}" if result[:message]
          puts "   Should stop: #{result[:should_stop]}" if result.key?(:should_stop)
        else
          puts "\n❌ Could not create #{provider_name} provider instance"
        end
      rescue => e
        puts "\n💥 Exception testing #{provider_name}: #{e.message}"
      end
    end
    
    puts "\n2️⃣ Testing the full lookup service:"
    result = service.call
    puts "   Success: #{result[:success]}"
    puts "   Error: #{result[:error]}" if result[:error] 
    puts "   Message: #{result[:message]}" if result[:message]
    puts "   Should stop: #{result[:should_stop]}" if result.key?(:should_stop)
    puts "   Source: #{result[:source]}" if result[:source]
    
    puts "\n3️⃣ Checking exhausted providers after test:"
    StockApiProviders::PROVIDERS.each do |provider_config|
      provider_name = provider_config[:name]
      is_exhausted = StockApiProviders.provider_exhausted?(provider_name)
      puts "   #{provider_name}: #{is_exhausted ? 'Exhausted' : 'Available'}"
    end
  end
end