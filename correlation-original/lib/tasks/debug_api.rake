namespace :debug do
  desc "Debug API provider functionality"
  task api_test: :environment do
    puts "🔧 Debugging API Provider System"
    puts "=" * 50
    
    # Test FMP provider directly
    if ENV['FMP_API_KEY'].present?
      puts "📡 Testing FMP API directly..."
      
      begin
        provider = ApiProviders::FmpApiProvider.new(ENV['FMP_API_KEY'])
        puts "Provider created successfully"
        
        result = provider.fetch_company_info('AAPL')
        puts "Result: #{result.inspect}"
      rescue => e
        puts "❌ Error: #{e.message}"
        puts "Backtrace: #{e.backtrace[0..5].join("\n")}"
      end
    else
      puts "❌ FMP_API_KEY not set"
    end
    
    # Test Alpha Vantage if available
    if ENV['ALPHA_VANTAGE_API_KEY'].present?
      puts "\n📡 Testing Alpha Vantage API directly..."
      
      begin
        provider = ApiProviders::AlphaVantageApiProvider.new(ENV['ALPHA_VANTAGE_API_KEY'])
        puts "Provider created successfully"
        
        result = provider.fetch_company_info('AAPL')
        puts "Result: #{result.inspect}"
      rescue => e
        puts "❌ Error: #{e.message}"
        puts "Backtrace: #{e.backtrace[0..5].join("\n")}"
      end
    else
      puts "❌ ALPHA_VANTAGE_API_KEY not set"
    end
    
    puts "\n🌐 Testing HTTP gem directly..."
    begin
      response = HTTP.get("https://httpbin.org/get")
      puts "HTTP test successful: #{response.status}"
    rescue => e
      puts "❌ HTTP error: #{e.message}"
    end
  end
end