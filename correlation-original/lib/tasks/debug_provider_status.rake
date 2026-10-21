namespace :debug do
  desc "Debug provider_status method specifically"
  task provider_status_debug: :environment do
    puts "🔍 Provider Status Debug"
    puts "=" * 50
    
    # Mark FMP as exhausted first
    puts "📝 Marking FMP as exhausted..."
    StockApiProviders.mark_provider_exhausted(:fmp, 'status_debug_test')
    
    # Test the provider_exhausted? method directly
    puts "\n🧪 Testing provider_exhausted? method directly:"
    StockApiProviders::PROVIDERS.each do |provider|
      provider_name = provider[:name]
      is_exhausted = StockApiProviders.provider_exhausted?(provider_name)
      puts "  #{provider_name}: #{is_exhausted}"
    end
    
    # Test the provider_status method
    puts "\n📊 Testing provider_status method:"
    status_array = StockApiProviders.provider_status
    status_array.each do |provider|
      puts "  #{provider[:name]}: available=#{provider[:available]}, exhausted=#{provider[:exhausted]}"
    end
    
    # Clean up
    Rails.cache.delete("stock_api_exhausted_fmp")
  end
end