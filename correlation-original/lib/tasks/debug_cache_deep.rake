namespace :debug do
  desc "Deep debug of cache keys and provider exhaustion"
  task cache_deep: :environment do
    puts "🔍 Deep Cache Debug"
    puts "=" * 50
    
    # Check what happens when we mark providers
    puts "📝 Marking FMP as exhausted..."
    StockApiProviders.mark_provider_exhausted(:fmp, 'deep_debug_test')
    
    # Check all possible cache key variations
    possible_keys = [
      "stock_api_exhausted_fmp",
      "stock_api_exhausted_:fmp", 
      "stock_api_exhausted_#{:fmp}",
      "stock_api_exhausted_#{:fmp.to_s}"
    ]
    
    puts "\n🔍 Checking all possible cache key variations:"
    possible_keys.each do |key|
      value = Rails.cache.read(key)
      exists = Rails.cache.exist?(key)
      puts "  #{key.ljust(35)}: exists=#{exists}, value=#{value}"
    end
    
    # Test the actual method call
    puts "\n🧪 Testing provider_exhausted? method:"
    is_exhausted = StockApiProviders.provider_exhausted?(:fmp)
    puts "  provider_exhausted?(:fmp): #{is_exhausted}"
    
    is_exhausted_str = StockApiProviders.provider_exhausted?('fmp')
    puts "  provider_exhausted?('fmp'): #{is_exhausted_str}"
    
    # Show the exact cache key being used
    cache_key = "stock_api_exhausted_#{:fmp.to_s}"
    puts "\n📋 Cache key being used: '#{cache_key}'"
    puts "   Cache exists?: #{Rails.cache.exist?(cache_key)}"
    puts "   Cache value: #{Rails.cache.read(cache_key)}"
    
    # Clean up
    Rails.cache.delete("stock_api_exhausted_fmp")
  end
end