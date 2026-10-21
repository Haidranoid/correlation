namespace :debug do
  desc "Final debug - step by step cache testing"
  task final: :environment do
    puts "🔍 Final Debug - Step by Step"
    puts "=" * 50
    
    # Step 1: Clear everything
    puts "1️⃣ Clearing all cache..."
    Rails.cache.clear
    
    # Step 2: Mark FMP as exhausted
    puts "2️⃣ Marking FMP as exhausted..."
    StockApiProviders.mark_provider_exhausted(:fmp, 'final_debug')
    
    # Step 3: Check cache directly
    puts "3️⃣ Checking cache directly..."
    cache_key = "stock_api_exhausted_fmp"
    exists = Rails.cache.exist?(cache_key)
    value = Rails.cache.read(cache_key)
    puts "   Cache key: #{cache_key}"
    puts "   Exists: #{exists}"
    puts "   Value: #{value}"
    
    # Step 4: Check provider_exhausted? method
    puts "4️⃣ Checking provider_exhausted? method..."
    is_exhausted_symbol = StockApiProviders.provider_exhausted?(:fmp)
    is_exhausted_string = StockApiProviders.provider_exhausted?('fmp')
    puts "   provider_exhausted?(:fmp): #{is_exhausted_symbol}"
    puts "   provider_exhausted?('fmp'): #{is_exhausted_string}"
    
    # Step 5: Check provider_status method
    puts "5️⃣ Checking provider_status method..."
    status = StockApiProviders.provider_status
    fmp_status = status.find { |p| p[:name] == :fmp }
    puts "   FMP status: #{fmp_status}"
    
    # Step 6: Test the mark and check cycle 10 times
    puts "6️⃣ Testing mark/check cycle 10 times..."
    10.times do |i|
      StockApiProviders.mark_provider_exhausted(:fmp, "test_#{i}")
      is_exhausted = StockApiProviders.provider_exhausted?(:fmp)
      print "#{is_exhausted ? '✅' : '❌'}"
    end
    puts ""
    
    puts "\n✅ Debug complete"
  end
end