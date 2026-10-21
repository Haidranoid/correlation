#!/usr/bin/env ruby

# Simple test script for FMP API integration
# Run with: ruby test_fmp_integration.rb

require_relative 'config/environment'

puts "Testing FMP API Integration"
puts "=" * 40

# Check if API key is set
api_key = ENV['FMP_API_KEY']
if api_key.present?
  puts "✓ FMP_API_KEY is set: #{api_key[0..10]}..."
else
  puts "⚠ FMP_API_KEY not set - will use fallback data"
end

# Test with a known symbol
test_symbol = 'AAPL'
puts "\nTesting with symbol: #{test_symbol}"

begin
  service = StockInfoLookupService.new(test_symbol)
  result = service.call
  
  if result
    puts "✓ Success! Created/found stock info:"
    puts "  Name: #{result.name}"
    puts "  Sector: #{result.sector}" 
    puts "  Industry: #{result.industry}"
    puts "  Exchange: #{result.exchange}"
    puts "  Country: #{result.country}"
    puts "  Market Cap: #{result.market_cap_formatted}" if result.market_cap
    puts "  Website: #{result.website}" if result.website
    puts "  Employees: #{result.employees_formatted}" if result.employees
    puts "  Description: #{result.description[0..100]}..." if result.description
  else
    puts "✗ Failed to create stock info"
  end
  
rescue => e
  puts "✗ Error: #{e.message}"
  puts e.backtrace.first(3)
end

puts "\nCurrent Stock Info Coverage:"
total_symbols = StockPrice.distinct.count(:symbol)
with_info = StockInfo.count
puts "  Total symbols: #{total_symbols}"
puts "  With info: #{with_info}"
puts "  Coverage: #{((with_info.to_f / total_symbols) * 100).round(1)}%"

puts "\n" + "=" * 40
puts "Test completed!"

# Show example usage
puts "\nTo use with real API:"
puts "1. Get free API key from: https://financialmodelingprep.com/developer"
puts "2. Set environment variable: export FMP_API_KEY=your_key_here"
puts "3. Run: LIMIT=10 rails stock_info:batch_update"