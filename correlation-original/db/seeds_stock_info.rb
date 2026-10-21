# Sample stock information data
stock_info_data = [
  {
    symbol: 'AAPL',
    name: 'Apple Inc.',
    description: 'Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. The company also sells various related services.',
    sector: 'Technology',
    industry: 'Consumer Electronics',
    market_cap: 3000000000000.00, # $3T
    exchange: 'NASDAQ',
    country: 'United States',
    website: 'https://www.apple.com',
    employees: 164000,
    founded: 1976
  },
  {
    symbol: 'MSFT',
    name: 'Microsoft Corporation',
    description: 'Microsoft Corporation develops and supports software, services, devices and solutions worldwide. The company operates through Productivity and Business Processes, Intelligent Cloud, and More Personal Computing segments.',
    sector: 'Technology',
    industry: 'Software',
    market_cap: 2800000000000.00, # $2.8T
    exchange: 'NASDAQ',
    country: 'United States',
    website: 'https://www.microsoft.com',
    employees: 221000,
    founded: 1975
  },
  {
    symbol: 'GOOGL',
    name: 'Alphabet Inc.',
    description: 'Alphabet Inc. provides various products and platforms in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America through Google and Other Bets segments.',
    sector: 'Technology',
    industry: 'Internet Content & Information',
    market_cap: 1700000000000.00, # $1.7T
    exchange: 'NASDAQ',
    country: 'United States',
    website: 'https://abc.xyz',
    employees: 182000,
    founded: 1998
  },
  {
    symbol: 'AMZN',
    name: 'Amazon.com Inc.',
    description: 'Amazon.com, Inc. engages in the retail sale of consumer products and subscriptions through online and physical stores. It operates through North America, International, and Amazon Web Services segments.',
    sector: 'Consumer Discretionary',
    industry: 'E-commerce',
    market_cap: 1500000000000.00, # $1.5T
    exchange: 'NASDAQ',
    country: 'United States',
    website: 'https://www.amazon.com',
    employees: 1541000,
    founded: 1994
  },
  {
    symbol: 'TSLA',
    name: 'Tesla Inc.',
    description: 'Tesla, Inc. designs, develops, manufactures, leases, and sells electric vehicles, and energy generation and storage systems in the United States, China, and internationally.',
    sector: 'Consumer Discretionary',
    industry: 'Auto Manufacturers',
    market_cap: 800000000000.00, # $800B
    exchange: 'NASDAQ',
    country: 'United States',
    website: 'https://www.tesla.com',
    employees: 127000,
    founded: 2003
  }
]

# Clear existing data and insert new data
puts "Clearing existing stock info data..."
StockInfo.delete_all

puts "Creating stock info records..."
stock_info_data.each do |info|
  stock_info = StockInfo.find_or_create_by(symbol: info[:symbol]) do |si|
    si.assign_attributes(info)
  end
  puts "Created/Updated: #{stock_info.symbol} - #{stock_info.name}"
end

puts "Stock info seeding completed!"
puts "Total stock info records: #{StockInfo.count}"