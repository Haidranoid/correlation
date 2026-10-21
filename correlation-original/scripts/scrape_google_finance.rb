require 'nokogiri'
require 'selenium-webdriver'

# Function to scrape detailed stock information
def scrape_stock_details(ticker)
  # Set up the Selenium WebDriver (using Chrome in headless mode)
  options = Selenium::WebDriver::Chrome::Options.new
  options.add_argument('--headless') # Run Chrome in headless mode
  driver = Selenium::WebDriver.for :chrome, options: options

  # Navigate to the Google Finance page for the given ticker
  url = "https://www.google.com/finance/quote/#{ticker}:NASDAQ"
  driver.get(url)

  # Wait for the page to load completely
  driver.manage.timeouts.implicit_wait = 10

  # Get the page source after it has fully loaded
  page_source = driver.page_source

  # Parse the HTML response using Nokogiri
  document = Nokogiri::HTML(page_source)

  # Scrape the data by looking for more general patterns
  symbol = ticker
  name = document.at_css('div[data-qa="quote-header"] h1')&.text.strip || 'Not found'
  description = document.at_css('meta[name="description"]')&.[]('content') || 'Not found'
  sector = document.at_css('div[data-qa="sector"]')&.text.strip || 'Not found'
  industry = document.at_css('div[data-qa="industry"]')&.text.strip || 'Not found'

  # Use keywords to locate Market Cap, Exchange, and other data
  market_cap = document.at_css('div[data-qa="market-cap"]')&.text.strip || 'Not found'
  exchange = document.at_css('div[data-qa="exchange"]')&.text.strip || 'Not found'
  country = document.at_css('div[data-qa="country"]')&.text.strip || 'Not found'

  # Improved website extraction targeting abc.xyz domain directly
  website = document.at_css('a[href*="abc.xyz"]')&.text.strip || 'Not found'

  # Search for employees and founded based on nearby keywords
  employees = document.at_css('div[data-qa="employees"]')&.text.strip || 'Not found'
  founded = document.at_css('div[data-qa="founded"]')&.text.strip || 'Not found'

  # Output the results
  puts "Symbol: #{symbol}"
  puts "Name: #{name}"
  puts "Description: #{description}"
  puts "Sector: #{sector}"
  puts "Industry: #{industry}"
  puts "Market Cap: #{market_cap}"
  puts "Exchange: #{exchange}"
  puts "Country: #{country}"
  puts "Website: #{website}"
  puts "Employees: #{employees}"
  puts "Founded: #{founded}"

  # Close the browser after scraping
  driver.quit
end

# Example usage: Scrape the detailed stock information for Google (GOOG)
scrape_stock_details('GOOG')
