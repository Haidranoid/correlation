class CountryMapper
  # Main mapping method that routes to specific mappers based on data type
  def self.map_symbol_to_country(symbol, market, data_type)
    case data_type
    when "bonds" then map_bond_to_country(symbol)
    when "crypto" then "Global"
    when "forex_major", "forex_other" then map_forex_to_countries(symbol)
    when "indices" then map_index_to_country(symbol)
    when "money_market" then map_money_market_to_country(symbol)
    when "stock_indices" then map_stock_index_to_country(symbol)
    else
      MARKET_TO_COUNTRY[market.upcase] || "Unknown"
    end
  end

  private

  # Legacy market code mappings
  MARKET_TO_COUNTRY = {
    "US" => "United States", "DE" => "Germany", "PA" => "France", "L" => "United Kingdom",
    "T" => "Japan", "TO" => "Canada", "AS" => "Netherlands", "SW" => "Switzerland",
    "ST" => "Sweden", "CO" => "Denmark", "HE" => "Finland", "OL" => "Norway",
    "LS" => "Portugal", "MC" => "Spain", "MI" => "Italy", "VI" => "Austria",
    "BR" => "Belgium", "WA" => "Poland", "PR" => "Czech Republic", "BU" => "Hungary",
    "HK" => "Hong Kong", "SS" => "China", "SI" => "Singapore", "KS" => "South Korea",
    "TW" => "Taiwan", "BO" => "India", "SA" => "Brazil", "MX" => "Mexico",
    "BA" => "Argentina", "SN" => "Chile", "IS" => "Turkey", "TA" => "Israel",
    "CA" => "Egypt", "JO" => "South Africa", "AX" => "Australia", "NZ" => "New Zealand",
    "V" => "Austria"
  }.freeze

  # Bond symbol mappings
  BOND_PATTERNS = {
    /10YUSY|30YUSY|10YUSP|30YUSP/ => "United States", /10YBGY|10YCLY|10YCOY|10YCYY/ => "Germany",
    /10YEGY/ => "Egypt", /10YHKY/ => "Hong Kong", /10YIDY/ => "Indonesia", /10YILY/ => "Israel",
    /10YISY/ => "Iceland", /10YKEY/ => "United Kingdom", /10YLKY/ => "Sri Lanka", /10YMAY/ => "Malaysia",
    /10YMTY/ => "Malta", /10YMUY/ => "Mauritius", /10YMYY/ => "Myanmar", /10YNAY/ => "Nigeria",
    /10YNGY/ => "Nicaragua", /10YNZY/ => "New Zealand", /10YPEY/ => "Peru", /10YPHY/ => "Philippines",
    /10YITY|30YITY/ => "Italy", /10YFRY|30YFRY/ => "France", /10YJPY|30YJPY/ => "Japan",
    /10YCAY|30YCAY/ => "Canada", /10YAUY|30YAUY/ => "Australia", /10YCHY|30YCHY/ => "Switzerland",
    /10YSEY|30YSEY/ => "Sweden", /10YNOY|30YNOY/ => "Norway", /10YDKY|30YDKY/ => "Denmark",
    /10YFIY|30YFIY/ => "Finland", /10YBEY|30YBEY/ => "Belgium", /10YATY|30YATY/ => "Austria",
    /10YNLY|30YNLY/ => "Netherlands", /10YESY|30YESY/ => "Spain", /10YPTY|30YPTY/ => "Portugal",
    /10YIEY|30YIEY/ => "Ireland", /10YGRY|30YGRY/ => "Greece", /10YPLY|30YPLY/ => "Poland",
    /10YCZY|30YCZY/ => "Czech Republic", /10YHUY|30YHUY/ => "Hungary", /10YTRY|30YTRY/ => "Turkey",
    /10YBRY|30YBRY/ => "Brazil", /10YMXY|30YMXY/ => "Mexico", /10YKRY|30YKRY/ => "South Korea",
    /10YINY|30YINY/ => "India", /10YCNY|30YCNY/ => "China", /10YRUY|30YRUY/ => "Russia"
  }.freeze

  # Forex symbol mappings (based on currency prefix)
  FOREX_PATTERNS = {
    /^EUR/ => "Eurozone", /^USD/ => "United States", /^GBP/ => "United Kingdom", /^JPY/ => "Japan",
    /^CHF/ => "Switzerland", /^CAD/ => "Canada", /^AUD/ => "Australia", /^NZD/ => "New Zealand",
    /^SEK/ => "Sweden", /^NOK/ => "Norway", /^DKK/ => "Denmark", /^PLN/ => "Poland",
    /^CZK/ => "Czech Republic", /^HUF/ => "Hungary", /^RUB/ => "Russia", /^CNY/ => "China",
    /^INR/ => "India", /^KRW/ => "South Korea", /^SGD/ => "Singapore", /^HKD/ => "Hong Kong",
    /^MXN/ => "Mexico", /^BRL/ => "Brazil", /^ZAR/ => "South Africa", /^TRY/ => "Turkey",
    /^ILS/ => "Israel", /^ISK/ => "Iceland"
  }.freeze

  # Index symbol mappings
  INDEX_PATTERNS = {
    /^MOEX|^MCFTR/ => "Russia", /^OMXR/ => "Sweden", /^SPX/ => "United States",
    /^NASDAQ/ => "United States", /^DAX/ => "Germany", /^FTSE/ => "United Kingdom",
    /^NIKKEI/ => "Japan", /^CAC/ => "France", /^SMI/ => "Switzerland", /^AEX/ => "Netherlands",
    /^IBEX/ => "Spain", /^BEL/ => "Belgium", /^OSLO/ => "Norway", /^STOCKHOLM/ => "Sweden",
    /^HELSINKI/ => "Finland", /^COPENHAGEN/ => "Denmark", /^VIENNA/ => "Austria",
    /^MILAN/ => "Italy", /^WARSAW/ => "Poland", /^PRAGUE/ => "Czech Republic", /^BUDAPEST/ => "Hungary"
  }.freeze

  # Money market symbol mappings
  MONEY_MARKET_PATTERNS = {
    /PLO|PLB/ => "Poland", /CNO|CNB/ => "China", /HUO|HUB/ => "Hungary",
    /EUR/ => "Eurozone", /USD/ => "United States", /GBP/ => "United Kingdom",
    /JPY/ => "Japan", /CHF/ => "Switzerland", /CAD/ => "Canada", /AUD/ => "Australia",
    /SEK/ => "Sweden", /NOK/ => "Norway", /DKK/ => "Denmark", /CZK/ => "Czech Republic", /RUB/ => "Russia"
  }.freeze

  # Stock index suffix mappings
  STOCK_INDEX_PATTERNS = {
    /_PL/ => "Poland", /_DE/ => "Germany", /_US/ => "United States", /_UK/ => "United Kingdom",
    /_FR/ => "France", /_IT/ => "Italy", /_ES/ => "Spain", /_NL/ => "Netherlands",
    /_BE/ => "Belgium", /_AT/ => "Austria", /_CH/ => "Switzerland", /_SE/ => "Sweden",
    /_NO/ => "Norway", /_DK/ => "Denmark", /_FI/ => "Finland", /_JP/ => "Japan",
    /_AU/ => "Australia", /_CA/ => "Canada", /_HK/ => "Hong Kong", /_SG/ => "Singapore",
    /_IN/ => "India", /_CN/ => "China", /_KR/ => "South Korea", /_BR/ => "Brazil",
    /_MX/ => "Mexico", /_RU/ => "Russia"
  }.freeze

  # Individual mapping methods
  def self.map_bond_to_country(symbol)
    BOND_PATTERNS.find { |pattern, _| symbol.match?(pattern) }&.last || "Unknown"
  end

  def self.map_forex_to_countries(symbol)
    FOREX_PATTERNS.find { |pattern, _| symbol.match?(pattern) }&.last || "Global"
  end

  def self.map_index_to_country(symbol)
    INDEX_PATTERNS.find { |pattern, _| symbol.match?(pattern) }&.last || "Unknown"
  end

  def self.map_money_market_to_country(symbol)
    MONEY_MARKET_PATTERNS.find { |pattern, _| symbol.match?(pattern) }&.last || "Unknown"
  end

  def self.map_stock_index_to_country(symbol)
    STOCK_INDEX_PATTERNS.find { |pattern, _| symbol.match?(pattern) }&.last || "Unknown"
  end
end