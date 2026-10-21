class HomeController < ApplicationController
  def index
    @stats = {
      total_stocks: StockPrice.active.distinct.count(:symbol),
      total_records: StockPrice.active.count,
      total_correlations: StockCorrelation.count,
      date_range: {
        start: StockPrice.active.minimum(:date),
        end: StockPrice.active.maximum(:date)
      },
      countries: StockPrice.active.distinct.count(:country),
      markets: StockPrice.active.distinct.count(:market),
      strongest_correlation: StockCorrelation.order(correlation_coefficient: :desc).first,
      recent_analysis: StockCorrelation.order(created_at: :desc).limit(5)
    }
  end

  def show
    @symbol = params[:symbol].upcase
    @stock_data = StockPrice.where(symbol: @symbol).order(:date)
    
    if @stock_data.empty?
      redirect_to root_path, alert: "Stock symbol #{@symbol} not found"
      return
    end

    # Get stock information
    @stock_info = StockInfo.find_by(symbol: @symbol)

    @stock_stats = {
      symbol: @symbol,
      total_records: @stock_data.count,
      date_range: {
        start: @stock_data.minimum(:date),
        end: @stock_data.maximum(:date)
      },
      price_range: {
        min: @stock_data.minimum(:close),
        max: @stock_data.maximum(:close),
        current: @stock_data.last&.close
      },
      market: @stock_data.first&.market,
      country: @stock_data.first&.country,
      correlations_as_symbol1: StockCorrelation.where(symbol1: @symbol).count,
      correlations_as_symbol2: StockCorrelation.where(symbol2: @symbol).count
    }

    @recent_prices = @stock_data.limit(30).reverse
    @top_correlations = StockCorrelation.where('symbol1 = ? OR symbol2 = ?', @symbol, @symbol)
                                       .order(correlation_coefficient: :desc)
                                       .limit(10)
  end

  def stocks_data
    # Use a single optimized query with aggregations (active symbols only)
    stocks = StockPrice.active.select(
      'stock_prices.symbol',
      'MAX(stock_prices.market) as market',
      'MAX(stock_prices.country) as country', 
      'COUNT(*) as record_count',
      'MIN(date) as start_date',
      'MAX(date) as end_date',
      'MIN(close) as min_price',
      'MAX(close) as max_price'
    ).group('stock_prices.symbol')
     .order('stock_prices.symbol')

    data = stocks.map do |stock|
      {
        symbol: stock.symbol,
        market: stock.market || 'N/A',
        country: stock.country || 'N/A',
        records: stock.record_count,
        date_range: "#{stock.start_date} to #{stock.end_date}",
        price_range: "#{stock.min_price&.round(2)} - #{stock.max_price&.round(2)}"
      }
    end

    render json: { data: data }
  end

  def chart_data
    # Get the earliest date for each active stock symbol first
    earliest_dates = StockPrice.active.select('stock_prices.symbol, MIN(date) as earliest_date')
                               .group('stock_prices.symbol')

    # Then get stocks count over time (by earliest listing year for each stock)
    # Use unscoped to avoid applying the active scope to the outer query
    stocks_over_time = StockPrice.unscoped.from("(#{earliest_dates.to_sql}) as earliest")
                                 .select('EXTRACT(YEAR FROM earliest_date) as year, COUNT(*) as stock_count')
                                 .group('EXTRACT(YEAR FROM earliest_date)')
                                 .order('year')

    # Get correlations count by offset days (include all correlations for historical context)
    correlations_by_offset = StockCorrelation.select(
      'offset_days',
      'COUNT(*) as correlation_count'
    ).group('offset_days')
     .order('offset_days')

    # Create year range from earliest stock to current year
    min_year = stocks_over_time.first&.year&.to_i || Date.current.year
    max_year = Date.current.year
    year_range = (min_year..max_year).to_a

    # Build cumulative stock counts
    stocks_cumulative = {}
    cumulative_count = 0
    stocks_over_time.each do |row|
      cumulative_count += row.stock_count
      stocks_cumulative[row.year.to_i] = cumulative_count
    end

    # Fill in missing years for stocks (carry forward last known count)
    stocks_data = year_range.map do |year|
      if stocks_cumulative[year]
        cumulative_count = stocks_cumulative[year]
      end
      [year, cumulative_count]
    end

    # Build correlations data by offset days
    correlations_data = correlations_by_offset.map do |row|
      [row.offset_days, row.correlation_count]
    end

    render json: {
      stocks: stocks_data,
      correlations: correlations_data
    }
  end

  def price_data
    symbol = params[:symbol].upcase
    
    # Get price data for the last 2 years or all data if less than 2 years
    price_data = StockPrice.where(symbol: symbol)
                          .where('date >= ?', 2.years.ago)
                          .order(:date)
                          .pluck(:date, :close)

    # If less than 500 points in 2 years, get all data
    if price_data.length < 500
      price_data = StockPrice.where(symbol: symbol)
                            .order(:date)
                            .pluck(:date, :close)
    end

    # Convert to timestamps and prices for uPlot
    chart_data = price_data.map do |date, price|
      [date.to_time.to_i, price.to_f]
    end

    render json: {
      data: chart_data,
      symbol: symbol,
      count: chart_data.length
    }
  end

  def recent_price_data
    symbol = params[:symbol].upcase
    
    # Get last 30 days of price data
    recent_data = StockPrice.where(symbol: symbol)
                           .where('date >= ?', 30.days.ago)
                           .order(:date)
                           .pluck(:date, :close)

    # If less than 10 points in 30 days, get last 30 records
    if recent_data.length < 10
      recent_data = StockPrice.where(symbol: symbol)
                             .order(:date)
                             .limit(30)
                             .pluck(:date, :close)
    end

    # Convert to timestamps and prices for uPlot
    chart_data = recent_data.map do |date, price|
      [date.to_time.to_i, price.to_f]
    end

    render json: {
      data: chart_data,
      symbol: symbol,
      count: chart_data.length
    }
  end
end
