# frozen_string_literal: true

require_relative '../country_mapper'
require 'terminal-table'

namespace :correlation do
  desc "View saved correlations from database"
  task :list, [:limit] => :environment do |t, args|
    limit = args[:limit]&.to_i || 10
    
    puts "📊 TOP #{limit} SAVED CORRELATIONS"
    puts "="*80
    
    correlations = StockCorrelation.order(correlation_coefficient: :desc).limit(limit)
    
    if correlations.empty?
      puts "❌ No correlations found in database"
      puts "💡 Run correlation:analyze[SYMBOL1,SYMBOL2,WINDOW1,WINDOW2] to generate some!"
    else
      table_data = correlations.map.with_index do |corr, index|
        [
          index + 1,
          corr.symbol1,
          corr.symbol2,
          corr.window1_days,
          corr.window2_days,
          corr.offset_days,
          corr.correlation_coefficient.round(4),
          corr.r_squared.round(4),
          corr.data_points,
          corr.analysis_date.strftime('%Y-%m-%d')
        ]
      end
      
      headers = ['Rank', 'Symbol1', 'Symbol2', 'Win1', 'Win2', 'Offset', 'Correlation', 'R²', 'Points', 'Analysis Date']
      
      # Display table using terminal-table
      table = Terminal::Table.new(headings: headers, rows: table_data)
      puts table
      puts "💡 Use correlation:analyze to add more correlations to the database"
    end
  end

  desc "Calculate correlation between two stock symbols with different time windows and offset"
  task :analyze, [:symbol1, :symbol2, :window1, :window2, :offset, :start_date, :end_date] => :environment do |t, args|
    # Validate required parameters
    unless args[:symbol1] && args[:symbol2] && args[:window1] && args[:window2]
      display_usage_instructions
      exit 1
    end

    # Parse and validate inputs
    symbol1 = args[:symbol1].upcase
    symbol2 = args[:symbol2].upcase
    window1 = args[:window1].to_i
    window2 = args[:window2].to_i
    
    # Determine date range
    start_date, end_date = determine_analysis_date_range(args, symbol1, symbol2, window1, window2)
    return if start_date.nil? || end_date.nil?

    # Perform correlation analysis
    analysis_service = CorrelationAnalysisService.new
    
    if args[:offset]
      perform_single_offset_analysis(analysis_service, symbol1, symbol2, window1, window2, args[:offset].to_i, start_date, end_date)
    else
      perform_best_offset_analysis(analysis_service, symbol1, symbol2, window1, window2, start_date, end_date)
    end
  end

  private

  # Display usage instructions for the analyze task
  def display_usage_instructions
    puts <<~USAGE
      ❌ Usage: rails correlation:analyze[SYMBOL1,SYMBOL2,WINDOW1_DAYS,WINDOW2_DAYS,OFFSET_DAYS,START_DATE,END_DATE]
      
      📊 Calculate correlation between two stocks with different time windows and offset
      
      Parameters:
      - SYMBOL1: First stock symbol (e.g., MSFT)
      - SYMBOL2: Second stock symbol (e.g., AAPL) 
      - WINDOW1_DAYS: Time window for SYMBOL1 percentage changes (e.g., 7 for weekly)
      - WINDOW2_DAYS: Time window for SYMBOL2 percentage changes (e.g., 30 for monthly)
      - OFFSET_DAYS: Days to offset SYMBOL2 (positive = future, negative = past, 0 = no offset, optional - if not specified, finds best offset automatically)
      - START_DATE: Start date for analysis (YYYY-MM-DD, optional - defaults to maximum available range)
      - END_DATE: End date for analysis (YYYY-MM-DD, optional - defaults to maximum available range)
      
      📈 Examples: 
      
      # Find best offset automatically (tests all possible offsets)
      rails correlation:analyze[MSFT,AAPL,7,30]
      
      # Use specific offset
      rails correlation:analyze[MSFT,AAPL,7,30,5]
      
      # Use specific date range with specific offset
      rails correlation:analyze[MSFT,AAPL,7,30,5,2023-01-01,2024-01-01]
      
      This calculates correlation between:
      - MSFT weekly (7-day) percentage changes
      - AAPL monthly (30-day) percentage changes offset by 5 days into the future
    USAGE
  end

  # Determine the date range for analysis
  def determine_analysis_date_range(args, symbol1, symbol2, window1, window2)
    if args[:start_date] && args[:end_date]
      start_date = Date.parse(args[:start_date])
      end_date = Date.parse(args[:end_date])
      puts "📊 Using specified date range: #{start_date} to #{end_date}"
      [start_date, end_date]
    else
      puts "📊 Determining maximum available date range for #{symbol1} and #{symbol2}..."
      start_date, end_date = StockDataService.determine_max_date_range(symbol1, symbol2, window1, window2, 0)
      
      if start_date.nil? || end_date.nil?
        puts "❌ No overlapping data found for #{symbol1} and #{symbol2}"
        return [nil, nil]
      end
      
      puts "📊 Using maximum available date range: #{start_date} to #{end_date}"
      [start_date, end_date]
    end
  end

  # Perform correlation analysis with a single specified offset
  def perform_single_offset_analysis(analysis_service, symbol1, symbol2, window1, window2, offset, start_date, end_date)
    puts "📊 Calculating correlation with specified offset: #{offset} days"
    
    begin
      result = analysis_service.analyze_correlation(symbol1, symbol2, window1, window2, offset, start_date, end_date)
      display_single_correlation_result(result)
      analysis_service.save_correlation_result(result)
      puts "\n✅ Analysis completed and saved to database"
    rescue CorrelationAnalysisService::AnalysisError => e
      puts "❌ Analysis failed: #{e.message}"
    end
  end

  # Perform correlation analysis to find all statistically relevant correlations across all possibilities
  def perform_best_offset_analysis(analysis_service, symbol1, symbol2, window1, window2, start_date, end_date)
    puts "📊 Finding all statistically relevant correlations across all possible offsets..."
    puts "📋 Criteria: |correlation| ≥ 0.3 AND R² ≥ 0.09 (moderate correlation or stronger)"
    
    begin
      results = analysis_service.find_best_correlation(symbol1, symbol2, window1, window2, start_date, end_date)
      
      if results[:error]
        puts "❌ #{results[:error]}"
      elsif results[:relevant_correlations].empty?
        puts "❌ No statistically relevant correlations found (|r| ≥ 0.3, R² ≥ 0.09)"
        puts "📊 Total correlations tested: #{results[:total_tested]}"
      else
        display_relevant_correlation_results(results, symbol1, symbol2)
        
        # Save all statistically relevant correlations to database (replaces existing records with same parameters)
        analysis_service.save_relevant_correlations(results[:relevant_correlations], symbol1, symbol2, window1, window2)
        puts "\n✅ #{results[:total_relevant]} statistically relevant correlation(s) analysis completed and saved to database"
      end
    rescue CorrelationAnalysisService::AnalysisError => e
      puts "❌ Analysis failed: #{e.message}"
    end
  end

  # Display results for single correlation analysis
  def display_single_correlation_result(result)
    puts "\n" + "="*60
    puts "📊 CORRELATION ANALYSIS RESULTS"
    puts "="*60
    
    puts "\n📈 Analysis Configuration:"
    puts "  #{result[:symbol1]}: #{result[:window1]}-day percentage changes"
    puts "  #{result[:symbol2]}: #{result[:window2]}-day percentage changes (#{result[:offset]}-day offset)"
    puts "  Data points analyzed: #{result[:data_points]}"
    puts "  Date range covered: #{result[:start_date]} to #{result[:end_date]}"
    
    puts "\n🔢 Correlation Coefficient: #{result[:correlation].round(4)}"
    puts "📊 Interpretation: #{result[:interpretation][:description]}"
    
    puts "\n📋 What this means:"
    if result[:correlation] > 0
      puts "  • When #{result[:symbol1]} increases over #{result[:window1]} days, #{result[:symbol2]} tends to increase over #{result[:window2]} days"
      puts "  • When #{result[:symbol1]} decreases over #{result[:window1]} days, #{result[:symbol2]} tends to decrease over #{result[:window2]} days"
    else
      puts "  • When #{result[:symbol1]} increases over #{result[:window1]} days, #{result[:symbol2]} tends to decrease over #{result[:window2]} days"
      puts "  • When #{result[:symbol1]} decreases over #{result[:window1]} days, #{result[:symbol2]} tends to increase over #{result[:window2]} days"
    end
    
    if result[:offset] != 0
      offset_desc = result[:offset] > 0 ? "#{result[:offset]} days in the future" : "#{result[:offset].abs} days in the past"
      puts "  • #{result[:symbol2]} changes are measured #{offset_desc} relative to #{result[:symbol1]}"
    end
    
    puts "\n📊 Statistical Notes:"
    puts "  • Sample size: #{result[:data_points]} paired observations"
    puts "  • R² (coefficient of determination): #{result[:r_squared].round(4)} (#{result[:interpretation][:variance_explained]}% of variance explained)"
    
    # Show sample data points
    puts "\n📋 Sample Data Points (first 5):"
    sample_data = result[:aligned_data].first(5).map do |change1, change2, date1, date2|
      [date1.to_s, "#{change1.round(2)}%", date2.to_s, "#{change2.round(2)}%"]
    end
    
    sample_headers = ["Date (#{result[:symbol1]})", "#{result[:symbol1]} Change", "Date (#{result[:symbol2]})", "#{result[:symbol2]} Change"]
    
    # Display sample data table using terminal-table
    sample_table = Terminal::Table.new(headings: sample_headers, rows: sample_data)
    puts sample_table
    
    puts "\n" + "="*60
  end

  # Display results for statistically relevant correlation analysis
  def display_relevant_correlation_results(results, symbol1, symbol2)
    best = results[:best_correlation]
    
    puts "\n🎉 STATISTICALLY RELEVANT CORRELATIONS FOUND!"
    puts "="*80
    puts "🔢 Best correlation coefficient: #{best[:correlation].round(4)} (offset: #{best[:offset]} days)"
    puts "📊 R²: #{best[:r_squared].round(4)} (#{(best[:r_squared] * 100).round(1)}% variance explained)"
    puts "📈 Data points: #{best[:data_points]}"
    puts "📅 Date range: #{best[:start_date]} to #{best[:end_date]}"
    puts "🔄 Total offsets tested: #{results[:total_tested]}"
    puts "✅ Statistically relevant found: #{results[:total_relevant]} (|r| ≥ 0.3, R² ≥ 0.09)"
    
    # Show all statistically relevant correlations
    puts "\n📊 ALL STATISTICALLY RELEVANT CORRELATIONS (will be saved to database):"
    puts "="*80
    
    relevant_table_data = results[:relevant_correlations].map.with_index do |result, index|
      [
        index + 1,
        result[:offset],
        result[:correlation].round(4),
        result[:r_squared].round(4),
        result[:data_points]
      ]
    end
    
    relevant_headers = ['Rank', 'Offset', 'Correlation', 'R²', 'Data Points']
    
    # Display relevant correlations table using terminal-table
    relevant_table = Terminal::Table.new(headings: relevant_headers, rows: relevant_table_data)
    puts relevant_table
    puts "="*80
  end
end