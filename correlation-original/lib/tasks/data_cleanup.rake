namespace :data_cleanup do
  desc "Analyze data quality and show recommendations for cleanup"
  task analyze: :environment do
    puts "🔍 STOCK DATA QUALITY ANALYSIS"
    puts "=" * 50
    
    total_symbols = StockInfo.count
    active_symbols = StockInfo.active.count
    inactive_symbols = StockInfo.inactive.count
    
    puts "Total symbols: #{total_symbols}"
    puts "Active symbols: #{active_symbols}"
    puts "Inactive symbols: #{inactive_symbols}"
    puts
    
    # Analyze record count distribution for active symbols only
    # Use a more efficient query for this analysis
    record_counts = StockPrice.joins(:stock_info)
                              .where(stock_infos: { inactive: false })
                              .group('stock_prices.symbol')
                              .count
    
    puts "📊 Record count distribution (active symbols only):"
    thresholds = [1, 5, 10, 50, 100, 500, 1000]
    thresholds.each do |threshold|
      count = record_counts.values.count { |c| c < threshold }
      percentage = record_counts.size > 0 ? (count.to_f / record_counts.size * 100).round(1) : 0
      puts "  < #{threshold} records: #{count} symbols (#{percentage}%)"
    end
    
    puts
    puts "📉 Symbols with very few records (< 50):"
    low_quality_symbols = record_counts.select { |symbol, count| count < 50 }
    puts "  Found #{low_quality_symbols.size} symbols with < 50 records"
    
    if low_quality_symbols.size > 0
      puts "  Sample symbols:"
      low_quality_symbols.sort_by { |k,v| v }.first(10).each do |symbol, count|
        puts "    #{symbol}: #{count} records"
      end
    end
    
    puts
    puts "📈 Quality statistics:"
    if record_counts.size > 0
      avg_records = record_counts.values.sum.to_f / record_counts.size
      median_records = record_counts.values.sort[record_counts.size / 2]
      puts "  Average records per symbol: #{avg_records.round(1)}"
      puts "  Median records per symbol: #{median_records}"
    end
    
    # Check for symbols without stock info (should be 0 due to foreign key)
    symbols_without_info = StockPrice.active.distinct.pluck(:symbol) - StockInfo.active.pluck(:symbol)
    puts "  Symbols missing stock info: #{symbols_without_info.size}"
    
    # Check for placeholder stock info
    placeholder_stock_infos = StockInfo.where(
      "(name LIKE ? AND description LIKE ?) OR description LIKE ?",
      "% Company", "%placeholder%", "%This is a placeholder%"
    ).count
    puts "  Symbols with placeholder stock info: #{placeholder_stock_infos}"
    
    puts
    puts "💡 RECOMMENDATIONS:"
    puts "  - Consider marking symbols with < 50 records as inactive"
    puts "  - Consider marking symbols without stock info as inactive"
    puts "  - Consider marking symbols with placeholder stock info as inactive (#{placeholder_stock_infos} found)"
    puts "  - Run 'rails data_cleanup:mark_inactive' to apply automatic cleanup"
  end
  
  desc "Mark low-quality symbols as inactive (specify MIN_RECORDS=50)"
  task mark_inactive: :environment do
    min_records = ENV['MIN_RECORDS']&.to_i || 50
    require_stock_info = ENV['REQUIRE_STOCK_INFO'] != 'false'
    exclude_placeholder_info = ENV['EXCLUDE_PLACEHOLDER'] != 'false'
    recheck_inactive = ENV['RECHECK_INACTIVE'] != 'false'
    dry_run = ENV['DRY_RUN'] == 'true'
    
    puts "🧹 MARKING LOW-QUALITY SYMBOLS AS INACTIVE"
    puts "=" * 50
    puts "Minimum records threshold: #{min_records}"
    puts "Require stock info: #{require_stock_info}"
    puts "Exclude placeholder stock info: #{exclude_placeholder_info}"
    puts "Recheck inactive symbols: #{recheck_inactive}"
    puts "Dry run mode: #{dry_run ? 'YES (no changes will be made)' : 'NO (changes will be applied)'}"
    puts
    
    # Step 1: Recheck inactive symbols and reactivate if they now qualify
    symbols_reactivated = 0
    if recheck_inactive
      puts "🔄 Checking inactive symbols for reactivation..."
      
      # Build conditions for symbols that should be reactivated
      reactivation_conditions = []
      reactivation_conditions << "stock_infos.inactive = true"
      
      # Require minimum records
      having_clause = "COUNT(stock_prices.id) >= #{min_records}"
      
      # Require stock info name if specified
      if require_stock_info
        reactivation_conditions << "stock_infos.name IS NOT NULL AND stock_infos.name != ''"
      end
      
      # Exclude placeholder info if specified
      if exclude_placeholder_info
        reactivation_conditions << "NOT ((stock_infos.name LIKE '% Company' AND stock_infos.description LIKE '%placeholder%') OR stock_infos.description LIKE '%This is a placeholder%')"
      end
      
      # Use single SQL query to find symbols that qualify for reactivation
      symbols_to_reactivate = StockInfo.joins(:stock_prices)
                                      .where(reactivation_conditions.join(' AND '))
                                      .group('stock_infos.symbol, stock_infos.name')
                                      .having(having_clause)
                                      .pluck('stock_infos.symbol')
      
      if symbols_to_reactivate.any?
        puts "  Found #{symbols_to_reactivate.size} inactive symbols that now qualify for reactivation"
        puts "  Sample symbols to reactivate: #{symbols_to_reactivate.first(5).join(', ')}"
        
        unless dry_run
          # Reactivate by updating stock_infos
          StockInfo.where(symbol: symbols_to_reactivate).update_all(inactive: false)
          symbols_reactivated = symbols_to_reactivate.size
          puts "  ✅ Reactivated #{symbols_reactivated} symbols"
        else
          puts "  🔍 DRY RUN: Would reactivate #{symbols_to_reactivate.size} symbols"
        end
      else
        puts "  No inactive symbols qualify for reactivation"
      end
      
      puts
    end
    
    # Step 2: Find symbols to mark as inactive
    record_counts = StockPrice.joins(:stock_info)
                              .where(stock_infos: { inactive: false })
                              .group('stock_prices.symbol')
                              .count
    low_record_symbols = record_counts.select { |symbol, count| count < min_records }.keys
    
    symbols_to_deactivate = Set.new(low_record_symbols)
    
    if require_stock_info
      # Should be empty due to foreign key constraint, but check anyway
      symbols_without_info = StockPrice.active.distinct.pluck(:symbol) - StockInfo.active.pluck(:symbol)
      symbols_to_deactivate.merge(symbols_without_info)
    end
    
    if exclude_placeholder_info
      # Find symbols with placeholder/incomplete stock info
      placeholder_stock_infos = StockInfo.active.where(
        "(name LIKE ? AND description LIKE ?) OR description LIKE ?",
        "% Company", "%placeholder%", "%This is a placeholder%"
      ).pluck(:symbol)
      symbols_to_deactivate.merge(placeholder_stock_infos)
    end
    
    puts "📊 Analysis results:"
    puts "  Symbols with < #{min_records} records: #{low_record_symbols.size}"
    if require_stock_info
      symbols_without_info = StockPrice.active.distinct.pluck(:symbol) - StockInfo.active.pluck(:symbol)
      puts "  Symbols without stock info: #{symbols_without_info.size}"
    end
    if exclude_placeholder_info
      placeholder_count = StockInfo.active.where(
        "(name LIKE ? AND description LIKE ?) OR description LIKE ?",
        "% Company", "%placeholder%", "%This is a placeholder%"
      ).count
      puts "  Symbols with placeholder stock info: #{placeholder_count}"
    end
    puts "  Total symbols to mark inactive: #{symbols_to_deactivate.size}"
    
    if symbols_to_deactivate.empty?
      puts "✅ No symbols need to be marked as inactive!"
      next
    end
    
    puts
    puts "🎯 Sample symbols to be marked inactive:"
    symbols_to_deactivate.first(10).each do |symbol|
      record_count = record_counts[symbol] || 0
      has_info = StockInfo.exists?(symbol: symbol)
      stock_info = StockInfo.find_by(symbol: symbol)
      is_placeholder = stock_info && (
        (stock_info.name&.include?("Company") && stock_info.description&.include?("placeholder")) ||
        stock_info.description&.include?("This is a placeholder")
      )
      puts "  #{symbol}: #{record_count} records, stock_info: #{has_info ? 'Yes' : 'No'}#{is_placeholder ? ' (placeholder)' : ''}"
    end
    
    if dry_run
      puts
      puts "🔍 DRY RUN MODE - No changes made"
      puts "To apply changes, run: MIN_RECORDS=#{min_records} rails data_cleanup:mark_inactive"
    else
      puts
      puts "⚠️  This will mark #{symbols_to_deactivate.size} symbols as inactive."
      puts "Continue? (y/N)"
      response = $stdin.gets.chomp.downcase
      
      if response == 'y' || response == 'yes'
        puts
        puts "🔄 Marking symbols as inactive..."
        
        # Update stock_infos only (this will cascade to affect stock_prices queries)
        updated_stock_infos = 0
        symbols_to_deactivate.each_slice(1000).with_index do |batch, batch_index|
          count = StockInfo.where(symbol: batch).update_all(inactive: true)
          updated_stock_infos += count
          puts "  Processed batch #{batch_index + 1}: #{batch.size} symbols"
        end
        
        puts
        puts "✅ CLEANUP COMPLETED!"
        puts "  Stock info records marked inactive: #{updated_stock_infos}"
        if symbols_reactivated > 0
          puts "  Symbols reactivated this run: #{symbols_reactivated}"
        end
        puts
        puts "📊 New statistics:"
        remaining_active = StockInfo.active.count
        total_inactive = StockInfo.inactive.count
        puts "  Active symbols: #{remaining_active}"
        puts "  Inactive symbols: #{total_inactive}"
      else
        puts "❌ Operation cancelled"
      end
    end
  end
  
  desc "Reactivate symbols (specify SYMBOLS='AAPL,MSFT' or MIN_RECORDS=100)"
  task reactivate: :environment do
    symbols_param = ENV['SYMBOLS']
    min_records = ENV['MIN_RECORDS']&.to_i
    
    puts "🔄 REACTIVATING SYMBOLS"
    puts "=" * 30
    
    if symbols_param
      # Reactivate specific symbols
      symbols_to_reactivate = symbols_param.split(',').map(&:strip).map(&:upcase)
      puts "Reactivating specific symbols: #{symbols_to_reactivate.join(', ')}"
      
      symbols_to_reactivate.each do |symbol|
        stock_count = StockPrice.where(symbol: symbol, inactive: true).update_all(inactive: false)
        info_count = StockInfo.where(symbol: symbol).exists? ? StockInfo.where(symbol: symbol).update_all(inactive: false) : 0
        puts "  #{symbol}: #{stock_count} price records, #{info_count} info records reactivated"
      end
    elsif min_records
      # Reactivate symbols that now meet the minimum threshold
      puts "Reactivating symbols with >= #{min_records} records..."
      
      inactive_symbols = StockPrice.where(inactive: true).group(:symbol).count
      symbols_to_reactivate = inactive_symbols.select { |symbol, count| count >= min_records }.keys
      
      puts "Found #{symbols_to_reactivate.size} symbols meeting the criteria"
      
      symbols_to_reactivate.each_slice(1000) do |batch|
        StockPrice.where(symbol: batch, inactive: true).update_all(inactive: false)
        StockInfo.where(symbol: batch).update_all(inactive: false) if StockInfo.column_names.include?('inactive')
      end
      
      puts "✅ Reactivated #{symbols_to_reactivate.size} symbols"
    else
      puts "❌ Please specify SYMBOLS='AAPL,MSFT' or MIN_RECORDS=100"
    end
  end
  
  desc "Show cleanup statistics"
  task stats: :environment do
    puts "📊 DATA CLEANUP STATISTICS"
    puts "=" * 40
    
    total_symbols = StockInfo.count
    active_symbols = StockInfo.active.count
    inactive_symbols = StockInfo.inactive.count
    
    puts "Symbol Status:"
    puts "  Total symbols: #{total_symbols}"
    puts "  Active symbols: #{active_symbols} (#{(active_symbols.to_f / total_symbols * 100).round(1)}%)"
    puts "  Inactive symbols: #{inactive_symbols} (#{(inactive_symbols.to_f / total_symbols * 100).round(1)}%)"
    
    puts
    puts "Record Statistics:"
    total_records = StockPrice.count
    active_records = StockPrice.active.count
    inactive_records = total_records - active_records
    
    puts "  Total records: #{number_with_delimiter(total_records)}"
    puts "  Active records: #{number_with_delimiter(active_records)} (#{(active_records.to_f / total_records * 100).round(1)}%)"
    puts "  Inactive records: #{number_with_delimiter(inactive_records)} (#{(inactive_records.to_f / total_records * 100).round(1)}%)"
    
    puts
    puts "Stock Info Status:"
    puts "  Total stock infos: #{total_symbols}"
    puts "  Active stock infos: #{active_symbols}"
    puts "  Inactive stock infos: #{inactive_symbols}"
  end
  
  private
  
  def number_with_delimiter(number)
    number.to_s.reverse.gsub(/(\d{3})(?=\d)/, '\\1,').reverse
  end
end