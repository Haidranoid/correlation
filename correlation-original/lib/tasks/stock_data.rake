require 'zip'
require 'parallel'
require 'terminal-table'
require_relative '../country_mapper'

namespace :stock_data do
  desc "Import bulk global stock data from STOOQ file (all countries)"
  task :import_stooq_data, [:file_path] => :environment do |t, args|
    unless args[:file_path] && File.exist?(args[:file_path])
      puts <<~USAGE
        ❌ Usage: rails stock_data:import_stooq_data[/path/to/stooq_file.zip]
        
        📥 To download STOOQ data:
        1. Go to: https://stooq.com/db/h/
        2. Download any data file (World, US, Europe, etc.)
        3. Click the download link in the 'ASCII' column
        4. Complete CAPTCHA and download the ZIP file
        5. Run: rails stock_data:import_stooq_data[/path/to/downloaded_file.zip]
        
        💡 This will import ALL stocks from the file (US, international, etc.)
      USAGE
      exit 1
    end
    
    puts "📦 Importing STOOQ global stock data from: #{args[:file_path]}"
    import_stooq_file(args[:file_path])
  end
  
  desc "Clean up temporary files created during import processes"
  task cleanup: :environment do
    puts "🧹 Manually cleaning up temporary files..."
    cleanup_temporary_files
  end

  private

  def import_stooq_file(zip_file_path)
    puts "📂 Processing STOOQ ZIP file: #{File.basename(zip_file_path)}"
    puts "🚀 Using #{Parallel.processor_count} CPU cores for parallel processing"
    
    start_time = Time.now
    
    begin
      Zip::File.open(zip_file_path) do |zip_file|
        file_data = extract_file_data(zip_file)
        return if file_data.empty?
        
        results = process_files_parallel(file_data, start_time)
        display_results(results, file_data.length, Time.now - start_time)
        cleanup_temporary_files
      end
    rescue StandardError => e
      puts "❌ Error processing STOOQ file: #{e.message}"
      cleanup_temporary_files
    end
  end

  def extract_file_data(zip_file)
    puts "📋 Analyzing ZIP contents..."
    
    entries = zip_file.entries.select { |entry| entry.file? && entry.name.end_with?('.txt') }
    total_entries = entries.length
    
    file_data = []
    
    entries.each_with_index do |entry, index|
      filename = File.basename(entry.name, '.txt')
      symbol, market, data_type = parse_filename(entry.name, filename)
      country = CountryMapper.map_symbol_to_country(symbol, market, data_type)
      
      if symbol.length.between?(1, 15) && symbol.match?(/^[A-Z0-9^_]+$/)
        file_data << {
          symbol: symbol, market: market, country: country, data_type: data_type,
          filename: entry.name, content: entry.get_input_stream.read
        }
      end
      
      # Show progress every 100 files or at the end
      if (index + 1) % 100 == 0 || index == total_entries - 1
        percent = ((index + 1).to_f / total_entries * 100).round(1)
        puts "  📦 Extracted #{index + 1}/#{total_entries} files (#{percent}%)"
      end
    end
    
    puts "📊 Found #{file_data.length} valid files to process"
    puts "📊 Data types: #{file_data.group_by { |f| f[:data_type] }.transform_values(&:count)}"
    puts "🌍 Markets: #{file_data.group_by { |f| f[:market] }.transform_values(&:count)}"
    
    file_data
  end

  def parse_filename(file_path, filename)
    data_type = case file_path
                when /\/bonds\// then "bonds"
                when /\/cryptocurrencies\// then "crypto" 
                when /\/currencies\/major\// then "forex_major"
                when /\/currencies\/other\// then "forex_other"
                when /\/indices\// then "indices"
                when /\/money market\// then "money_market"
                when /\/stooq stocks indices\// then "stock_indices"
                else "unknown"
                end

    if filename.include?('.')
      parts = filename.split('.')
      [parts[0].upcase, parts[1]&.upcase, data_type]
    else
      market = case data_type
               when "forex_major", "forex_other" then "FX"
               when "indices", "stock_indices" then "IDX" 
               when "money_market" then "MM"
               else "UNKNOWN"
               end
      [filename.upcase, market, data_type]
    end
  end

  def process_files_parallel(file_data, start_time)
    chunk_size = [(file_data.length / Parallel.processor_count).ceil, 1].max
    puts "\n🚀 Starting parallel processing with #{Parallel.processor_count} cores"
    puts "📋 Chunk size: #{chunk_size} files per core"
    puts "🎯 Processing #{file_data.length} files total...\n"
    
    # Create a shared progress tracking mechanism
    progress_file = "import_progress.tmp"
    File.write(progress_file, "0")
    
    results = Parallel.map(file_data.each_slice(chunk_size), in_processes: Parallel.processor_count) do |chunk|
      ActiveRecord::Base.connection.reconnect!
      process_chunk(chunk, start_time, file_data.length, progress_file)
    end
    
    # Clean up progress files
    File.delete(progress_file) if File.exist?(progress_file)
    File.delete("#{progress_file}.lock") if File.exist?("#{progress_file}.lock")
    
    results
  end

  def process_chunk(chunk, start_time, total_files, progress_file)
    batch_records = []
    chunk_created = 0
    process_id = Process.pid
    last_reported = 0
    
    chunk.each_with_index do |file_info, file_index|
      lines = file_info[:content].split("\n")
      
      lines.each do |line|
        next if line.strip.empty? || line.include?('<TICKER>')
        
        record = parse_line(line, file_info)
        next unless record
        
        batch_records << record
        
        if batch_records.length >= 1000
          chunk_created += process_batch(batch_records)
          batch_records.clear
        end
      end
      
      # Update shared progress every 10 files
      if (file_index + 1) % 10 == 0 || file_index == chunk.length - 1
        files_completed = (file_index + 1) - last_reported
        update_global_progress(progress_file, files_completed, total_files, start_time, process_id, chunk_created)
        last_reported = file_index + 1
      end
    end
    
    chunk_created += process_batch(batch_records) if batch_records.any?
    { created: chunk_created, processed: chunk.length }
  end

  def update_global_progress(progress_file, files_completed, total_files, start_time, process_id, records_created)
    begin
      # Thread-safe progress update
      File.open("#{progress_file}.lock", File::RDWR|File::CREAT, 0644) do |lock_file|
        lock_file.flock(File::LOCK_EX)
        
        current_progress = File.read(progress_file).to_i rescue 0
        new_progress = current_progress + files_completed
        File.write(progress_file, new_progress)
        
        elapsed = Time.now - start_time
        percent = (new_progress.to_f / total_files * 100).round(1)
        
        puts "  🌍 Overall Progress: #{new_progress}/#{total_files} files (#{percent}%) - #{elapsed.round(1)}s elapsed - #{records_created} records processed"
      end
    rescue StandardError => e
      # Continue if progress tracking fails
      puts "  ⚠️  Progress tracking error: #{e.message}"
    end
  end

  def parse_line(line, file_info)
    parts = line.split(',')
    return nil if parts.length < 5
    
    begin
      date = if parts.length >= 10
               # Standard STOOQ format: TICKER,PERIOD,DATE,TIME,OPEN,HIGH,LOW,CLOSE,VOL,OPENINT
               date_str = parts[2]
               Date.parse("#{date_str[0..3]}-#{date_str[4..5]}-#{date_str[6..7]}")
             else
               Date.parse(parts[0])
             end
      
      return nil if date.saturday? || date.sunday?
      
      prices = parts.length >= 10 ? parts[4..7].map(&:to_f) : parts[1..4].map(&:to_f)
      volume = (parts.length >= 10 ? parts[8] : parts[5])&.to_i || 0
      
      file_info.slice(:symbol, :market, :country, :data_type).merge(
        date: date, open: prices[0], high: prices[1], low: prices[2], 
        close: prices[3], volume: volume, created_at: Time.current, updated_at: Time.current
      )
    rescue StandardError
      nil
    end
  end

  def process_batch(batch_records)
    return 0 if batch_records.empty?
    
    begin
      ActiveRecord::Base.transaction do
        batch_records.each_slice(500) do |mini_batch|
          values = mini_batch.map { |r| 
            "(#{%w[symbol market country data_type date open high low close volume created_at updated_at]
              .map { |field| ActiveRecord::Base.connection.quote(r[field.to_sym]) }.join(', ')})"
          }.join(', ')
          
          sql = <<~SQL
            INSERT INTO stock_prices (symbol, market, country, data_type, date, open, high, low, close, volume, created_at, updated_at)
            VALUES #{values}
            ON CONFLICT (symbol, market, date) DO UPDATE SET
              open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, close = EXCLUDED.close,
              volume = EXCLUDED.volume, country = EXCLUDED.country, data_type = EXCLUDED.data_type,
              updated_at = EXCLUDED.updated_at
          SQL
          
          ActiveRecord::Base.connection.execute(sql)
        end
      end
      batch_records.length
    rescue StandardError
      batch_records.each do |record|
        begin
          StockPrice.find_or_create_by(record.slice(:symbol, :market, :date)) { |sp| sp.assign_attributes(record) }
        rescue StandardError
          next
        end
      end
      batch_records.length
    end
  end

  def display_results(results, total_files, total_time)
    total_created = results.sum { |r| r[:created] }
    puts <<~RESULTS
      
      🎉 STOOQ import complete!
      ⏱️  Total time: #{total_time.round(1)} seconds
      📂 Files processed: #{total_files}
      📈 Records created: #{total_created}
      💾 Total symbols: #{StockPrice.distinct.count(:symbol)}
      🌍 Countries: #{StockPrice.distinct.count(:country)}
      📊 Markets: #{StockPrice.distinct.count(:market)}
      ⚡ Processing rate: #{(total_files.to_f / total_time).round(1)} files/second
    RESULTS
  end


  def cleanup_temporary_files
    puts "\n🧹 Cleaning up temporary files..."
    
    patterns = %w[import_progress.tmp import_progress.lock *.tmp *.lock progress_*.tmp import*.lock progress*.lock]
    cleaned_files = 0
    
    patterns.each do |pattern|
      Dir.glob(pattern).each do |file|
        next if %w[Gemfile.lock].include?(file) || file.include?('stock_symbols_2024') || file.include?('stock_data_progress.json')
        
        begin
          File.delete(file)
          puts "  🗑️  Deleted: #{file}"
          cleaned_files += 1
        rescue StandardError => e
          puts "  ⚠️  Could not delete #{file}: #{e.message}"
        end
      end
    end
    
    # Clean up Rails tmp directory
    rails_tmp_dir = Rails.root.join('tmp')
    if Dir.exist?(rails_tmp_dir)
      Dir.glob(File.join(rails_tmp_dir, "import_*.tmp")).each do |file|
        begin
          File.delete(file)
          puts "  🗑️  Deleted: #{file}"
          cleaned_files += 1
        rescue StandardError => e
          puts "  ⚠️  Could not delete #{file}: #{e.message}"
        end
      end
    end
    
    # Clean up data directory after import
    data_dir = Rails.root.join('data')
    if Dir.exist?(data_dir)
      begin
        FileUtils.rm_rf(data_dir)
        puts "  🗑️  Deleted data directory (freed ~1.1GB)"
        cleaned_files += 1
      rescue StandardError => e
        puts "  ⚠️  Could not delete data directory: #{e.message}"
      end
    end

    puts cleaned_files > 0 ? "✅ Cleaned up #{cleaned_files} temporary files and data directory" : "✅ No temporary files found to clean up"
  end
end