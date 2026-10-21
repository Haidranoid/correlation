namespace :db do
  namespace :backup do
    desc "Dump database to a file named with day of week (for daily cron rotation)"
    task dump: :environment do
      day_of_week = Date.current.strftime('%A').downcase
      filename = "correlation_db_#{day_of_week}.sql.gz"
      backup_dir = Rails.root.join('db', 'backups')
      
      # Create backup directory if it doesn't exist
      FileUtils.mkdir_p(backup_dir)
      
      backup_path = backup_dir.join(filename)
      
      # Get database configuration
      config = Rails.application.config.database_configuration[Rails.env]
      database = config['database']
      host = config['host'] || 'localhost'
      port = config['port'] || 5432
      username = config['username']
      password = config['password']
      
      # Build pg_dump command with gzip compression
      pg_dump_cmd = ["pg_dump"]
      pg_dump_cmd << "--host=#{host}" if host != 'localhost' || username
      pg_dump_cmd << "--port=#{port}" if port != 5432 || username
      pg_dump_cmd << "--username=#{username}" if username
      pg_dump_cmd << "--no-password" if username
      pg_dump_cmd << "--verbose"
      pg_dump_cmd << "--clean"
      pg_dump_cmd << "--no-owner"
      pg_dump_cmd << "--no-privileges"
      pg_dump_cmd << "--format=plain"
      pg_dump_cmd << database
      
      # Set PGPASSWORD environment variable if password is provided
      env = {}
      env['PGPASSWORD'] = password if password
      
      puts "Creating compressed database backup: #{backup_path}"
      puts "Command: #{pg_dump_cmd.join(' ')} | gzip -5 > #{backup_path}"
      
      # Use pipe to compress output with gzip -5
      success = system(env, "#{pg_dump_cmd.join(' ')} | gzip -5 > #{backup_path}")
      
      if success
        file_size = File.size(backup_path)
        puts "✅ Database backup completed successfully!"
        puts "📁 File: #{backup_path}"
        puts "📊 Size: #{ActiveSupport::NumberHelper.number_to_human_size(file_size)}"
        puts "🗓️  Day: #{day_of_week.capitalize}"
      else
        puts "❌ Database backup failed!"
        exit 1
      end
    end
    
    desc "Restore database from a day-of-week backup file"
    task :restore, [:day] => :environment do |task, args|
      day_of_week = args[:day]
      
      unless day_of_week
        puts "❌ Please specify a day of the week:"
        puts "Usage: rake db:backup:restore[monday]"
        puts "Available days: monday, tuesday, wednesday, thursday, friday, saturday, sunday"
        exit 1
      end
      
      day_of_week = day_of_week.downcase
      valid_days = %w[monday tuesday wednesday thursday friday saturday sunday]
      
      unless valid_days.include?(day_of_week)
        puts "❌ Invalid day specified: #{day_of_week}"
        puts "Valid days: #{valid_days.join(', ')}"
        exit 1
      end
      
      filename = "correlation_db_#{day_of_week}.sql.gz"
      backup_dir = Rails.root.join('db', 'backups')
      backup_path = backup_dir.join(filename)
      
      unless File.exist?(backup_path)
        puts "❌ Backup file not found: #{backup_path}"
        puts "Available backups:"
        Dir.glob(backup_dir.join("correlation_db_*.sql.gz")).each do |file|
          puts "  - #{File.basename(file)}"
        end
        exit 1
      end
      
      # Get database configuration
      config = Rails.application.config.database_configuration[Rails.env]
      database = config['database']
      host = config['host'] || 'localhost'
      port = config['port'] || 5432
      username = config['username']
      password = config['password']
      
      # Confirm restoration
      puts "⚠️  WARNING: This will completely replace the current database!"
      puts "📁 Backup file: #{backup_path}"
      puts "🗃️  Target database: #{database}"
      puts "🗓️  Backup from: #{day_of_week.capitalize}"
      
      file_size = File.size(backup_path)
      puts "📊 Backup size: #{ActiveSupport::NumberHelper.number_to_human_size(file_size)}"
      
      print "Are you sure you want to continue? (y/N): "
      response = STDIN.gets.chomp.downcase
      
      unless response == 'y' || response == 'yes'
        puts "❌ Database restore cancelled."
        exit 0
      end
      
      # Build psql command with gunzip decompression
      psql_cmd = ["psql"]
      psql_cmd << "--host=#{host}" if host != 'localhost' || username
      psql_cmd << "--port=#{port}" if port != 5432 || username
      psql_cmd << "--username=#{username}" if username
      psql_cmd << "--no-password" if username
      psql_cmd << "--dbname=#{database}"
      psql_cmd << "--quiet"
      
      # Set PGPASSWORD environment variable if password is provided
      env = {}
      env['PGPASSWORD'] = password if password
      
      puts "🔄 Restoring database from compressed backup..."
      puts "Command: gunzip -c #{backup_path} | #{psql_cmd.join(' ')}"
      
      # Use gunzip to decompress and pipe to psql
      success = system(env, "gunzip -c #{backup_path} | #{psql_cmd.join(' ')}")
      
      if success
        puts "✅ Database restore completed successfully!"
        puts "📁 Restored from: #{backup_path}"
        puts "🗓️  Backup day: #{day_of_week.capitalize}"
      else
        puts "❌ Database restore failed!"
        exit 1
      end
    end
    
  end
end