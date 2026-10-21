class ToolsController < ApplicationController
  before_action :authenticate_user!
  
  def index
    @backup_files = load_backup_files
  end
  
  def download_backup
    filename = params[:filename].is_a?(Array) ? params[:filename].join('/') : params[:filename]
    
    unless filename.present? && valid_backup_filename?(filename)
      redirect_to tools_path, alert: "Invalid backup file specified."
      return
    end
    
    backup_path = backup_directory.join(filename)
    
    unless File.exist?(backup_path)
      redirect_to tools_path, alert: "Backup file not found."
      return
    end
    
    send_file backup_path,
              filename: filename,
              type: 'application/gzip',
              disposition: 'attachment'
  end
  
  private
  
  def load_backup_files
    return [] unless Dir.exist?(backup_directory)
    
    Dir.glob(backup_directory.join("correlation_db_*.sql.gz"))
       .map { |file| backup_file_info(file) }
       .sort_by { |info| info[:day_order] }
  end
  
  def backup_file_info(file_path)
    filename = File.basename(file_path)
    file_size = File.size(file_path)
    modified_time = File.mtime(file_path)
    
    day = filename.match(/correlation_db_(\w+)\.sql\.gz/)&.captures&.first
    day_order = day_order_for(day)
    
    {
      filename: filename,
      day: day&.capitalize || 'Unknown',
      size: ActiveSupport::NumberHelper.number_to_human_size(file_size),
      modified: modified_time.strftime('%Y-%m-%d %H:%M'),
      day_order: day_order
    }
  end
  
  def day_order_for(day)
    days = %w[monday tuesday wednesday thursday friday saturday sunday]
    days.index(day&.downcase) || 999
  end
  
  def valid_backup_filename?(filename)
    filename.match?(/\Acorrelation_db_\w+\.sql\.gz\z/)
  end
  
  def backup_directory
    Rails.root.join('db', 'backups')
  end
end