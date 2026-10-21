class RemoveAnalysisDateFromStockCorrelations < ActiveRecord::Migration[8.0]
  def change
    remove_index :stock_correlations, :analysis_date if index_exists?(:stock_correlations, :analysis_date)
    remove_column :stock_correlations, :analysis_date, :datetime
  end
end
