class RenameCorrelationsToStockCorrelations < ActiveRecord::Migration[8.0]
  def change
    rename_table :correlations, :stock_correlations
  end
end
