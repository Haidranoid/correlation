class AddDataTypeToStockPrices < ActiveRecord::Migration[8.0]
  def change
    add_column :stock_prices, :data_type, :string
    add_index :stock_prices, :data_type
    add_index :stock_prices, [:data_type, :country]
  end
end
