class AddInactiveToStockPrices < ActiveRecord::Migration[8.0]
  def change
    add_column :stock_prices, :inactive, :boolean, default: false, null: false
    add_index :stock_prices, :inactive
  end
end
