class AddCountryToStockPrices < ActiveRecord::Migration[8.0]
  def change
    add_column :stock_prices, :country, :string
    add_index :stock_prices, :country
    add_index :stock_prices, [:country, :market]
  end
end
