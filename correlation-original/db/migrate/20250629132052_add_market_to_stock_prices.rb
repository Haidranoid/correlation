class AddMarketToStockPrices < ActiveRecord::Migration[8.0]
  def change
    add_column :stock_prices, :market, :string
    add_index :stock_prices, :market
    add_index :stock_prices, [:symbol, :market, :date], unique: true, name: 'index_stock_prices_on_symbol_market_date'
  end
end
