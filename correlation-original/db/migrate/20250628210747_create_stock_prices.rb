class CreateStockPrices < ActiveRecord::Migration[8.0]
  def change
    create_table :stock_prices do |t|
      t.string :symbol
      t.date :date
      t.decimal :open
      t.decimal :high
      t.decimal :low
      t.decimal :close
      t.bigint :volume

      t.timestamps
    end
    add_index :stock_prices, :symbol
    add_index :stock_prices, :date
    add_index :stock_prices, [:symbol, :date], unique: true
  end
end
