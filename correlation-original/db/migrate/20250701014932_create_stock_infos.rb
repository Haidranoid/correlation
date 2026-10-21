class CreateStockInfos < ActiveRecord::Migration[8.0]
  def change
    create_table :stock_infos do |t|
      t.string :symbol, null: false
      t.string :name
      t.text :description
      t.string :sector
      t.string :industry
      t.decimal :market_cap, precision: 15, scale: 2
      t.string :exchange
      t.string :country
      t.string :website
      t.integer :employees
      t.integer :founded

      t.timestamps
    end
    
    add_index :stock_infos, :symbol, unique: true
    add_index :stock_infos, :sector
    add_index :stock_infos, :industry
  end
end
