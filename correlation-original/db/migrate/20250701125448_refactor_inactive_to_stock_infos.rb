class RefactorInactiveToStockInfos < ActiveRecord::Migration[8.0]
  def up
    # Step 1: Ensure stock_infos.inactive column exists and is properly set
    unless column_exists?(:stock_infos, :inactive)
      add_column :stock_infos, :inactive, :boolean, default: false, null: false
      add_index :stock_infos, :inactive
    end
    
    # Step 2: Sync inactive status from stock_prices to stock_infos
    # Find symbols that are marked inactive in stock_prices
    execute <<-SQL
      UPDATE stock_infos 
      SET inactive = true 
      WHERE symbol IN (
        SELECT DISTINCT symbol 
        FROM stock_prices 
        WHERE inactive = true
      )
    SQL
    
    # Step 3: Remove inactive column from stock_prices (it's redundant now)
    if column_exists?(:stock_prices, :inactive)
      remove_index :stock_prices, :inactive if index_exists?(:stock_prices, :inactive)
      remove_column :stock_prices, :inactive
    end
    
    # Step 4: Add foreign key relationship for data integrity
    unless foreign_key_exists?(:stock_prices, :stock_infos, column: :symbol, primary_key: :symbol)
      # Note: This is a custom foreign key since we're using symbol instead of id
      execute <<-SQL
        ALTER TABLE stock_prices 
        ADD CONSTRAINT fk_stock_prices_symbol 
        FOREIGN KEY (symbol) REFERENCES stock_infos(symbol)
      SQL
    end
  end
  
  def down
    # Rollback: Move inactive status back to stock_prices
    
    # Step 1: Add inactive column back to stock_prices
    unless column_exists?(:stock_prices, :inactive)
      add_column :stock_prices, :inactive, :boolean, default: false, null: false
      add_index :stock_prices, :inactive
    end
    
    # Step 2: Sync inactive status from stock_infos back to stock_prices
    execute <<-SQL
      UPDATE stock_prices 
      SET inactive = stock_infos.inactive 
      FROM stock_infos 
      WHERE stock_prices.symbol = stock_infos.symbol
    SQL
    
    # Step 3: Remove foreign key constraint
    if foreign_key_exists?(:stock_prices, name: 'fk_stock_prices_symbol')
      remove_foreign_key :stock_prices, name: 'fk_stock_prices_symbol'
    end
    
    # Step 4: Remove inactive column from stock_infos
    if column_exists?(:stock_infos, :inactive)
      remove_index :stock_infos, :inactive if index_exists?(:stock_infos, :inactive)
      remove_column :stock_infos, :inactive
    end
  end
end
