class CreateStockApiProviderStatuses < ActiveRecord::Migration[8.0]
  def change
    create_table :stock_api_provider_statuses do |t|
      t.string :symbol, null: false
      t.string :provider_name, null: false
      t.string :status, null: false # 'not_found', 'found', 'rate_limited', 'error'
      t.datetime :last_checked_at, null: false

      t.timestamps
    end
    
    add_index :stock_api_provider_statuses, [:symbol, :provider_name], unique: true
    add_index :stock_api_provider_statuses, :provider_name
    add_index :stock_api_provider_statuses, :status
  end
end
