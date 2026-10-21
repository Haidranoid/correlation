class AddNotFoundToStockInfo < ActiveRecord::Migration[8.0]
  def change
    add_column :stock_infos, :not_found_in_apis, :boolean, default: false, null: false
    add_index :stock_infos, :not_found_in_apis
  end
end
