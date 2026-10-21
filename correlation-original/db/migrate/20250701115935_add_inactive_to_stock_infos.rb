class AddInactiveToStockInfos < ActiveRecord::Migration[8.0]
  def change
    add_column :stock_infos, :inactive, :boolean, default: false, null: false
    add_index :stock_infos, :inactive
  end
end
