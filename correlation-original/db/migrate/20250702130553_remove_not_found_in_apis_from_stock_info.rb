class RemoveNotFoundInApisFromStockInfo < ActiveRecord::Migration[8.0]
  def change
    remove_column :stock_infos, :not_found_in_apis, :boolean
  end
end
