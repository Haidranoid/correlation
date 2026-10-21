class AddMissingTrackableFieldsToUsers < ActiveRecord::Migration[8.0]
  def change
    add_column :users, :sign_in_count, :integer, default: 0, null: false
    add_column :users, :current_sign_in_at, :datetime
    add_column :users, :current_sign_in_ip, :string
  end
end
