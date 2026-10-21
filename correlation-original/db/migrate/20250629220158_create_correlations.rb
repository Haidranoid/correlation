class CreateCorrelations < ActiveRecord::Migration[8.0]
  def change
    create_table :correlations do |t|
      t.string :symbol1, null: false
      t.string :symbol2, null: false
      t.integer :window1_days, null: false
      t.integer :window2_days, null: false
      t.integer :offset_days, null: false
      t.decimal :correlation_coefficient, precision: 10, scale: 6, null: false
      t.date :start_date, null: false
      t.date :end_date, null: false
      t.integer :data_points, null: false
      t.decimal :r_squared, precision: 10, scale: 6, null: false
      t.datetime :analysis_date, null: false

      t.timestamps
    end

    add_index :correlations, [:symbol1, :symbol2]
    add_index :correlations, :correlation_coefficient
    add_index :correlations, :analysis_date
    add_index :correlations, [:symbol1, :symbol2, :window1_days, :window2_days, :offset_days], 
              name: 'index_correlations_on_analysis_params', unique: true
  end
end
