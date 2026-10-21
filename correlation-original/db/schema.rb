# This file is auto-generated from the current state of the database. Instead
# of editing this file, please use the migrations feature of Active Record to
# incrementally modify your database, and then regenerate this schema definition.
#
# This file is the source Rails uses to define your schema when running `bin/rails
# db:schema:load`. When creating a new database, `bin/rails db:schema:load` tends to
# be faster and is potentially less error prone than running all of your
# migrations from scratch. Old migrations may fail to apply correctly if those
# migrations use external dependencies or application code.
#
# It's strongly recommended that you check this file into your version control system.

ActiveRecord::Schema[8.0].define(version: 2025_07_02_130553) do
  # These are extensions that must be enabled in order to support this database
  enable_extension "pg_catalog.plpgsql"

  create_table "stock_api_provider_statuses", force: :cascade do |t|
    t.string "symbol", null: false
    t.string "provider_name", null: false
    t.string "status", null: false
    t.datetime "last_checked_at", null: false
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["provider_name"], name: "index_stock_api_provider_statuses_on_provider_name"
    t.index ["status"], name: "index_stock_api_provider_statuses_on_status"
    t.index ["symbol", "provider_name"], name: "index_stock_api_provider_statuses_on_symbol_and_provider_name", unique: true
  end

  create_table "stock_correlations", force: :cascade do |t|
    t.string "symbol1", null: false
    t.string "symbol2", null: false
    t.integer "window1_days", null: false
    t.integer "window2_days", null: false
    t.integer "offset_days", null: false
    t.decimal "correlation_coefficient", precision: 10, scale: 6, null: false
    t.date "start_date", null: false
    t.date "end_date", null: false
    t.integer "data_points", null: false
    t.decimal "r_squared", precision: 10, scale: 6, null: false
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["correlation_coefficient"], name: "index_stock_correlations_on_correlation_coefficient"
    t.index ["symbol1", "symbol2", "window1_days", "window2_days", "offset_days"], name: "index_correlations_on_analysis_params", unique: true
    t.index ["symbol1", "symbol2"], name: "index_stock_correlations_on_symbol1_and_symbol2"
  end

  create_table "stock_infos", force: :cascade do |t|
    t.string "symbol", null: false
    t.string "name"
    t.text "description"
    t.string "sector"
    t.string "industry"
    t.decimal "market_cap", precision: 15, scale: 2
    t.string "exchange"
    t.string "country"
    t.string "website"
    t.integer "employees"
    t.integer "founded"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.boolean "inactive", default: false, null: false
    t.index ["inactive"], name: "index_stock_infos_on_inactive"
    t.index ["industry"], name: "index_stock_infos_on_industry"
    t.index ["sector"], name: "index_stock_infos_on_sector"
    t.index ["symbol"], name: "index_stock_infos_on_symbol", unique: true
  end

  create_table "stock_prices", force: :cascade do |t|
    t.string "symbol"
    t.date "date"
    t.decimal "open"
    t.decimal "high"
    t.decimal "low"
    t.decimal "close"
    t.bigint "volume"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.string "market"
    t.string "country"
    t.string "data_type"
    t.index ["country", "market"], name: "index_stock_prices_on_country_and_market"
    t.index ["country"], name: "index_stock_prices_on_country"
    t.index ["data_type", "country"], name: "index_stock_prices_on_data_type_and_country"
    t.index ["data_type"], name: "index_stock_prices_on_data_type"
    t.index ["date"], name: "index_stock_prices_on_date"
    t.index ["market"], name: "index_stock_prices_on_market"
    t.index ["symbol", "date"], name: "index_stock_prices_on_symbol_and_date", unique: true
    t.index ["symbol", "market", "date"], name: "index_stock_prices_on_symbol_market_date", unique: true
    t.index ["symbol"], name: "index_stock_prices_on_symbol"
  end

  create_table "users", force: :cascade do |t|
    t.string "email", default: "", null: false
    t.string "encrypted_password", default: "", null: false
    t.string "reset_password_token"
    t.datetime "reset_password_sent_at"
    t.datetime "remember_created_at"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.datetime "last_sign_in_at"
    t.string "last_sign_in_ip"
    t.integer "sign_in_count", default: 0, null: false
    t.datetime "current_sign_in_at"
    t.string "current_sign_in_ip"
    t.index ["email"], name: "index_users_on_email", unique: true
    t.index ["reset_password_token"], name: "index_users_on_reset_password_token", unique: true
  end

  add_foreign_key "stock_prices", "stock_infos", column: "symbol", primary_key: "symbol", name: "fk_stock_prices_symbol"
end
