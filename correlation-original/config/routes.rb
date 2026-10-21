Rails.application.routes.draw do
  devise_for :users, skip: [:registrations], controllers: {
    sessions: 'users/sessions'
  }
  
  # Admin-only user management routes
  resources :admin_users, path: 'admin/users', only: [:index, :new, :create, :edit, :update, :destroy]
  # Home routes
  root "home#index"
  get "stocks/:symbol", to: "home#show", as: :stock_detail
  
  # Documentation and Visualization routes
  get 'documentation', to: 'docs#index'
  get 'documentation/:page', to: 'docs#show', as: :documentation_page
  get 'visualizations', to: 'visualizations#index'
  get 'visualizations/:type', to: 'visualizations#show', as: :visualization
  
  # Tools routes
  get 'tools', to: 'tools#index'
  get 'tools/download/*filename', to: 'tools#download_backup', as: :download_backup, constraints: { filename: /.*/ }
  
  # API routes for DataTables and charts
  get "api/stocks", to: "home#stocks_data"
  get "api/chart_data", to: "home#chart_data"
  get "api/price_data/:symbol", to: "home#price_data"
  get "api/recent_price_data/:symbol", to: "home#recent_price_data"
  
  # Define your application routes per the DSL in https://guides.rubyonrails.org/routing.html

  # Reveal health status on /up that returns 200 if the app boots with no exceptions, otherwise 500.
  # Can be used by load balancers and uptime monitors to verify that the app is live.
  get "up" => "rails/health#show", as: :rails_health_check

  # Render dynamic PWA files from app/views/pwa/* (remember to link manifest in application.html.erb)
  # get "manifest" => "rails/pwa#manifest", as: :pwa_manifest
  # get "service-worker" => "rails/pwa#service_worker", as: :pwa_service_worker
end
