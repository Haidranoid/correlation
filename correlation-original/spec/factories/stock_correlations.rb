FactoryBot.define do
  factory :stock_correlation do
    symbol1 { 'MSFT' }
    symbol2 { 'AAPL' }
    window1_days { Faker::Number.between(from: 1, to: 30) }
    window2_days { Faker::Number.between(from: 1, to: 30) }
    offset_days { Faker::Number.between(from: -100, to: 100) }
    correlation_coefficient { Faker::Number.between(from: -1.0, to: 1.0) }
    r_squared { correlation_coefficient.abs ** 2 }
    
    after(:build) do |correlation|
      correlation.r_squared = correlation.correlation_coefficient.abs ** 2
    end
    data_points { Faker::Number.between(from: 50, to: 1000) }
    start_date { Faker::Date.between(from: 2.years.ago, to: 6.months.ago) }
    end_date { start_date + Faker::Number.between(from: 30, to: 365).days }
    analysis_date { Date.current }
    
    trait :strong_positive do
      correlation_coefficient { Faker::Number.between(from: 0.7, to: 1.0) }
    end
    
    trait :strong_negative do
      correlation_coefficient { Faker::Number.between(from: -1.0, to: -0.7) }
    end
    
    trait :weak_correlation do
      correlation_coefficient { Faker::Number.between(from: -0.3, to: 0.3) }
    end
    
    trait :no_correlation do
      correlation_coefficient { Faker::Number.between(from: -0.1, to: 0.1) }
    end
    
    trait :perfect_positive do
      correlation_coefficient { 1.0 }
    end
    
    trait :perfect_negative do
      correlation_coefficient { -1.0 }
    end
    
    trait :msft_aapl do
      symbol1 { 'MSFT' }
      symbol2 { 'AAPL' }
    end
    
    trait :msft_googl do
      symbol1 { 'MSFT' }
      symbol2 { 'GOOGL' }
    end
    
    trait :aapl_googl do
      symbol1 { 'AAPL' }
      symbol2 { 'GOOGL' }
    end
    
    trait :short_term do
      window1_days { Faker::Number.between(from: 1, to: 7) }
      window2_days { Faker::Number.between(from: 1, to: 7) }
    end
    
    trait :long_term do
      window1_days { Faker::Number.between(from: 30, to: 90) }
      window2_days { Faker::Number.between(from: 30, to: 90) }
    end
    
    trait :recent_analysis do
      analysis_date { Faker::Date.between(from: 7.days.ago, to: Date.current) }
    end
    
    trait :old_analysis do
      analysis_date { Faker::Date.between(from: 6.months.ago, to: 2.months.ago) }
    end
    
    trait :zero_offset do
      offset_days { 0 }
    end
    
    trait :positive_offset do
      offset_days { Faker::Number.between(from: 1, to: 30) }
    end
    
    trait :negative_offset do
      offset_days { Faker::Number.between(from: -30, to: -1) }
    end
    
    trait :large_dataset do
      data_points { Faker::Number.between(from: 500, to: 2000) }
    end
    
    trait :small_dataset do
      data_points { Faker::Number.between(from: 10, to: 50) }
    end
  end
end