FactoryBot.define do
  factory :stock_price do
    symbol { 'MSFT' }
    date { Date.current }
    close { Faker::Number.decimal(l_digits: 3, r_digits: 2) }
    
    trait :msft do
      symbol { 'MSFT' }
    end
    
    trait :aapl do
      symbol { 'AAPL' }
    end
    
    trait :googl do
      symbol { 'GOOGL' }
    end
    
    trait :with_sequence_date do
      sequence(:date) { |n| Date.current + n.days }
    end
    
    trait :historical do
      date { Faker::Date.between(from: 2.years.ago, to: 1.month.ago) }
    end
    
    trait :recent do
      date { Faker::Date.between(from: 1.month.ago, to: Date.current) }
    end
    
    trait :high_price do
      close { Faker::Number.decimal(l_digits: 3, r_digits: 2) + 1000 }
    end
    
    trait :low_price do
      close { Faker::Number.decimal(l_digits: 1, r_digits: 2) + 1 }
    end
  end
end