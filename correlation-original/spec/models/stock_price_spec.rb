require 'rails_helper'

RSpec.describe StockPrice, type: :model do
  describe 'validations' do
    subject { build(:stock_price) }
    
    it { should validate_presence_of(:symbol) }
    it { should validate_presence_of(:date) }
    it { should validate_presence_of(:close) }
    
    it { should validate_length_of(:symbol).is_at_most(10) }
    it { should validate_numericality_of(:close).is_greater_than(0) }
    
    it 'validates uniqueness of symbol and date combination' do
      create(:stock_price, symbol: 'MSFT', date: Date.current)
      duplicate = build(:stock_price, symbol: 'MSFT', date: Date.current)
      
      expect(duplicate).not_to be_valid
      expect(duplicate.errors[:symbol]).to include('has already been taken')
    end
    
    it 'allows same symbol on different dates' do
      create(:stock_price, symbol: 'MSFT', date: Date.current)
      different_date = build(:stock_price, symbol: 'MSFT', date: Date.current + 1.day)
      
      expect(different_date).to be_valid
    end
    
    it 'allows different symbols on same date' do
      create(:stock_price, symbol: 'MSFT', date: Date.current)
      different_symbol = build(:stock_price, symbol: 'AAPL', date: Date.current)
      
      expect(different_symbol).to be_valid
    end
  end
  
  describe 'scopes' do
    let!(:msft_jan_1) { create(:stock_price, symbol: 'MSFT', date: Date.parse('2023-01-01'), close_price: 100.0) }
    let!(:msft_jan_2) { create(:stock_price, symbol: 'MSFT', date: Date.parse('2023-01-02'), close_price: 102.0) }
    let!(:msft_jan_3) { create(:stock_price, symbol: 'MSFT', date: Date.parse('2023-01-03'), close_price: 98.0) }
    let!(:aapl_jan_1) { create(:stock_price, symbol: 'AAPL', date: Date.parse('2023-01-01'), close_price: 150.0) }
    let!(:aapl_jan_2) { create(:stock_price, symbol: 'AAPL', date: Date.parse('2023-01-02'), close_price: 155.0) }
    
    describe '.for_symbol' do
      it 'returns prices for specified symbol only' do
        msft_prices = StockPrice.for_symbol('MSFT')
        
        expect(msft_prices.count).to eq(3)
        expect(msft_prices.pluck(:symbol).uniq).to eq(['MSFT'])
      end
      
      it 'handles case-insensitive symbol lookup' do
        msft_prices = StockPrice.for_symbol('msft')
        
        expect(msft_prices.count).to eq(3)
        expect(msft_prices.pluck(:symbol).uniq).to eq(['MSFT'])
      end
      
      it 'returns empty relation for non-existent symbol' do
        prices = StockPrice.for_symbol('INVALID')
        
        expect(prices).to be_empty
      end
    end
    
    describe '.in_date_range' do
      it 'returns prices within specified date range' do
        prices = StockPrice.in_date_range(Date.parse('2023-01-01'), Date.parse('2023-01-02'))
        
        expect(prices.count).to eq(4)  # 2 MSFT + 2 AAPL
        expect(prices.pluck(:date).uniq.sort).to eq([Date.parse('2023-01-01'), Date.parse('2023-01-02')])
      end
      
      it 'excludes prices outside date range' do
        prices = StockPrice.in_date_range(Date.parse('2023-01-02'), Date.parse('2023-01-02'))
        
        expect(prices.count).to eq(2)  # Only Jan 2 prices
        expect(prices.pluck(:date).uniq).to eq([Date.parse('2023-01-02')])
      end
      
      it 'returns empty relation when no prices in range' do
        prices = StockPrice.in_date_range(Date.parse('2024-01-01'), Date.parse('2024-01-02'))
        
        expect(prices).to be_empty
      end
    end
    
    describe '.ordered_by_date' do
      it 'returns prices in chronological order' do
        # Create prices in random order
        create(:stock_price, symbol: 'TEST', date: Date.parse('2023-01-05'), close_price: 100.0)
        create(:stock_price, symbol: 'TEST', date: Date.parse('2023-01-04'), close_price: 100.0)
        create(:stock_price, symbol: 'TEST', date: Date.parse('2023-01-06'), close_price: 100.0)
        
        ordered_prices = StockPrice.for_symbol('TEST').ordered_by_date
        dates = ordered_prices.pluck(:date)
        
        expect(dates).to eq(dates.sort)
      end
    end
    
    describe 'scope chaining' do
      it 'allows chaining of scopes' do
        msft_prices = StockPrice.for_symbol('MSFT')
                                .in_date_range(Date.parse('2023-01-01'), Date.parse('2023-01-02'))
                                .ordered_by_date
        
        expect(msft_prices.count).to eq(2)
        expect(msft_prices.pluck(:symbol).uniq).to eq(['MSFT'])
        expect(msft_prices.first.date).to eq(Date.parse('2023-01-01'))
        expect(msft_prices.last.date).to eq(Date.parse('2023-01-02'))
      end
    end
  end
  
  describe 'class methods' do
    describe '.available_symbols' do
      before do
        create(:stock_price, symbol: 'MSFT', date: Date.parse('2023-01-01'))
        create(:stock_price, symbol: 'AAPL', date: Date.parse('2023-01-01'))
        create(:stock_price, symbol: 'MSFT', date: Date.parse('2023-01-02'))  # Different date
      end
      
      it 'returns unique list of available symbols' do
        symbols = StockPrice.available_symbols
        
        expect(symbols).to contain_exactly('MSFT', 'AAPL')
      end
      
      it 'returns symbols in alphabetical order' do
        create(:stock_price, symbol: 'ZZYX', date: Date.parse('2023-01-03'))
        symbols = StockPrice.available_symbols
        
        expect(symbols).to eq(symbols.sort)
      end
    end
    
    describe '.date_range_for_symbol' do
      before do
        create(:stock_price, symbol: 'MSFT', date: Date.parse('2023-01-01'))
        create(:stock_price, symbol: 'MSFT', date: Date.parse('2023-01-15'))
        create(:stock_price, symbol: 'MSFT', date: Date.parse('2023-01-10'))
      end
      
      it 'returns earliest and latest dates for symbol' do
        start_date, end_date = StockPrice.date_range_for_symbol('MSFT')
        
        expect(start_date).to eq(Date.parse('2023-01-01'))
        expect(end_date).to eq(Date.parse('2023-01-15'))
      end
      
      it 'returns nil for non-existent symbol' do
        start_date, end_date = StockPrice.date_range_for_symbol('INVALID')
        
        expect(start_date).to be_nil
        expect(end_date).to be_nil
      end
    end
  end
  
  describe 'instance methods' do
    let(:stock_price) { create(:stock_price, symbol: 'MSFT', date: Date.parse('2023-01-01'), close: 100.50) }
    
    describe '#to_s' do
      it 'returns formatted string representation' do
        expect(stock_price.to_s).to eq('MSFT - 2023-01-01: $100.5')
      end
    end
    
    describe '#percentage_change_from' do
      let(:previous_price) { create(:stock_price, symbol: 'MSFT', date: Date.parse('2022-12-31'), close: 95.0) }
      
      it 'calculates percentage change from another price' do
        change = stock_price.percentage_change_from(previous_price)
        
        expect(change).to be_within(0.01).of(5.79)  # (100.50 - 95) / 95 * 100
      end
      
      it 'handles zero previous price' do
        zero_price = build(:stock_price, close: 0.0)
        
        expect {
          stock_price.percentage_change_from(zero_price)
        }.to raise_error(ZeroDivisionError)
      end
    end
  end
  
  describe 'database constraints' do
    it 'enforces database-level uniqueness constraint' do
      create(:stock_price, symbol: 'MSFT', date: Date.current)
      
      expect {
        # Try to insert duplicate directly through SQL to bypass Rails validations
        StockPrice.connection.execute(
          "INSERT INTO stock_prices (symbol, date, close, created_at, updated_at) " +
          "VALUES ('MSFT', '#{Date.current}', 100.0, NOW(), NOW())"
        )
      }.to raise_error(ActiveRecord::RecordNotUnique)
    end
  end
  
  describe 'data integrity' do
    it 'automatically converts symbol to uppercase' do
      price = create(:stock_price, symbol: 'msft')
      
      expect(price.reload.symbol).to eq('MSFT')
    end
    
    it 'stores close price with appropriate precision' do
      price = create(:stock_price, close: 123.456789)
      
      # Should preserve decimal precision
      expect(price.reload.close).to be_within(0.000001).of(123.456789)
    end
    
    it 'handles date storage correctly' do
      date = Date.parse('2023-01-01')
      price = create(:stock_price, date: date)
      
      expect(price.reload.date).to eq(date)
    end
  end
end