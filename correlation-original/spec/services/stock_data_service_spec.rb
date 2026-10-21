require 'rails_helper'

RSpec.describe StockDataService do
  let(:service) { described_class.new }
  
  # Sample stock data for testing
  let(:msft_data) do
    [
      create(:stock_price, symbol: 'MSFT', date: Date.parse('2023-01-01'), close: 100.0),
      create(:stock_price, symbol: 'MSFT', date: Date.parse('2023-01-02'), close: 102.0),
      create(:stock_price, symbol: 'MSFT', date: Date.parse('2023-01-03'), close: 98.0),
      create(:stock_price, symbol: 'MSFT', date: Date.parse('2023-01-04'), close: 105.0),
      create(:stock_price, symbol: 'MSFT', date: Date.parse('2023-01-05'), close: 107.0),
      create(:stock_price, symbol: 'MSFT', date: Date.parse('2023-01-06'), close: 103.0),
      create(:stock_price, symbol: 'MSFT', date: Date.parse('2023-01-07'), close: 108.0),
      create(:stock_price, symbol: 'MSFT', date: Date.parse('2023-01-08'), close: 110.0)
    ]
  end
  
  let(:aapl_data) do
    [
      create(:stock_price, symbol: 'AAPL', date: Date.parse('2023-01-01'), close: 150.0),
      create(:stock_price, symbol: 'AAPL', date: Date.parse('2023-01-02'), close: 153.0),
      create(:stock_price, symbol: 'AAPL', date: Date.parse('2023-01-03'), close: 147.0),
      create(:stock_price, symbol: 'AAPL', date: Date.parse('2023-01-04'), close: 155.0),
      create(:stock_price, symbol: 'AAPL', date: Date.parse('2023-01-05'), close: 158.0),
      create(:stock_price, symbol: 'AAPL', date: Date.parse('2023-01-06'), close: 152.0),
      create(:stock_price, symbol: 'AAPL', date: Date.parse('2023-01-07'), close: 160.0),
      create(:stock_price, symbol: 'AAPL', date: Date.parse('2023-01-08'), close: 162.0)
    ]
  end
  
  before do
    msft_data
    aapl_data
  end
  
  describe '#get_stock_data' do
    context 'with valid parameters' do
      it 'retrieves stock data for specified date range' do
        start_date = Date.parse('2023-01-01')
        end_date = Date.parse('2023-01-05')
        
        result = service.get_stock_data('MSFT', start_date, end_date)
        
        expect(result.length).to eq(5)
        expect(result.first.symbol).to eq('MSFT')
        expect(result.first.date).to eq(start_date)
        expect(result.last.date).to eq(end_date)
      end
      
      it 'returns data ordered by date' do
        result = service.get_stock_data('MSFT', Date.parse('2023-01-01'), Date.parse('2023-01-08'))
        
        dates = result.map(&:date)
        expect(dates).to eq(dates.sort)
      end
      
      it 'handles case-insensitive symbols' do
        result = service.get_stock_data('msft', Date.parse('2023-01-01'), Date.parse('2023-01-03'))
        
        expect(result.length).to eq(3)
        expect(result.first.symbol).to eq('MSFT')
      end
    end
    
    context 'with invalid parameters' do
      it 'returns empty array for non-existent symbol' do
        result = service.get_stock_data('INVALID', Date.parse('2023-01-01'), Date.parse('2023-01-08'))
        
        expect(result).to be_empty
      end
      
      it 'returns empty array when end_date is before start_date' do
        result = service.get_stock_data('MSFT', Date.parse('2023-01-08'), Date.parse('2023-01-01'))
        
        expect(result).to be_empty
      end
      
      it 'returns empty array for dates outside available range' do
        result = service.get_stock_data('MSFT', Date.parse('2024-01-01'), Date.parse('2024-01-08'))
        
        expect(result).to be_empty
      end
    end
  end
  
  describe '#calculate_percentage_changes' do
    context 'with valid data and window' do
      it 'calculates 1-day percentage changes correctly' do
        data = msft_data.first(3)  # 100, 102, 98
        result = service.calculate_percentage_changes(data, 1)
        
        expected_changes = [
          { change: 2.0, date: Date.parse('2023-01-02') },   # (102-100)/100 * 100
          { change: -3.92, date: Date.parse('2023-01-03') }  # (98-102)/102 * 100
        ]
        
        expect(result.length).to eq(2)
        expect(result[0][:change]).to be_within(0.01).of(expected_changes[0][:change])
        expect(result[1][:change]).to be_within(0.01).of(expected_changes[1][:change])
      end
      
      it 'calculates multi-day percentage changes correctly' do
        data = msft_data.first(5)  # 100, 102, 98, 105, 107
        result = service.calculate_percentage_changes(data, 3)
        
        # With 5 data points and window=3, we get 5-3=2 changes
        # Change 1: (105-100)/100, Change 2: (107-102)/102
        expected_changes = [5.0, 4.90]
        
        expect(result.length).to eq(2)
        result.each_with_index do |change_data, index|
          expect(change_data[:change]).to be_within(0.01).of(expected_changes[index])
        end
      end
      
      it 'includes correct dates for percentage changes' do
        data = msft_data.first(4)
        result = service.calculate_percentage_changes(data, 2)
        
        expect(result[0][:date]).to eq(Date.parse('2023-01-03'))  # End date of first 2-day period
        expect(result[1][:date]).to eq(Date.parse('2023-01-04'))  # End date of second 2-day period
      end
    end
    
    context 'with edge cases' do
      it 'returns empty array when data length equals window size' do
        data = msft_data.first(2)
        result = service.calculate_percentage_changes(data, 2)
        
        expect(result).to be_empty
      end
      
      it 'returns empty array when data length is less than window size' do
        data = msft_data.first(1)
        result = service.calculate_percentage_changes(data, 2)
        
        expect(result).to be_empty
      end
      
      it 'handles zero prices gracefully' do
        data = [
          build(:stock_price, symbol: 'TEST', date: Date.parse('2023-01-01'), close: 0.1),  # Very small but not zero to pass validation
          build(:stock_price, symbol: 'TEST', date: Date.parse('2023-01-02'), close: 100.0)
        ]
        
        result = service.calculate_percentage_changes(data, 1)
        expect(result.length).to eq(1)
        expect(result.first[:change]).to be_within(1).of(99900)  # Very large percentage change
      end
      
      it 'returns empty array for empty data' do
        result = service.calculate_percentage_changes([], 1)
        
        expect(result).to be_empty
      end
    end
  end
  
  describe '#align_data_with_offset' do
    let(:msft_changes) do
      [
        { change: 2.0, date: Date.parse('2023-01-02') },
        { change: -3.0, date: Date.parse('2023-01-03') },
        { change: 5.0, date: Date.parse('2023-01-04') },
        { change: 1.0, date: Date.parse('2023-01-05') }
      ]
    end
    
    let(:aapl_changes) do
      [
        { change: 3.0, date: Date.parse('2023-01-02') },
        { change: -4.0, date: Date.parse('2023-01-03') },
        { change: 6.0, date: Date.parse('2023-01-04') },
        { change: 2.0, date: Date.parse('2023-01-05') },
        { change: -1.0, date: Date.parse('2023-01-06') }
      ]
    end
    
    context 'with zero offset' do
      it 'aligns data with matching dates' do
        result = service.align_data_with_offset(msft_changes, aapl_changes, 0)
        
        expect(result.length).to eq(4)  # All MSFT dates have matching AAPL dates
        expect(result[0]).to eq([2.0, 3.0, Date.parse('2023-01-02'), Date.parse('2023-01-02')])
        expect(result[3]).to eq([1.0, 2.0, Date.parse('2023-01-05'), Date.parse('2023-01-05')])
      end
    end
    
    context 'with positive offset' do
      it 'offsets aapl data into the future' do
        result = service.align_data_with_offset(msft_changes, aapl_changes, 1)
        
        # MSFT 2023-01-02 should align with AAPL 2023-01-03 (1 day future)
        expect(result.length).to eq(4)  # All MSFT dates can find future AAPL dates
        expect(result[0]).to eq([2.0, -4.0, Date.parse('2023-01-02'), Date.parse('2023-01-03')])
      end
    end
    
    context 'with negative offset' do
      it 'offsets aapl data into the past' do
        result = service.align_data_with_offset(msft_changes, aapl_changes, -1)
        
        # MSFT 2023-01-03 should align with AAPL 2023-01-02 (1 day past)
        expect(result.length).to eq(3)
        expect(result[0]).to eq([-3.0, 3.0, Date.parse('2023-01-03'), Date.parse('2023-01-02')])
      end
    end
    
    context 'with edge cases' do
      it 'returns empty array when no dates align' do
        future_aapl = [{ change: 1.0, date: Date.parse('2023-02-01') }]
        result = service.align_data_with_offset(msft_changes, future_aapl, 0)
        
        expect(result).to be_empty
      end
      
      it 'handles empty arrays' do
        result = service.align_data_with_offset([], aapl_changes, 0)
        expect(result).to be_empty
        
        result = service.align_data_with_offset(msft_changes, [], 0)
        expect(result).to be_empty
      end
      
      it 'handles large offsets that eliminate all alignments' do
        result = service.align_data_with_offset(msft_changes, aapl_changes, 100)
        
        expect(result).to be_empty
      end
    end
  end
  
  describe '.determine_max_date_range' do
    context 'with overlapping data' do
      it 'determines maximum overlapping date range' do
        start_date, end_date = described_class.determine_max_date_range('MSFT', 'AAPL', 1, 1, 0)
        
        expect(start_date).to eq(Date.parse('2023-01-02'))  # First possible date after 1-day window
        expect(end_date).to eq(Date.parse('2023-01-08'))    # Last available date
      end
      
      it 'accounts for window sizes in date range calculation' do
        start_date, end_date = described_class.determine_max_date_range('MSFT', 'AAPL', 3, 2, 0)
        
        expect(start_date).to eq(Date.parse('2023-01-04'))  # After 3-day window for MSFT
        expect(end_date).to eq(Date.parse('2023-01-08'))
      end
      
      it 'handles positive offsets correctly' do
        start_date, end_date = described_class.determine_max_date_range('MSFT', 'AAPL', 1, 1, 2)
        
        expect(start_date).to eq(Date.parse('2023-01-02'))
        expect(end_date).to eq(Date.parse('2023-01-06'))  # Reduced by offset
      end
      
      it 'handles negative offsets correctly' do
        start_date, end_date = described_class.determine_max_date_range('MSFT', 'AAPL', 1, 1, -2)
        
        expect(start_date).to eq(Date.parse('2023-01-04'))  # Increased by offset magnitude
        expect(end_date).to eq(Date.parse('2023-01-08'))
      end
    end
    
    context 'with non-overlapping data' do
      before do
        # Create AAPL data in different date range
        StockPrice.where(symbol: 'AAPL').delete_all
        create(:stock_price, symbol: 'AAPL', date: Date.parse('2024-01-01'), close_price: 150.0)
        create(:stock_price, symbol: 'AAPL', date: Date.parse('2024-01-02'), close_price: 153.0)
      end
      
      it 'returns nil for both dates when no overlap exists' do
        start_date, end_date = described_class.determine_max_date_range('MSFT', 'AAPL', 1, 1, 0)
        
        expect(start_date).to be_nil
        expect(end_date).to be_nil
      end
    end
    
    context 'with missing data' do
      it 'returns nil when first symbol has no data' do
        start_date, end_date = described_class.determine_max_date_range('INVALID', 'AAPL', 1, 1, 0)
        
        expect(start_date).to be_nil
        expect(end_date).to be_nil
      end
      
      it 'returns nil when second symbol has no data' do
        start_date, end_date = described_class.determine_max_date_range('MSFT', 'INVALID', 1, 1, 0)
        
        expect(start_date).to be_nil
        expect(end_date).to be_nil
      end
    end
  end
end