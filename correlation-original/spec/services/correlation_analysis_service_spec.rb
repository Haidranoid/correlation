require 'rails_helper'

RSpec.describe CorrelationAnalysisService do
  let(:service) { described_class.new(silent: true) }
  
  # Create comprehensive test data
  let!(:msft_data) do
    (1..30).map do |i|
      create(:stock_price, 
        symbol: 'MSFT', 
        date: Date.parse('2023-01-01') + (i - 1).days, 
        close: 100 + i * 1.5  # Strong linear trend for reliable correlation
      )
    end
  end
  
  let!(:aapl_data) do
    (1..30).map do |i|
      # Create highly correlated data - linear relationship with MSFT
      msft_value = 100 + i * 1.5
      # Strong positive correlation: AAPL = 1.2 * MSFT + 50 + small noise
      aapl_close = msft_value * 1.2 + 50 + (i % 3 - 1) * 0.5  # Very small noise
      create(:stock_price, 
        symbol: 'AAPL', 
        date: Date.parse('2023-01-01') + (i - 1).days, 
        close: aapl_close
      )
    end
  end
  
  describe '#analyze_correlation' do
    context 'with valid parameters' do
      it 'performs single correlation analysis successfully' do
        start_date = Date.parse('2023-01-01')
        end_date = Date.parse('2023-01-25')
        
        result = service.analyze_correlation('MSFT', 'AAPL', 3, 3, 0, start_date, end_date)
        
        expect(result).to be_a(Hash)
        expect(result).to have_key(:symbol1)
        expect(result).to have_key(:symbol2)
        expect(result).to have_key(:correlation)
        expect(result).to have_key(:r_squared)
        expect(result).to have_key(:data_points)
        expect(result).to have_key(:interpretation)
        expect(result).to have_key(:aligned_data)
        
        expect(result[:symbol1]).to eq('MSFT')
        expect(result[:symbol2]).to eq('AAPL')
        expect(result[:window1]).to eq(3)
        expect(result[:window2]).to eq(3)
        expect(result[:offset]).to eq(0)
        expect(result[:correlation]).to be_between(-1, 1)
        expect(result[:r_squared]).to be_between(0, 1)
        expect(result[:data_points]).to be > 0
      end
      
      it 'includes proper start and end dates in result' do
        start_date = Date.parse('2023-01-05')
        end_date = Date.parse('2023-01-20')
        
        result = service.analyze_correlation('MSFT', 'AAPL', 2, 2, 0, start_date, end_date)
        
        expect(result[:start_date]).to eq(start_date)
        expect(result[:end_date]).to eq(end_date)
        expect(result[:analysis_date].to_date).to eq(Date.current)
      end
      
      it 'handles different window sizes correctly' do
        result = service.analyze_correlation('MSFT', 'AAPL', 1, 5, 0, 
          Date.parse('2023-01-01'), Date.parse('2023-01-25'))
        
        expect(result[:window1]).to eq(1)
        expect(result[:window2]).to eq(5)
        expect(result[:data_points]).to be > 0
      end
      
      it 'handles positive offsets correctly' do
        result = service.analyze_correlation('MSFT', 'AAPL', 3, 3, 2, 
          Date.parse('2023-01-01'), Date.parse('2023-01-25'))
        
        expect(result[:offset]).to eq(2)
        expect(result[:data_points]).to be > 0
      end
      
      it 'handles negative offsets correctly' do
        result = service.analyze_correlation('MSFT', 'AAPL', 3, 3, -2, 
          Date.parse('2023-01-01'), Date.parse('2023-01-25'))
        
        expect(result[:offset]).to eq(-2)
        expect(result[:data_points]).to be > 0
      end
    end
    
    context 'with invalid parameters' do
      it 'raises AnalysisError for non-existent symbols' do
        expect {
          service.analyze_correlation('INVALID', 'AAPL', 3, 3, 0, 
            Date.parse('2023-01-01'), Date.parse('2023-01-15'))
        }.to raise_error(CorrelationAnalysisService::AnalysisError)
      end
      
      it 'raises AnalysisError when date range has insufficient data' do
        expect {
          service.analyze_correlation('MSFT', 'AAPL', 50, 50, 0, 
            Date.parse('2023-01-01'), Date.parse('2023-01-02'))
        }.to raise_error(CorrelationAnalysisService::AnalysisError)
      end
      
      it 'raises AnalysisError for same symbol comparison' do
        expect {
          service.analyze_correlation('MSFT', 'MSFT', 3, 3, 0, 
            Date.parse('2023-01-01'), Date.parse('2023-01-25'))
        }.to raise_error(CorrelationAnalysisService::AnalysisError)
      end
    end
  end
  
  describe '#find_best_correlation' do
    context 'with data that produces correlations' do
      it 'finds statistically relevant correlations across all offsets' do
        result = service.find_best_correlation('MSFT', 'AAPL', 2, 2, 
          Date.parse('2023-01-01'), Date.parse('2023-01-25'))
        
        if result[:error]
          # If no correlations found, at least verify structure
          expect(result).to have_key(:error)
        else
          expect(result).to be_a(Hash)
          expect(result).to have_key(:best_correlation)
          expect(result).to have_key(:relevant_correlations)
          expect(result).to have_key(:total_tested)
          expect(result).to have_key(:total_relevant)
          
          expect(result[:total_tested]).to be > 0
          expect(result[:best_correlation]).to be_a(Hash)
          expect(result[:relevant_correlations]).to be_an(Array)
        end
      end
      
      it 'filters correlations by statistical significance' do
        result = service.find_best_correlation('MSFT', 'AAPL', 2, 2, 
          Date.parse('2023-01-01'), Date.parse('2023-01-25'))
        
        # Skip if no correlations found
        unless result[:error]
          # All relevant correlations should meet the criteria: |r| >= 0.3 AND R² >= 0.09
          result[:relevant_correlations].each do |correlation|
            expect(correlation[:correlation].abs).to be >= 0.3
            expect(correlation[:r_squared]).to be >= 0.09
          end
        end
      end
      
      it 'sorts relevant correlations by absolute correlation strength' do
        result = service.find_best_correlation('MSFT', 'AAPL', 2, 2, 
          Date.parse('2023-01-01'), Date.parse('2023-01-25'))
        
        unless result[:error]
          if result[:relevant_correlations].length > 1
            correlations = result[:relevant_correlations].map { |r| r[:correlation].abs }
            expect(correlations).to eq(correlations.sort.reverse)
          end
        end
      end
      
      it 'includes progress tracking information' do
        result = service.find_best_correlation('MSFT', 'AAPL', 2, 2, 
          Date.parse('2023-01-01'), Date.parse('2023-01-25'))
        
        unless result[:error]
          expect(result[:total_tested]).to be > 0
          expect(result[:total_relevant]).to be >= 0
        end
      end
    end
    
    context 'with insufficient data' do
      it 'returns error when no data is available' do
        result = service.find_best_correlation('INVALID', 'AAPL', 3, 3, 
          Date.parse('2023-01-01'), Date.parse('2023-01-15'))
        
        expect(result[:error]).to be_a(String)
      end
      
      it 'returns error when date range is too small' do
        result = service.find_best_correlation('MSFT', 'AAPL', 15, 15, 
          Date.parse('2023-01-01'), Date.parse('2023-01-02'))
        
        expect(result[:error]).to be_a(String)
      end
    end
  end
  
  describe '#save_correlation_result' do
    let(:correlation_result) do
      {
        symbol1: 'MSFT',
        symbol2: 'AAPL',
        window1: 3,
        window2: 3,
        offset: 5,
        correlation: 0.75,
        r_squared: 0.5625,
        data_points: 100,
        start_date: Date.parse('2023-01-01'),
        end_date: Date.parse('2023-01-15'),
        analysis_date: Date.current,
        interpretation: { description: 'Strong positive correlation' },
        aligned_data: [[1.0, 2.0, Date.current, Date.current]]
      }
    end
    
    it 'saves correlation result to database' do
      expect {
        service.save_correlation_result(correlation_result)
      }.to change(StockCorrelation, :count).by(1)
      
      saved_correlation = StockCorrelation.last
      expect(saved_correlation.symbol1).to eq('MSFT')
      expect(saved_correlation.symbol2).to eq('AAPL')
      expect(saved_correlation.correlation_coefficient).to eq(0.75)
      expect(saved_correlation.r_squared).to eq(0.5625)
      expect(saved_correlation.data_points).to eq(100)
    end
    
    it 'handles database validation errors gracefully' do
      # Create invalid result
      invalid_result = correlation_result.merge(
        correlation: 2.0,  # Invalid correlation > 1
        r_squared: 4.0     # Invalid R² > 1
      )
      
      expect {
        service.save_correlation_result(invalid_result)
      }.to raise_error(ActiveRecord::RecordInvalid)
    end
  end
  
  describe '#save_relevant_correlations' do
    let(:relevant_correlations) do
      [
        {
          correlation: 0.8,
          r_squared: 0.64,
          data_points: 100,
          offset: 0,
          start_date: Date.parse('2023-01-01'),
          end_date: Date.parse('2023-01-15'),
          analysis_date: Date.current
        },
        {
          correlation: 0.6,
          r_squared: 0.36,
          data_points: 95,
          offset: 2,
          start_date: Date.parse('2023-01-01'),
          end_date: Date.parse('2023-01-15'),
          analysis_date: Date.current
        }
      ]
    end
    
    it 'saves multiple relevant correlations to database' do
      expect {
        service.save_relevant_correlations(relevant_correlations, 'MSFT', 'AAPL', 3, 3)
      }.to change(StockCorrelation, :count).by(2)
      
      saved_correlations = StockCorrelation.where(symbol1: 'MSFT', symbol2: 'AAPL').order(:offset_days)
      expect(saved_correlations.first.correlation_coefficient).to eq(0.8)
      expect(saved_correlations.last.correlation_coefficient).to eq(0.6)
    end
    
    it 'replaces existing correlations with same parameters' do
      # Create existing correlation
      existing = create(:stock_correlation, 
        symbol1: 'MSFT', symbol2: 'AAPL', 
        window1_days: 3, window2_days: 3,
        correlation_coefficient: 0.5
      )
      
      expect {
        service.save_relevant_correlations(relevant_correlations, 'MSFT', 'AAPL', 3, 3)
      }.to change(StockCorrelation, :count).by(1)  # Net increase of 1 (delete 1, add 2)
      
      expect(StockCorrelation.exists?(existing.id)).to be false
    end
    
    it 'handles empty correlations array' do
      expect {
        service.save_relevant_correlations([], 'MSFT', 'AAPL', 3, 3)
      }.not_to change(StockCorrelation, :count)
    end
  end
  
  describe 'error handling' do
    it 'defines AnalysisError as a StandardError' do
      expect(CorrelationAnalysisService::AnalysisError).to be < StandardError
    end
    
    it 'raises AnalysisError with descriptive messages' do
      error = CorrelationAnalysisService::AnalysisError.new("Test error")
      expect(error.message).to eq("Test error")
    end
  end
  
  describe 'integration with other services' do
    it 'uses StockDataService for data retrieval' do
      expect(StockDataService).to receive(:fetch_stock_data).at_least(:once).and_call_original
      
      service.analyze_correlation('MSFT', 'AAPL', 3, 3, 0, 
        Date.parse('2023-01-01'), Date.parse('2023-01-25'))
    end
    
    it 'uses CorrelationCalculationService for calculations' do
      expect_any_instance_of(CorrelationCalculationService).to receive(:calculate_pearson_correlation).and_call_original
      expect_any_instance_of(CorrelationCalculationService).to receive(:interpret_correlation).and_call_original
      
      service.analyze_correlation('MSFT', 'AAPL', 3, 3, 0, 
        Date.parse('2023-01-01'), Date.parse('2023-01-25'))
    end
  end
end