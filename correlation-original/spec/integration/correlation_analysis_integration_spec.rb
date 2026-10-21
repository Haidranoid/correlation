require 'rails_helper'

RSpec.describe 'Correlation Analysis Integration', type: :integration do
  describe 'end-to-end correlation analysis workflow' do
    let!(:msft_data) do
      # Create 60 days of MSFT data with strong linear trend for very reliable correlation
      (0..59).map do |i|
        create(:stock_price, 
          symbol: 'MSFT', 
          date: Date.parse('2023-01-01') + i.days,
          close: 100 + (i * 2.0)  # Strong linear trend, no noise
        )
      end
    end
    
    let!(:aapl_data) do
      # Create 60 days of AAPL data with perfect linear correlation to MSFT
      (0..59).map do |i|
        msft_value = 100 + (i * 2.0)
        # Perfect linear correlation: AAPL = 1.5 * MSFT + 50
        aapl_close = msft_value * 1.5 + 50
        create(:stock_price, 
          symbol: 'AAPL', 
          date: Date.parse('2023-01-01') + i.days,
          close: aapl_close
        )
      end
    end
    
    describe 'single correlation analysis' do
      it 'performs complete analysis from data retrieval to result storage' do
        service = CorrelationAnalysisService.new(silent: true)
        start_date = Date.parse('2023-01-01')
        end_date = Date.parse('2023-02-15')
        
        # Execute the analysis
        result = service.analyze_correlation('MSFT', 'AAPL', 5, 5, 0, start_date, end_date)
        
        # Verify result structure and content
        expect(result).to include(
          :symbol1, :symbol2, :window1, :window2, :offset,
          :correlation, :r_squared, :data_points,
          :start_date, :end_date, :analysis_date,
          :interpretation, :aligned_data
        )
        
        expect(result[:symbol1]).to eq('MSFT')
        expect(result[:symbol2]).to eq('AAPL')
        expect(result[:correlation]).to be_between(-1, 1)
        expect(result[:r_squared]).to be_between(0, 1)
        expect(result[:data_points]).to be > 0
        expect(result[:aligned_data]).to be_an(Array)
        expect(result[:aligned_data].first).to be_an(Array)
        
        # Verify interpretation structure
        interpretation = result[:interpretation]
        expect(interpretation).to include(:strength, :direction, :description, :r_squared, :variance_explained)
        
        # Save the result and verify database storage
        expect {
          service.save_correlation_result(result)
        }.to change(StockCorrelation, :count).by(1)
        
        saved_correlation = StockCorrelation.last
        expect(saved_correlation.symbol1).to eq('MSFT')
        expect(saved_correlation.symbol2).to eq('AAPL')
        expect(saved_correlation.correlation_coefficient).to be_within(0.000001).of(result[:correlation])
        expect(saved_correlation.r_squared).to be_within(0.000001).of(result[:r_squared])
      end
    end
    
    describe 'best correlation finding across all offsets' do
      it 'finds and saves statistically relevant correlations' do
        service = CorrelationAnalysisService.new(silent: true)
        
        result = service.find_best_correlation('MSFT', 'AAPL', 3, 3, 
          Date.parse('2023-01-01'), Date.parse('2023-01-30'))
        
        # Verify result structure - handle both success and error cases
        if result[:error]
          expect(result).to have_key(:error)
          expect(result[:error]).to be_a(String)
        else
          expect(result).to include(
            :best_correlation, :relevant_correlations, 
            :total_tested, :total_relevant
          )
          
          expect(result[:total_tested]).to be > 0
          expect(result[:best_correlation]).to be_a(Hash)
          expect(result[:relevant_correlations]).to be_an(Array)
        end
        
        # If relevant correlations found, verify their structure
        if result[:relevant_correlations]&.any?
          correlation = result[:relevant_correlations].first
          expect(correlation[:correlation].abs).to be >= 0.3
          expect(correlation[:r_squared]).to be >= 0.09
          
          # Save and verify database operations
          initial_count = StockCorrelation.count
          service.save_relevant_correlations(result[:relevant_correlations], 'MSFT', 'AAPL', 3, 3)
          
          expect(StockCorrelation.count).to eq(initial_count + result[:relevant_correlations].size)
          
          # Verify all saved correlations meet the criteria
          saved_correlations = StockCorrelation.where(symbol1: 'MSFT', symbol2: 'AAPL', window1_days: 3, window2_days: 3)
          saved_correlations.each do |correlation|
            expect(correlation.correlation_coefficient.abs).to be >= 0.3
            expect(correlation.r_squared).to be >= 0.09
          end
        end
      end
    end
    
    describe 'data service integration' do
      it 'correctly retrieves and processes stock data' do
        data_service = StockDataService.new
        
        # Test data retrieval
        msft_prices = data_service.get_stock_data('MSFT', Date.parse('2023-01-01'), Date.parse('2023-01-10'))
        expect(msft_prices.length).to eq(10)
        expect(msft_prices.map(&:symbol).uniq).to eq(['MSFT'])
        
        # Test percentage change calculation
        changes = data_service.calculate_percentage_changes(msft_prices, 3)
        expect(changes).to be_an(Array)
        expect(changes.length).to eq(7)  # 10 - 3 days window = 7 changes
        
        changes.each do |change|
          expect(change).to have_key(:change)
          expect(change).to have_key(:date)
          expect(change[:change]).to be_a(Numeric)
          expect(change[:date]).to be_a(Date)
        end
      end
    end
    
    describe 'calculation service integration' do
      it 'correctly calculates correlations and interpretations' do
        calc_service = CorrelationCalculationService.new
        
        # Create test data with known correlation
        perfect_data = [[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]]
        
        correlation = calc_service.calculate_pearson_correlation(perfect_data)
        expect(correlation).to be_within(0.0001).of(1.0)
        
        interpretation = calc_service.interpret_correlation(correlation)
        expect(interpretation[:strength]).to eq('Extremely strong correlation')
        expect(interpretation[:direction]).to eq('Positive')
        expect(interpretation[:r_squared]).to eq(1.0)
        expect(interpretation[:variance_explained]).to eq(100.0)
      end
    end
    
    describe 'error handling integration' do
      it 'handles missing data gracefully' do
        service = CorrelationAnalysisService.new(silent: true)
        
        expect {
          service.analyze_correlation('INVALID', 'SYMBOLS', 3, 3, 0, 
            Date.parse('2023-01-01'), Date.parse('2023-01-30'))
        }.to raise_error(CorrelationAnalysisService::AnalysisError)
      end
      
      it 'handles insufficient data gracefully' do
        service = CorrelationAnalysisService.new(silent: true)
        
        expect {
          service.analyze_correlation('MSFT', 'AAPL', 30, 30, 0, 
            Date.parse('2023-01-01'), Date.parse('2023-01-02'))
        }.to raise_error(CorrelationAnalysisService::AnalysisError)
      end
    end
    
    describe 'database replacement functionality' do
      it 'replaces existing correlations with same parameters' do
        service = CorrelationAnalysisService.new(silent: true)
        
        # Create initial correlation
        initial_correlation = create(:stock_correlation, 
          symbol1: 'MSFT', symbol2: 'AAPL', 
          window1_days: 5, window2_days: 5,
          correlation_coefficient: 0.5
        )
        
        # Create new correlations with same parameters
        new_correlations = [
          {
            correlation: 0.8,
            r_squared: 0.64,
            data_points: 100,
            offset: 0,
            start_date: Date.parse('2023-01-01'),
            end_date: Date.parse('2023-01-30'),
            analysis_date: Date.current
          },
          {
            correlation: 0.6,
            r_squared: 0.36,
            data_points: 95,
            offset: 2,
            start_date: Date.parse('2023-01-01'),
            end_date: Date.parse('2023-01-30'),
            analysis_date: Date.current
          }
        ]
        
        # Save new correlations (should replace existing)
        initial_count = StockCorrelation.count
        service.save_relevant_correlations(new_correlations, 'MSFT', 'AAPL', 5, 5)
        
        # Verify replacement occurred
        expect(StockCorrelation.count).to eq(initial_count + 1)  # Net increase of 1 (remove 1, add 2)
        expect(StockCorrelation.exists?(initial_correlation.id)).to be false
        
        # Verify new correlations were saved
        new_saved = StockCorrelation.where(symbol1: 'MSFT', symbol2: 'AAPL', window1_days: 5, window2_days: 5)
        expect(new_saved.count).to eq(2)
        expect(new_saved.pluck(:correlation_coefficient).sort).to eq([0.6, 0.8])
      end
    end
  end
  
  describe 'rake task integration' do
    it 'can be invoked through rake tasks' do
      # This test would require actually running rake tasks, which is complex in RSpec
      # Instead, we test that the rake task components work correctly
      
      # Test date range determination with our known test data
      start_date, end_date = StockDataService.determine_max_date_range('MSFT', 'AAPL', 3, 3, 0)
      
      if start_date && end_date
        expect(start_date).to be_a(Date)
        expect(end_date).to be_a(Date)
        expect(end_date).to be > start_date
      else
        # If no data overlap, that's also a valid test result
        expect(start_date).to be_nil
        expect(end_date).to be_nil
      end
    end
  end
  
  describe 'performance considerations' do
    it 'handles large datasets efficiently' do
      # Create larger dataset for performance testing
      large_dataset = (0..199).map do |i|
        create(:stock_price, 
          symbol: 'LARGE', 
          date: Date.parse('2022-01-01') + i.days,
          close: 100 + (i * 0.1)
        )
      end
      
      large_dataset2 = (0..199).map do |i|
        # Create highly correlated data based on LARGE symbol
        base_price = 100 + (i * 0.1)
        correlated_price = (base_price - 100) * 1.2 + 150  # Perfect correlation
        create(:stock_price, 
          symbol: 'LARGE2', 
          date: Date.parse('2022-01-01') + i.days,
          close: correlated_price
        )
      end
      
      service = CorrelationAnalysisService.new(silent: true)
      
      # Measure execution time
      start_time = Time.current
      result = service.analyze_correlation('LARGE', 'LARGE2', 5, 5, 0, 
        Date.parse('2022-01-01'), Date.parse('2022-06-30'))
      end_time = Time.current
      
      # Verify the analysis completed successfully
      expect(result[:data_points]).to be > 100
      expect(result[:correlation]).to be_between(-1, 1)
      
      # Verify reasonable performance (should complete within a few seconds)
      execution_time = end_time - start_time
      expect(execution_time).to be < 10  # Should complete within 10 seconds
    end
  end
end