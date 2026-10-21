require 'rails_helper'

RSpec.describe StockCorrelation, type: :model do
  describe 'validations' do
    subject { build(:stock_correlation) }
    
    describe 'presence validations' do
      it { should validate_presence_of(:symbol1) }
      it { should validate_presence_of(:symbol2) }
      it { should validate_presence_of(:window1_days) }
      it { should validate_presence_of(:window2_days) }
      it { should validate_presence_of(:offset_days) }
      it { should validate_presence_of(:correlation_coefficient) }
      it { should validate_presence_of(:r_squared) }
      it { should validate_presence_of(:data_points) }
      it { should validate_presence_of(:start_date) }
      it { should validate_presence_of(:end_date) }
      it { should validate_presence_of(:analysis_date) }
    end
    
    describe 'length validations' do
      it { should validate_length_of(:symbol1).is_at_most(10) }
      it { should validate_length_of(:symbol2).is_at_most(10) }
    end
    
    describe 'numericality validations' do
      it { should validate_numericality_of(:window1_days).is_greater_than(0).is_less_than_or_equal_to(365) }
      it { should validate_numericality_of(:window2_days).is_greater_than(0).is_less_than_or_equal_to(365) }
      it { should validate_numericality_of(:offset_days).only_integer }
      it { should validate_numericality_of(:correlation_coefficient).is_greater_than_or_equal_to(-1.0).is_less_than_or_equal_to(1.0) }
      it { should validate_numericality_of(:r_squared).is_greater_than_or_equal_to(0.0).is_less_than_or_equal_to(1.0) }
      it { should validate_numericality_of(:data_points).is_greater_than(0) }
    end
    
    describe 'custom validations' do
      describe 'end_date_after_start_date' do
        it 'is valid when end_date is after start_date' do
          correlation = build(:stock_correlation, 
            start_date: Date.parse('2023-01-01'), 
            end_date: Date.parse('2023-01-31')
          )
          
          expect(correlation).to be_valid
        end
        
        it 'is invalid when end_date equals start_date' do
          correlation = build(:stock_correlation, 
            start_date: Date.parse('2023-01-01'), 
            end_date: Date.parse('2023-01-01')
          )
          
          expect(correlation).not_to be_valid
          expect(correlation.errors[:end_date]).to include('must be after start date')
        end
        
        it 'is invalid when end_date is before start_date' do
          correlation = build(:stock_correlation, 
            start_date: Date.parse('2023-01-31'), 
            end_date: Date.parse('2023-01-01')
          )
          
          expect(correlation).not_to be_valid
          expect(correlation.errors[:end_date]).to include('must be after start date')
        end
      end
      
      describe 'symbols_are_different' do
        it 'is valid when symbols are different' do
          correlation = build(:stock_correlation, symbol1: 'MSFT', symbol2: 'AAPL')
          
          expect(correlation).to be_valid
        end
        
        it 'is invalid when symbols are the same' do
          correlation = build(:stock_correlation, symbol1: 'MSFT', symbol2: 'MSFT')
          
          expect(correlation).not_to be_valid
          expect(correlation.errors[:symbol2]).to include('must be different from symbol1')
        end
        
        it 'is case-insensitive for symbol comparison' do
          correlation = build(:stock_correlation, symbol1: 'MSFT', symbol2: 'msft')
          
          expect(correlation).not_to be_valid  # Same symbol, different case should be invalid
          expect(correlation.errors[:symbol2]).to include('must be different from symbol1')
        end
      end
    end
  end
  
  describe 'scopes' do
    let!(:msft_aapl_strong) { create(:stock_correlation, 
      symbol1: 'MSFT', symbol2: 'AAPL', 
      correlation_coefficient: 0.8, r_squared: 0.64) }
    let!(:msft_googl_weak) { create(:stock_correlation, 
      symbol1: 'MSFT', symbol2: 'GOOGL', 
      correlation_coefficient: 0.3, r_squared: 0.09) }
    let!(:aapl_googl_negative) { create(:stock_correlation, 
      symbol1: 'AAPL', symbol2: 'GOOGL', 
      correlation_coefficient: -0.6, r_squared: 0.36) }
    let!(:old_correlation) { create(:stock_correlation, 
      symbol1: 'TSLA', symbol2: 'NVDA',
      correlation_coefficient: 0.5, r_squared: 0.25,
      analysis_date: 60.days.ago) }
    
    describe '.by_symbol_pair' do
      it 'returns correlations for specific symbol pair' do
        correlations = StockCorrelation.by_symbol_pair('MSFT', 'AAPL')
        
        expect(correlations.count).to eq(1)
        expect(correlations.first).to eq(msft_aapl_strong)
      end
      
      it 'returns empty relation for non-existent pair' do
        correlations = StockCorrelation.by_symbol_pair('INVALID', 'SYMBOLS')
        
        expect(correlations).to be_empty
      end
    end
    
    describe '.by_correlation_strength' do
      it 'returns correlations ordered by correlation coefficient descending' do
        correlations = StockCorrelation.by_correlation_strength.limit(4)
        
        expect(correlations.first).to eq(msft_aapl_strong)  # 0.8
        expect(correlations.second).to eq(old_correlation)  # 0.5  
        expect(correlations.third).to eq(msft_googl_weak)  # 0.3
        expect(correlations.fourth).to eq(aapl_googl_negative)  # -0.6
      end
    end
    
    describe '.strong_correlations' do
      it 'returns correlations with absolute value >= 0.7' do
        correlations = StockCorrelation.strong_correlations
        
        expect(correlations.count).to eq(1)
        expect(correlations.first).to eq(msft_aapl_strong)
      end
      
      it 'includes strong negative correlations' do
        create(:stock_correlation, 
          symbol1: 'AMC', symbol2: 'GME',
          correlation_coefficient: -0.85, r_squared: 0.7225)
        correlations = StockCorrelation.strong_correlations
        
        expect(correlations.count).to eq(2)
      end
    end
    
    describe '.recent' do
      it 'returns correlations analyzed within last 30 days' do
        correlations = StockCorrelation.recent
        
        expect(correlations.count).to eq(3)  # Excludes old_correlation
        expect(correlations).not_to include(old_correlation)
      end
    end
  end
  
  describe 'instance methods' do
    describe '#correlation_strength' do
      it 'returns correct strength for no correlation' do
        correlation = build(:stock_correlation, correlation_coefficient: 0.05)
        
        expect(correlation.correlation_strength).to eq('No correlation')
      end
      
      it 'returns correct strength for weak correlation' do
        correlation = build(:stock_correlation, correlation_coefficient: 0.25)
        
        expect(correlation.correlation_strength).to eq('Weak correlation')
      end
      
      it 'returns correct strength for moderate correlation' do
        correlation = build(:stock_correlation, correlation_coefficient: 0.45)
        
        expect(correlation.correlation_strength).to eq('Moderate correlation')
      end
      
      it 'returns correct strength for strong correlation' do
        correlation = build(:stock_correlation, correlation_coefficient: 0.65)
        
        expect(correlation.correlation_strength).to eq('Strong correlation')
      end
      
      it 'returns correct strength for very strong correlation' do
        correlation = build(:stock_correlation, correlation_coefficient: 0.85)
        
        expect(correlation.correlation_strength).to eq('Very strong correlation')
      end
      
      it 'returns correct strength for extremely strong correlation' do
        correlation = build(:stock_correlation, correlation_coefficient: 0.95)
        
        expect(correlation.correlation_strength).to eq('Extremely strong correlation')
      end
      
      it 'uses absolute value for negative correlations' do
        correlation = build(:stock_correlation, correlation_coefficient: -0.85)
        
        expect(correlation.correlation_strength).to eq('Very strong correlation')
      end
      
      it 'returns invalid correlation for out-of-range values' do
        # This shouldn't happen due to validations, but test the method behavior
        correlation = build(:stock_correlation)
        allow(correlation).to receive(:correlation_coefficient).and_return(1.5)
        
        expect(correlation.correlation_strength).to eq('Invalid correlation')
      end
    end
    
    describe '#correlation_direction' do
      it 'returns Positive for positive correlations' do
        correlation = build(:stock_correlation, correlation_coefficient: 0.75)
        
        expect(correlation.correlation_direction).to eq('Positive')
      end
      
      it 'returns Negative for negative correlations' do
        correlation = build(:stock_correlation, correlation_coefficient: -0.75)
        
        expect(correlation.correlation_direction).to eq('Negative')
      end
      
      it 'returns Negative for zero correlation' do
        correlation = build(:stock_correlation, correlation_coefficient: 0.0)
        
        expect(correlation.correlation_direction).to eq('Negative')
      end
    end
    
    describe '#analysis_period_days' do
      it 'calculates correct number of days between start and end dates' do
        correlation = build(:stock_correlation, 
          start_date: Date.parse('2023-01-01'), 
          end_date: Date.parse('2023-01-31')
        )
        
        expect(correlation.analysis_period_days).to eq(30)
      end
      
      it 'handles single day analysis' do
        correlation = build(:stock_correlation, 
          start_date: Date.parse('2023-01-01'), 
          end_date: Date.parse('2023-01-02')
        )
        
        expect(correlation.analysis_period_days).to eq(1)
      end
    end
  end
  
  describe 'data consistency' do
    it 'automatically stores symbols in uppercase' do
      correlation = create(:stock_correlation, 
        symbol1: 'msft', symbol2: 'aapl',
        correlation_coefficient: 0.5, r_squared: 0.25)
      
      expect(correlation.reload.symbol1).to eq('MSFT')
      expect(correlation.reload.symbol2).to eq('AAPL')
    end
    
    it 'preserves decimal precision for correlation coefficients' do
      correlation = create(:stock_correlation, 
        correlation_coefficient: 0.123456789, r_squared: 0.015)
      
      expect(correlation.reload.correlation_coefficient).to be_within(0.000001).of(0.123456789)
    end
    
    it 'preserves decimal precision for R-squared values' do
      correlation = build(:stock_correlation, 
        correlation_coefficient: 0.99, r_squared: 0.987654321)
      correlation.save(validate: false)  # Skip factory callback
      correlation.update_column(:r_squared, 0.987654321)  # Set precise value
      
      expect(correlation.reload.r_squared).to be_within(0.000001).of(0.987654321)
    end
  end
  
  describe 'edge cases' do
    it 'handles extreme correlation values' do
      perfect_positive = build(:stock_correlation, correlation_coefficient: 1.0)
      perfect_negative = build(:stock_correlation, correlation_coefficient: -1.0)
      
      expect(perfect_positive).to be_valid
      expect(perfect_negative).to be_valid
    end
    
    it 'handles extreme R-squared values' do
      min_r_squared = build(:stock_correlation, correlation_coefficient: 0.0, r_squared: 0.0)
      max_r_squared = build(:stock_correlation, correlation_coefficient: 1.0, r_squared: 1.0)
      
      expect(min_r_squared).to be_valid
      expect(max_r_squared).to be_valid
    end
    
    it 'handles large offset values' do
      large_positive_offset = build(:stock_correlation, 
        correlation_coefficient: 0.5, r_squared: 0.25, offset_days: 50000)
      large_negative_offset = build(:stock_correlation, 
        correlation_coefficient: 0.5, r_squared: 0.25, offset_days: -50000)
      
      expect(large_positive_offset).to be_valid
      expect(large_negative_offset).to be_valid
    end
    
    it 'handles maximum window sizes' do
      max_windows = build(:stock_correlation, 
        correlation_coefficient: 0.5, r_squared: 0.25, 
        window1_days: 365, window2_days: 365)
      
      expect(max_windows).to be_valid
    end
  end
  
  describe 'relationships with CorrelationCalculationService' do
    it 'uses the same correlation ranges as the service' do
      correlation = build(:stock_correlation, correlation_coefficient: 0.75)
      
      # Should use the same ranges defined in CorrelationCalculationService
      expect(correlation.correlation_strength).to eq('Very strong correlation')
      
      # Verify it matches what the service would return
      service = CorrelationCalculationService.new
      service_result = service.interpret_correlation(0.75)
      expect(correlation.correlation_strength).to eq(service_result[:strength])
    end
  end
end