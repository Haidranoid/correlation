require 'rails_helper'

RSpec.describe CorrelationCalculationService do
  let(:service) { described_class.new }
  
  describe '#calculate_pearson_correlation' do
    context 'with valid data points' do
      let(:perfectly_correlated_data) { [[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]] }
      let(:negatively_correlated_data) { [[1, 10], [2, 8], [3, 6], [4, 4], [5, 2]] }
      let(:uncorrelated_data) { [[1, 5], [2, 3], [3, 8], [4, 1], [5, 7]] }
      
      it 'calculates perfect positive correlation correctly' do
        result = service.calculate_pearson_correlation(perfectly_correlated_data)
        expect(result).to be_within(0.0001).of(1.0)
      end
      
      it 'calculates perfect negative correlation correctly' do
        result = service.calculate_pearson_correlation(negatively_correlated_data)
        expect(result).to be_within(0.0001).of(-1.0)
      end
      
      it 'calculates low correlation for uncorrelated data' do
        result = service.calculate_pearson_correlation(uncorrelated_data)
        expect(result.abs).to be < 0.5
      end
      
      it 'handles real-world financial data accurately' do
        # Simulate moderate positive correlation like MSFT-AAPL
        stock_data = [
          [0.05, 0.03], [0.02, 0.04], [-0.01, -0.02], [0.03, 0.02], [-0.02, -0.01],
          [0.04, 0.03], [0.01, 0.02], [-0.03, -0.02], [0.02, 0.01], [0.00, 0.01]
        ]
        
        result = service.calculate_pearson_correlation(stock_data)
        expect(result).to be_between(-1, 1)
        expect(result).to be > 0 # Should be positive correlation
      end
    end
    
    context 'with edge cases' do
      it 'returns nil for empty data' do
        result = service.calculate_pearson_correlation([])
        expect(result).to be_nil
      end
      
      it 'returns nil for single data point' do
        result = service.calculate_pearson_correlation([[1, 2]])
        expect(result).to be_nil
      end
      
      it 'handles zero variance in x values' do
        data = [[1, 2], [1, 4], [1, 6]]
        result = service.calculate_pearson_correlation(data)
        expect(result).to be_within(0.001).of(0.0)
      end
      
      it 'handles zero variance in y values' do
        data = [[1, 5], [2, 5], [3, 5]]
        result = service.calculate_pearson_correlation(data)
        expect(result).to be_within(0.001).of(0.0)
      end
      
      it 'handles mixed positive and negative values' do
        data = [[-5, -10], [-2, -4], [0, 0], [3, 6], [5, 10]]
        result = service.calculate_pearson_correlation(data)
        expect(result).to be_within(0.0001).of(1.0)
      end
    end
    
    context 'with malformed data' do
      it 'handles nil gracefully' do
        expect { service.calculate_pearson_correlation(nil) }.to raise_error(NoMethodError)
      end
      
      it 'handles non-array elements gracefully' do
        data = [[1, 2], "invalid", [3, 4]]
        expect { service.calculate_pearson_correlation(data) }.to raise_error(NoMethodError)
      end
    end
  end
  
  describe '#interpret_correlation' do
    context 'with various correlation strengths' do
      it 'interprets no correlation correctly' do
        result = service.interpret_correlation(0.05)
        expect(result[:strength]).to eq('No correlation')
        expect(result[:direction]).to eq('Positive')
        expect(result[:description]).to eq('Positive no correlation')
      end
      
      it 'interprets weak correlation correctly' do
        result = service.interpret_correlation(0.25)
        expect(result[:strength]).to eq('Weak correlation')
        expect(result[:direction]).to eq('Positive')
        expect(result[:r_squared]).to be_within(0.001).of(0.0625)
        expect(result[:variance_explained]).to be_within(0.1).of(6.2)
      end
      
      it 'interprets moderate correlation correctly' do
        result = service.interpret_correlation(0.4)
        expect(result[:strength]).to eq('Moderate correlation')
        expect(result[:r_squared]).to be_within(0.001).of(0.16)
        expect(result[:variance_explained]).to eq(16.0)
      end
      
      it 'interprets strong correlation correctly' do
        result = service.interpret_correlation(0.65)
        expect(result[:strength]).to eq('Strong correlation')
        expect(result[:r_squared]).to be_within(0.001).of(0.4225)
        expect(result[:variance_explained]).to be_within(0.1).of(42.2)
      end
      
      it 'interprets very strong correlation correctly' do
        result = service.interpret_correlation(0.85)
        expect(result[:strength]).to eq('Very strong correlation')
        expect(result[:direction]).to eq('Positive')
      end
      
      it 'interprets extremely strong correlation correctly' do
        result = service.interpret_correlation(0.95)
        expect(result[:strength]).to eq('Extremely strong correlation')
        expect(result[:r_squared]).to be_within(0.001).of(0.9025)
        expect(result[:variance_explained]).to be_within(0.1).of(90.2)
      end
      
      it 'interprets negative correlations correctly' do
        result = service.interpret_correlation(-0.7)
        expect(result[:strength]).to eq('Very strong correlation')
        expect(result[:direction]).to eq('Negative')
        expect(result[:description]).to eq('Negative very strong correlation')
      end
    end
    
    context 'with edge cases' do
      it 'handles perfect correlation' do
        result = service.interpret_correlation(1.0)
        expect(result[:strength]).to eq('Extremely strong correlation')
        expect(result[:r_squared]).to eq(1.0)
        expect(result[:variance_explained]).to eq(100.0)
      end
      
      it 'handles perfect negative correlation' do
        result = service.interpret_correlation(-1.0)
        expect(result[:strength]).to eq('Extremely strong correlation')
        expect(result[:direction]).to eq('Negative')
        expect(result[:r_squared]).to eq(1.0)
      end
      
      it 'handles zero correlation' do
        result = service.interpret_correlation(0.0)
        expect(result[:strength]).to eq('No correlation')
        expect(result[:r_squared]).to eq(0.0)
        expect(result[:variance_explained]).to eq(0.0)
      end
      
      it 'handles nil correlation' do
        result = service.interpret_correlation(nil)
        expect(result[:strength]).to eq('Invalid')
        expect(result[:direction]).to eq('None')
        expect(result[:description]).to eq('Invalid correlation value')
      end
      
      it 'handles out-of-range values' do
        result = service.interpret_correlation(1.5)
        expect(result[:strength]).to eq('Invalid correlation')
      end
    end
  end
  
  describe 'CORRELATION_RANGES constant' do
    it 'defines all expected correlation ranges' do
      expect(described_class::CORRELATION_RANGES.keys.size).to eq(6)
      expect(described_class::CORRELATION_RANGES.values).to include(
        'No correlation',
        'Weak correlation', 
        'Moderate correlation',
        'Strong correlation',
        'Very strong correlation',
        'Extremely strong correlation'
      )
    end
    
    it 'has non-overlapping ranges that cover 0 to 1' do
      ranges = described_class::CORRELATION_RANGES.keys
      
      # Check that ranges don't overlap by testing boundaries
      test_values = [0.05, 0.15, 0.35, 0.55, 0.75, 0.95]
      test_values.each do |value|
        matching_ranges = ranges.select { |r| r.cover?(value) }
        expect(matching_ranges.size).to eq(1), "Value #{value} should match exactly one range"
      end
      
      # Check coverage from 0 to 1
      expect(ranges.any? { |r| r.cover?(0) }).to be true
      expect(ranges.any? { |r| r.cover?(0.99) }).to be true
    end
  end
end