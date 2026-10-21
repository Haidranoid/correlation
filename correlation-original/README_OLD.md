# Stock Correlation Analysis System

A high-performance Ruby on Rails application for calculating and analyzing correlations between financial instruments. This system provides comprehensive statistical analysis of stock price relationships using parallel processing and advanced time series analysis techniques.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Statistical Methods](#statistical-methods)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Database Schema](#database-schema)
- [Performance Features](#performance-features)
- [API Reference](#api-reference)
- [Development](#development)

## Overview

### What This System Accomplishes

The Stock Correlation Analysis System performs sophisticated statistical analysis to:

1. **Discover Temporal Relationships**: Identify how one stock's performance correlates with another's, including lead/lag relationships
2. **Quantify Statistical Significance**: Filter results to only statistically meaningful correlations (|r| ≥ 0.3, R² ≥ 0.09)
3. **Support Multi-Timeframe Analysis**: Calculate correlations across different time windows (daily, weekly, monthly, etc.)
4. **Enable Predictive Analysis**: Discover offset patterns where one stock's movement predicts another's future movement
5. **Process Large Datasets Efficiently**: Handle millions of data points using parallel processing and optimized algorithms

### Real-World Applications

- **Portfolio Diversification**: Identify stocks with low correlation for risk reduction
- **Pairs Trading**: Find highly correlated stocks for trading strategies
- **Market Analysis**: Discover sector relationships and market leading indicators
- **Risk Management**: Quantify correlation risk in investment portfolios
- **Economic Research**: Study relationships between different market segments

## Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Rake Tasks    │    │  Service Layer  │    │   Data Layer    │
│                 │    │                 │    │                 │
│ • correlation:  │───▶│ • Correlation   │───▶│ • StockPrice    │
│   analyze       │    │   Analysis      │    │ • StockCorrel.  │
│ • correlation:  │    │ • StockData     │    │ • PostgreSQL    │
│   list          │    │ • Correlation   │    │   Indexes       │
│ • stock_data:   │    │   Calculation   │    │                 │
│   import_stooq  │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Parallel Processing Engine                    │
│  • Multi-core fork/join processes                               │
│  • Block-based offset testing                                   │
│  • JSON inter-process communication                             │
│  • Automatic CPU core detection                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Service Objects

#### CorrelationAnalysisService (Orchestrator)
**Responsibility**: Main workflow coordinator for correlation analysis

**Key Methods**:
- `analyze_correlation(symbol1, symbol2, window1, window2, offset, start_date, end_date)`: Single offset analysis
- `find_best_correlation(symbol1, symbol2, window1, window2, start_date, end_date)`: Comprehensive offset testing
- `save_relevant_correlations(results, symbol1, symbol2, window1, window2)`: Statistical filtering and database persistence

**Workflow**:
1. Fetch historical stock data for specified date ranges
2. Calculate percentage changes over specified time windows
3. Align data points temporally with offset adjustments
4. Perform correlation calculations
5. Filter for statistical significance
6. Store results in database

#### StockDataService (Data Processing)
**Responsibility**: Stock data fetching, transformation, and temporal alignment

**Key Methods**:
- `fetch_stock_data(symbol, start_date, end_date, window_days)`: Intelligent data fetching with buffer calculation
- `calculate_percentage_changes(stock_data, window_days)`: Rolling percentage change calculation
- `align_data_points(changes1, changes2, offset)`: Hash-based temporal alignment (O(n) complexity)
- `determine_max_date_range(symbol1, symbol2, window1, window2, offset)`: Smart date range optimization

**Performance Optimizations**:
- Hash-based data alignment for linear complexity
- Intelligent date range extension for window calculations
- Memory-efficient processing of large datasets

#### CorrelationCalculationService (Mathematical Engine)
**Responsibility**: High-performance correlation coefficient calculation

**Implementation Strategy**:
```ruby
# Inline C implementation for 10x+ performance improvement
class FastCorrelationCalculator
  inline(:C) do |builder|
    builder.c_raw <<-C
      static VALUE correlation_calculate(VALUE self, VALUE x_vals, VALUE y_vals) {
        // Optimized C implementation of Pearson correlation
        // Handles numerical stability and edge cases
      }
    C
  end
end
```

**Features**:
- Automatic fallback to Ruby implementation if C compilation fails
- Numerical stability handling (zero variance, NaN detection)
- Mathematical correctness verification through dual implementation

## Statistical Methods

### Pearson Correlation Coefficient

**Formula**: 
```
r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]
```

**Implementation Details**:
- Handles edge cases: identical values, zero variance, insufficient data
- Input validation: minimum 10 data points for statistical validity
- Output range validation: ensures -1 ≤ r ≤ 1

### Statistical Significance Filtering

**Criteria for "Statistically Relevant" Correlations**:
- **|correlation coefficient| ≥ 0.3**: Moderate correlation strength or higher (Cohen's guidelines)
- **R² ≥ 0.09**: At least 9% of variance explained (meaningful predictive power)
- **Minimum 10 data points**: Ensures statistical validity

**Correlation Strength Interpretation**:
- **0.0 - 0.1**: No correlation
- **0.1 - 0.3**: Weak correlation  
- **0.3 - 0.5**: Moderate correlation ⭐ (Saved)
- **0.5 - 0.7**: Strong correlation ⭐ (Saved)
- **0.7 - 0.9**: Very strong correlation ⭐ (Saved)
- **0.9 - 1.0**: Extremely strong correlation ⭐ (Saved)

### Time Series Analysis

**Window-Based Analysis**:
- Configurable time windows (1-365 days)
- Percentage change calculation over rolling windows
- Support for asymmetric windows (e.g., 7-day vs 30-day comparison)

**Offset Analysis**:
- Tests temporal relationships with configurable offsets
- Positive offsets: predictive relationships (stock A predicts stock B's future movement)
- Negative offsets: lagging relationships (stock A follows stock B's past movement)
- Automatic detection of optimal offset ranges based on data availability

## Installation & Setup

### Prerequisites

- Ruby 3.4.4+
- PostgreSQL 12+
- Rails 8.0+
- GCC (for inline C compilation)

### Installation

```bash
# Clone repository
git clone [repository-url]
cd correlation

# Install dependencies
bundle install

# Setup database
rails db:create
rails db:migrate

# Import stock data (optional - for testing)
rails stock_data:import_stooq_data[path/to/stooq_data.zip]
```

### Environment Configuration

```bash
# Required environment variables
export DATABASE_URL="postgresql://user:password@localhost/correlation_development"
export REDIS_URL="redis://localhost:6379/0"  # For background jobs (if used)
```

## Usage

### Basic Correlation Analysis

```bash
# Find all statistically relevant correlations between MSFT and AAPL
# Uses 7-day windows, tests all possible offsets
rails correlation:analyze[MSFT,AAPL,7,7]

# Specific offset analysis (no offset = simultaneous correlation)
rails correlation:analyze[MSFT,AAPL,7,7,0]

# Custom date range
rails correlation:analyze[MSFT,AAPL,7,7,0,2023-01-01,2024-01-01]

# Asymmetric windows (weekly vs monthly comparison)
rails correlation:analyze[MSFT,AAPL,7,30]
```

### Advanced Examples

```bash
# Predictive analysis: does MSFT predict AAPL's future performance?
rails correlation:analyze[MSFT,AAPL,7,7]  # Tests all offsets including predictive

# Sector correlation analysis
rails correlation:analyze[AAPL,GOOGL,14,14]  # Tech sector correlation

# Cross-timeframe analysis
rails correlation:analyze[SPY,QQQ,1,30]  # Daily SPY vs Monthly QQQ
```

### Viewing Results

```bash
# List top correlations in database
rails correlation:list[20]

# View all saved correlations
rails correlation:list
```

### Example Output

```
🎉 STATISTICALLY RELEVANT CORRELATIONS FOUND!
================================================================================
🔢 Best correlation coefficient: -0.5602 (offset: 14299 days)
📊 R²: 0.3139 (31.4% variance explained)
📈 Data points: 16
📅 Date range: 1986-03-20 to 1986-05-04
🔄 Total offsets tested: 14301
✅ Statistically relevant found: 5 (|r| ≥ 0.3, R² ≥ 0.09)

📊 ALL STATISTICALLY RELEVANT CORRELATIONS (will be saved to database):
================================================================================
Rank | Offset | Correlation | R²     | Data Points
--------------------------------------------------------------------------------
   1 |  14299 |  -0.5602 | 0.3139 |          16
   2 |  14295 |  -0.5516 | 0.3043 |          27
   3 |  14178 |   0.5418 | 0.2936 |          63
   4 |  14294 |  -0.5409 | 0.2926 |          32
   5 |  14285 |   0.5398 | 0.2914 |          22
================================================================================

✅ 5 statistically relevant correlation(s) analysis completed and saved to database
```

## Database Schema

### StockPrice Table
```sql
CREATE TABLE stock_prices (
  id BIGSERIAL PRIMARY KEY,
  symbol VARCHAR(10) NOT NULL,
  date DATE NOT NULL,
  open DECIMAL(10,2),
  high DECIMAL(10,2),
  low DECIMAL(10,2),
  close DECIMAL(10,2) NOT NULL,
  volume BIGINT,
  market VARCHAR(10),
  country VARCHAR(50),
  data_type VARCHAR(20) DEFAULT 'stocks',
  created_at TIMESTAMP(6) NOT NULL,
  updated_at TIMESTAMP(6) NOT NULL
);

-- Optimized indexes for query performance
CREATE UNIQUE INDEX index_stock_prices_on_symbol_market_date 
  ON stock_prices (symbol, market, date);
CREATE INDEX index_stock_prices_on_symbol ON stock_prices (symbol);
CREATE INDEX index_stock_prices_on_date ON stock_prices (date);
```

### StockCorrelation Table
```sql
CREATE TABLE stock_correlations (
  id BIGSERIAL PRIMARY KEY,
  symbol1 VARCHAR(10) NOT NULL,
  symbol2 VARCHAR(10) NOT NULL,
  window1_days INTEGER NOT NULL,
  window2_days INTEGER NOT NULL,
  offset_days INTEGER NOT NULL,
  correlation_coefficient DECIMAL(10,6) NOT NULL,
  r_squared DECIMAL(10,6) NOT NULL,
  start_date DATE NOT NULL,
  end_date DATE NOT NULL,
  data_points INTEGER NOT NULL,
  analysis_date TIMESTAMP(6) NOT NULL,
  created_at TIMESTAMP(6) NOT NULL,
  updated_at TIMESTAMP(6) NOT NULL
);

-- Prevent duplicate analyses
CREATE UNIQUE INDEX index_stock_correlations_unique 
  ON stock_correlations (symbol1, symbol2, window1_days, window2_days, offset_days);
```

## Performance Features

### Parallel Processing Architecture

**Design**: Clean implementation using the Parallel gem for maximum CPU utilization
- Automatically detects CPU core count using `Parallel.processor_count`
- Simple `Parallel.map` with `in_processes` for true multi-core parallelism
- Automatic database connection management per process
- Clean error handling with automatic nil filtering

**Performance Characteristics**:
- **CPU Utilization**: 100% multi-core usage
- **Memory Efficiency**: Each process handles individual offsets independently
- **Scalability**: Linear scaling with CPU core count
- **Simplicity**: Clean, maintainable code using proven Parallel gem
- **Fault Tolerance**: Individual process failures don't stop entire analysis

### Mathematical Optimizations

**Inline C Implementation**:
```ruby
# Automatic performance optimization
begin
  # Inline C for 10x+ speed improvement
  correlation = fast_correlation_calculator.calculate(x_values, y_values)
rescue => e
  # Automatic fallback to Ruby implementation
  correlation = ruby_correlation_fallback(x_values, y_values)
end
```

**Algorithm Complexity**:
- **Data Alignment**: O(n) hash-based temporal alignment
- **Correlation Calculation**: O(n) single-pass algorithm
- **Memory Usage**: Constant memory footprint regardless of dataset size

### Database Optimizations

**Query Performance**:
- Compound indexes on common query patterns
- Efficient date range queries with proper index utilization
- Batch operations with upsert logic for bulk imports

**Connection Management**:
- Proper connection handling in multi-process environment
- Connection pooling for concurrent operations
- Transaction management for data consistency

## API Reference

### CorrelationAnalysisService

#### `analyze_correlation(symbol1, symbol2, window1, window2, offset, start_date, end_date)`
Performs single offset correlation analysis.

**Parameters**:
- `symbol1` (String): First stock symbol
- `symbol2` (String): Second stock symbol  
- `window1` (Integer): Time window for first symbol (1-365 days)
- `window2` (Integer): Time window for second symbol (1-365 days)
- `offset` (Integer): Temporal offset in days
- `start_date` (Date): Analysis start date
- `end_date` (Date): Analysis end date

**Returns**: Hash with correlation results and statistical interpretation

#### `find_best_correlation(symbol1, symbol2, window1, window2, start_date, end_date)`
Finds all statistically relevant correlations across all possible offsets.

**Returns**: Hash with filtered results meeting statistical significance criteria

### StockCorrelation Model

#### Scopes
- `strong_correlations`: |correlation| ≥ 0.7
- `by_correlation_strength`: Ordered by correlation coefficient (descending)
- `recent`: Analyzed within last 30 days
- `by_symbol_pair(sym1, sym2)`: Filter by symbol pair

#### Methods
- `correlation_strength`: Returns human-readable strength description
- `correlation_direction`: Returns "Positive" or "Negative"
- `analysis_period_days`: Returns analysis period in days

## Available Rake Tasks

The system provides several rake tasks for correlation analysis and data management:

### Correlation Analysis Tasks

#### `correlation:analyze[symbol1,symbol2,window1,window2,offset,start_date,end_date]`
Calculate correlation between two stock symbols with different time windows and offset.

**Parameters**:
- `symbol1` (required): First stock symbol (e.g., AAPL)
- `symbol2` (required): Second stock symbol (e.g., MSFT)
- `window1` (required): Time window for first symbol in days (1-365)
- `window2` (required): Time window for second symbol in days (1-365)
- `offset` (optional): Temporal offset in days (omit to test all offsets)
- `start_date` (optional): Analysis start date (YYYY-MM-DD format)
- `end_date` (optional): Analysis end date (YYYY-MM-DD format)

**Examples**:
```bash
# Test all offsets between AAPL and MSFT with 7-day windows
rails correlation:analyze[AAPL,MSFT,7,7]

# Specific offset analysis
rails correlation:analyze[AAPL,MSFT,7,7,0,2023-01-01,2024-01-01]

# Asymmetric windows
rails correlation:analyze[AAPL,MSFT,7,30]
```

#### `correlation:list[limit]`
View saved correlations from database.

**Parameters**:
- `limit` (optional): Number of correlations to display (default: 10)

**Examples**:
```bash
# Show top 10 correlations
rails correlation:list

# Show top 50 correlations
rails correlation:list[50]
```

### Stock Data Management Tasks

#### `stock_data:import_stooq_data[file_path]`
Import bulk global stock data from STOOQ file (all countries).

**Parameters**:
- `file_path` (required): Path to the STOOQ data file

**Example**:
```bash
rails stock_data:import_stooq_data[/path/to/stooq_data.zip]
```

#### `stock_data:cleanup`
Clean up temporary files created during import processes.

**Example**:
```bash
rails stock_data:cleanup
```

### Database Tasks

Standard Rails database tasks are available:

- `db:create` - Create the database
- `db:migrate` - Run database migrations
- `db:seed` - Load seed data
- `db:reset` - Drop, recreate, and seed database

### Testing Tasks

- `spec` - Run all RSpec tests
- `spec:models` - Run model tests only
- `spec:services` - Run service tests only
- `spec:integration` - Run integration tests only
- `parallel:spec[count]` - Run tests in parallel with specified process count

### Parallel Database Tasks

- `parallel:create[count]` - Create test databases for parallel testing
- `parallel:prepare[count]` - Prepare test databases for parallel testing
- `parallel:drop[count]` - Drop test databases

## Development

### Testing

The system includes comprehensive test coverage for:
- Statistical accuracy verification
- Edge case handling (insufficient data, identical values)
- Performance benchmarking
- Database integrity constraints

### Code Quality

Following Rails best practices:
- Service object architecture for separation of concerns
- Comprehensive error handling with custom exception classes
- Detailed logging throughout analysis pipeline
- Input validation at multiple layers (database, model, service)

### Extending the System

**Adding New Statistical Methods**:
1. Implement in `CorrelationCalculationService`
2. Add corresponding interpretation logic
3. Update database schema if needed
4. Add rake task interface

**Performance Tuning**:
- Monitor database query performance
- Profile memory usage during large analyses
- Benchmark correlation calculation performance
- Optimize parallel processing block sizes

### Contributing

1. Follow Rails style guide and community best practices
2. Ensure comprehensive test coverage for new features
3. Document statistical methods and business logic
4. Validate performance impact of changes

---

## License

[Include your license information here]

## Contact

[Include contact information for questions and support]