# Rake Task Reference

This document provides a comprehensive guide to all rake tasks available in the Stock Correlation Analysis Platform.

## Table of Contents

- [Correlation Analysis Tasks](#correlation-analysis-tasks)
- [Stock Information Tasks](#stock-information-tasks)
- [Data Cleanup Tasks](#data-cleanup-tasks)
- [Database Tasks](#database-tasks)
- [Development Tasks](#development-tasks)

## Correlation Analysis Tasks

### `correlation:analyze`

Perform correlation analysis between two stock symbols.

**Usage:**
```bash
rails correlation:analyze[symbol1,symbol2,window1,window2,offset,start_date,end_date]
```

**Parameters:**
- `symbol1` (required): First stock symbol (e.g., "AAPL")
- `symbol2` (required): Second stock symbol (e.g., "MSFT")
- `window1` (optional): Analysis window for symbol1 in days (default: 7)
- `window2` (optional): Analysis window for symbol2 in days (default: 7)
- `offset` (optional): Time offset in days (default: 0)
- `start_date` (optional): Analysis start date (YYYY-MM-DD)
- `end_date` (optional): Analysis end date (YYYY-MM-DD)

**Examples:**
```bash
# Basic correlation analysis
rails correlation:analyze[AAPL,MSFT]

# With custom windows
rails correlation:analyze[AAPL,MSFT,14,14]

# With offset for predictive analysis
rails correlation:analyze[AAPL,MSFT,7,7,14]

# With date range
rails correlation:analyze[AAPL,MSFT,7,7,0,2023-01-01,2024-01-01]
```

**Output:**
- Correlation coefficient and statistical significance
- R-squared value and data points analyzed
- Detailed analysis with multiple offset values
- Automatic saving of statistically relevant correlations

### `correlation:list`

Display saved correlations from the database.

**Usage:**
```bash
rails correlation:list[limit]
```

**Parameters:**
- `limit` (optional): Number of correlations to display (default: 10)

**Examples:**
```bash
# Show last 10 correlations
rails correlation:list

# Show last 50 correlations
rails correlation:list[50]
```

### `correlation:clear`

Remove all correlation data from the database.

**Usage:**
```bash
rails correlation:clear
```

**Warning:** This permanently deletes all correlation analysis results.

## Stock Information Tasks

### `stock_info:update_with_api`

Smart, resumable task for updating stock information using the FMP API.

**Usage:**
```bash
LIMIT=100 rails stock_info:update_with_api
```

**Environment Variables:**
- `LIMIT`: Number of symbols to process (default: 50)
- `FMP_API_KEY`: Financial Modeling Prep API key (required)

**Features:**
- Prioritizes missing symbols over placeholder data
- Automatic rate limiting and error handling
- Resumable - continues from where it left off
- Comprehensive progress tracking

**Examples:**
```bash
# Process 100 symbols
LIMIT=100 rails stock_info:update_with_api

# Process with API key
FMP_API_KEY=your_key LIMIT=50 rails stock_info:update_with_api
```

### `stock_info:update_missing`

Update symbols that are completely missing from stock_infos table.

**Usage:**
```bash
rails stock_info:update_missing
```

**Note:** This task only processes symbols with no stock_info record at all.

### `stock_info:test_update`

Test API integration with first 20 missing symbols.

**Usage:**
```bash
rails stock_info:test_update
```

**Purpose:** Validate API key and service functionality before large batch operations.

### `stock_info:batch_update`

Update stock information in controlled batches.

**Usage:**
```bash
LIMIT=100 rails stock_info:batch_update
```

**Parameters:**
- `LIMIT`: Number of symbols to process (default: 100)

### `stock_info:refresh_existing`

Refresh all existing stock information records.

**Usage:**
```bash
rails stock_info:refresh_existing
```

**Warning:** This updates ALL existing stock_info records and may take a long time.

### `stock_info:summary`

Display stock information coverage statistics.

**Usage:**
```bash
rails stock_info:summary
```

**Output:**
- Total symbols vs symbols with information
- Coverage percentage
- Sample of missing symbols

## Data Cleanup Tasks

### `data_cleanup:analyze`

Analyze data quality and show cleanup recommendations.

**Usage:**
```bash
rails data_cleanup:analyze
```

**Output:**
- Symbol count distribution by record count
- Quality statistics (average, median records per symbol)
- Symbols with placeholder information
- Cleanup recommendations

### `data_cleanup:mark_inactive`

Mark low-quality symbols as inactive based on various criteria.

**Usage:**
```bash
rails data_cleanup:mark_inactive
```

**Environment Variables:**
- `MIN_RECORDS`: Minimum record threshold (default: 50)
- `REQUIRE_STOCK_INFO`: Require stock info presence (default: true)
- `EXCLUDE_PLACEHOLDER`: Exclude placeholder stock info (default: true)
- `RECHECK_INACTIVE`: Recheck inactive symbols for reactivation (default: true)
- `DRY_RUN`: Show what would be done without making changes (default: false)

**Examples:**
```bash
# Mark symbols with < 100 records as inactive
MIN_RECORDS=100 rails data_cleanup:mark_inactive

# Dry run to see what would be changed
DRY_RUN=true rails data_cleanup:mark_inactive

# Skip placeholder exclusion
EXCLUDE_PLACEHOLDER=false rails data_cleanup:mark_inactive
```

### `data_cleanup:reactivate`

Reactivate previously inactive symbols.

**Usage:**
```bash
# Reactivate specific symbols
SYMBOLS='AAPL,MSFT,GOOGL' rails data_cleanup:reactivate

# Reactivate symbols meeting record threshold
MIN_RECORDS=100 rails data_cleanup:reactivate
```

### `data_cleanup:stats`

Show comprehensive data cleanup statistics.

**Usage:**
```bash
rails data_cleanup:stats
```

**Output:**
- Active vs inactive symbol counts
- Record statistics and percentages
- Stock info status breakdown

## Database Tasks

### Standard Rails Database Tasks

```bash
# Create database
rails db:create

# Run migrations
rails db:migrate

# Rollback migration
rails db:rollback

# Seed database
rails db:seed

# Reset database
rails db:reset

# Drop database
rails db:drop
```

### Custom Database Tasks

```bash
# Check database status
rails db:status

# Show pending migrations
rails db:migrate:status
```

## Development Tasks

### Environment Setup

```bash
# Install dependencies
bundle install

# Setup database from scratch
rails db:setup

# Generate migration
rails generate migration AddColumnToTable column:type
```

### Testing Tasks

```bash
# Run all tests
rails test

# Run specific test file
rails test test/models/stock_price_test.rb

# Run tests with coverage
COVERAGE=true rails test
```

## Task Chaining Examples

### Complete Setup for New Environment

```bash
# 1. Setup database
rails db:create db:migrate db:seed

# 2. Test API integration
rails stock_info:test_update

# 3. Populate stock information
LIMIT=100 rails stock_info:update_with_api

# 4. Analyze data quality
rails data_cleanup:analyze

# 5. Clean up low-quality data
MIN_RECORDS=50 rails data_cleanup:mark_inactive

# 6. Run correlation analysis
rails correlation:analyze[AAPL,MSFT]
```

### Daily Maintenance Routine

```bash
# 1. Update stock information
LIMIT=200 rails stock_info:update_with_api

# 2. Check data quality
rails data_cleanup:stats

# 3. Run correlation analysis on new data
rails correlation:analyze[SPY,QQQ]
```

## Performance Considerations

### Large Dataset Operations

**For tasks processing many symbols:**
- Use `LIMIT` parameters to control batch sizes
- Monitor memory usage during large operations
- Consider running during off-peak hours

**Parallel Processing:**
- Correlation analysis tasks use all available CPU cores
- Database operations are optimized with proper indexing
- API calls are rate-limited to respect service limits

### Memory Management

**High Memory Tasks:**
- `correlation:analyze` with large date ranges
- `stock_info:refresh_existing` (processes all symbols)
- `data_cleanup:mark_inactive` (analyzes all symbols)

**Memory Optimization:**
- Tasks process data in batches
- Database queries use efficient joins
- Garbage collection is triggered appropriately

## Error Handling

### Common Issues and Solutions

**API Rate Limits:**
```
Error: HTTP 429 - Too Many Requests
Solution: Wait for quota reset (usually daily) and re-run
```

**Missing API Key:**
```
Error: FMP_API_KEY environment variable not set
Solution: Set API key: export FMP_API_KEY=your_key_here
```

**Database Connection Issues:**
```
Error: Could not connect to database
Solution: Ensure PostgreSQL is running and database exists
```

**Insufficient Data:**
```
Error: Not enough data points for analysis
Solution: Ensure symbols have sufficient historical data
```

## Best Practices

### API Usage

1. **Set reasonable limits:** Use `LIMIT` parameters to avoid overwhelming the API
2. **Monitor quotas:** Check your FMP API usage regularly
3. **Handle failures gracefully:** Tasks are designed to resume from failures

### Data Management

1. **Regular cleanup:** Run `data_cleanup:analyze` periodically
2. **Monitor quality:** Use `data_cleanup:stats` to track data health
3. **Backup before major operations:** Always backup before running destructive tasks

### Performance

1. **Batch operations:** Use appropriate batch sizes for your system
2. **Off-peak processing:** Run heavy tasks during low-usage periods
3. **Monitor resources:** Watch CPU and memory usage during large operations